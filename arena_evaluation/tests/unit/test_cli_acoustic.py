"""Unit tests for the acoustic field CLI over the canonical Star Schema episode summary."""

from __future__ import annotations

import argparse
import pathlib

import polars as pl
import pytest

from arena_evaluation.cli_acoustic import (
    _acoustic_list,
    _episode_of_max_dose,
    _load_episode_summary,
    _map_for_episode,
    _resolve_episode,
    _handle_acoustic,
    setup_acoustic_subparsers,
)
from star_benchmark import write_benchmark


def _rows() -> list[dict[str, object]]:
    return [
        {"episode_id": 1, "planner": "dwb", "stage": "s1", "world": "map1", "success": True, "total_acoustic_dose_pa2s": 41.5},
        {"episode_id": 2, "planner": "dwb", "stage": "s1", "world": "map1", "success": False, "total_acoustic_dose_pa2s": 83.4},
        {"episode_id": 3, "planner": "dwb", "stage": "s1", "world": "map2", "success": True, "total_acoustic_dose_pa2s": None},
    ]



def _benchmark(tmp_path: pathlib.Path, rows: list[dict[str, object]] | None = None) -> pathlib.Path:
    return write_benchmark(tmp_path / "bench", rows or _rows())


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup_acoustic_subparsers(subparsers)
    return parser


def _acoustic_argv(command: str, benchmark_dir: pathlib.Path, *extra: str) -> argparse.Namespace:
    return _arg_parser().parse_args(["acoustic", command, "--benchmark-dir", str(benchmark_dir), *extra])


def test_load_episode_summary_reads_the_canonical_view(tmp_path: pathlib.Path) -> None:
    df = _load_episode_summary(_benchmark(tmp_path))
    assert df["episode_id"].to_list() == [1, 2, 3]
    assert "total_acoustic_dose_pa2s" in df.columns


def test_handle_acoustic_requires_star_schema_tables(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(_acoustic_argv("list", bench_dir))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "no Star Schema episode summary found" in out
    assert "Run 'evaluation process --benchmark-dir ...' first." in out


def test_handle_acoustic_missing_dir(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(_acoustic_argv("list", pathlib.Path("/nonexistent/dir")))
    assert exc.value.code == 1
    assert "Error: benchmark directory does not exist: /nonexistent/dir" in capsys.readouterr().out


def test_handle_acoustic_list_dispatch(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _handle_acoustic(_acoustic_argv("list", _benchmark(tmp_path)))
    assert "EPISODE" in capsys.readouterr().out


def test_map_for_episode_names_the_recorded_world() -> None:
    summary = pl.DataFrame({"episode_id": [1, 2], "world": ["map1", "map2"]}, schema={"episode_id": pl.Int64, "world": pl.String})
    assert _map_for_episode(summary, "episode_001") == "map1"
    assert _map_for_episode(summary, "episode_002") == "map2"
    assert _map_for_episode(summary, "episode_009") is None


def test_episode_of_max_dose_picks_the_loudest_episode(capsys: pytest.CaptureFixture[str]) -> None:
    summary = pl.DataFrame(
        {"episode_id": [1, 2, 3], "total_acoustic_dose_pa2s": [41.5, 83.4, None]},
        schema={"episode_id": pl.Int64, "total_acoustic_dose_pa2s": pl.Float64},
    )
    assert _episode_of_max_dose(summary, "worst") == "episode_002"
    assert "dose=83.40" in capsys.readouterr().out


def test_episode_of_max_dose_without_measurements(capsys: pytest.CaptureFixture[str]) -> None:
    summary = pl.DataFrame({"episode_id": [1], "total_acoustic_dose_pa2s": [None]}, schema={"episode_id": pl.Int64, "total_acoustic_dose_pa2s": pl.Float64})
    assert _episode_of_max_dose(summary, "worst") is None
    assert "has no values" in capsys.readouterr().out


def test_resolve_episode_accepts_a_recorded_episode(tmp_path: pathlib.Path) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "episode_002", tmp_path / "bench") == "episode_002"


def test_resolve_episode_rejects_an_absent_episode(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "episode_007", tmp_path / "bench") is None
    assert "not found in the episode summary" in capsys.readouterr().out


def test_resolve_episode_rejects_an_unknown_specifier(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "bogus", tmp_path / "bench") is None
    assert "unknown episode specifier 'bogus'" in capsys.readouterr().out


def test_resolve_episode_defaults_to_the_worst_dose(tmp_path: pathlib.Path) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "worst", tmp_path / "bench") == "episode_002"
    assert _resolve_episode(summary, "max-total", tmp_path / "bench") == "episode_002"


def test_acoustic_list_prints_dose_descending(capsys: pytest.CaptureFixture[str]) -> None:
    summary = pl.DataFrame(
        {"episode_id": [1, 2, 3], "total_acoustic_dose_pa2s": [41.5, 83.4, None], "success": [True, False, True]},
        schema={"episode_id": pl.Int64, "total_acoustic_dose_pa2s": pl.Float64, "success": pl.Boolean},
    )
    _acoustic_list(summary)
    out = capsys.readouterr().out
    assert "EPISODE" in out and "DOSE_PA2S" in out
    episodes = [line.split()[0] for line in out.splitlines() if line.startswith("episode_")]
    assert episodes == ["episode_002", "episode_001"]
    assert "83.40" in out and "41.50" in out


def test_acoustic_list_without_measurements_prints_only_the_header(capsys: pytest.CaptureFixture[str]) -> None:
    summary = pl.DataFrame({"episode_id": [1], "total_acoustic_dose_pa2s": [None], "success": [True]}, schema={"episode_id": pl.Int64, "total_acoustic_dose_pa2s": pl.Float64, "success": pl.Boolean})
    _acoustic_list(summary)
    out = capsys.readouterr().out
    assert "EPISODE" in out
    assert not [line for line in out.splitlines() if line.startswith("episode_")]


def test_acoustic_flags_keep_their_names() -> None:
    args = _arg_parser().parse_args(["acoustic", "animate", "--benchmark-dir", "/b", "--episode", "worst", "--fps", "5", "--max-frames", "10", "--format", "mp4", "--no-door-overlay"])
    assert (args.acoustic_command, args.episode, args.fps, args.max_frames, args.format, args.no_door_overlay) == ("animate", "worst", 5, 10, "mp4", True)
    snapshot = _arg_parser().parse_args(["acoustic", "snapshot", "--benchmark-dir", "/b", "--frame", "3"])
    assert (snapshot.acoustic_command, snapshot.frame) == ("snapshot", 3)