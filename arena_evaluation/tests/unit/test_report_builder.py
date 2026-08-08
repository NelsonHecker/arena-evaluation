import pathlib
import pytest
import tempfile
import polars as pl
from unittest import mock
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.storage.schemas import PlotSpec
from arena_evaluation.presentation.viz_manifest import VizManifest

def test_report_builder_jinja2_rendering():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        # Create dummy metrics dataframe
        df = pl.DataFrame({
            "planner": ["dwa", "mppi"],
            "success": [1.0, 0.5],
            "time_to_goal": [15.0, 20.0],
            "path_length": [10.0, 12.0],
            "collision_amount": [0.0, 1.0],
        })

        # Save dummy combined_metrics.parquet
        combined_path = tmp_path / "combined_metrics.parquet"
        df.write_parquet(combined_path)

        # Setup mock manifest
        manifest = VizManifest(plots=[
            PlotSpec(
                id="plot_1",
                type="violin",
                title="Path Length Distribution",
                data_key="path_length",
                layout_group="efficiency"
            ),
            PlotSpec(
                id="plot_2",
                type="bar",
                title="Average Collisions",
                data_key="collision_amount",
                layout_group="safety"
            ),
            PlotSpec(
                id="plot_radar",
                type="radar",
                title="Performance Overview",
                data_key="*",
                layout_group="overview"
            )
        ])

        with mock.patch("arena_evaluation.presentation.viz_manifest.VizManifest.load", return_value=manifest), \
             mock.patch("arena_evaluation.presentation.plotly_renderer.PlotlyRenderer.render", return_value="<div id='mock-plotly'></div>"), \
             mock.patch("arena_evaluation.presentation.seaborn_renderer.SeabornRenderer.render") as mock_seaborn:

            builder = ReportBuilder(benchmark_dir=tmp_path)
            builder.build()

            # Verify outputs
            report_file = tmp_path / "report.html"
            assert report_file.exists()
            assert (tmp_path / "plotly.min.js").exists()

            # Read report contents and verify structure
            content = report_file.read_text()
            assert "Arena Evaluation Report" in content
            assert "Performance Summary" in content
            assert "Performance Overview" in content
            assert "Efficiency Metrics" in content
            assert "Safety & Collision Metrics" in content
            assert "Path Length Distribution" in content
            assert "Average Collisions" in content
            assert "dwa" in content
            assert "mppi" in content
            assert "<div id='mock-plotly'></div>" in content




def test_report_builder_characterization_from_metrics():
    """A characterization manifest on the metrics frame derives the summary
    table and line plots from the per-episode timeseries_char_* list columns."""
    from arena_evaluation.presentation.viz_manifest import ManifestGroup, SummarySpec

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        # Metrics frame: one episode row with per-sample char columns.
        df = pl.DataFrame({
            "planner": ["characterization"],
            "episode": [1],
            "timeseries_char_time_s": [[0.0, 1.0, 2.0]],
            "timeseries_char_power_total_w": [[40.0, 60.0, 55.0]],
            "timeseries_char_phase_kind": [["idle", "linear", "linear"]],
            "timeseries_char_vx_target": [[0.0, 0.5, 1.0]],
        })
        df.write_parquet(tmp_path / "combined_metrics.parquet")

        manifest = VizManifest(
            name="characterization",
            title="Characterization Report",
            data_source="metrics",
            groups=[ManifestGroup(id="power_curves", title="Power vs. Velocity Curves")],
            summary=[SummarySpec(metric="timeseries_char_power_total_w", label="Mean Power", format="{:.1f}")],
            summary_group_by=["timeseries_char_phase_kind"],
            units={"timeseries_char_power_total_w": "W"},
            plots=[
                PlotSpec(
                    id="line_power_vs_vx",
                    type="line",
                    title="Power vs Velocity",
                    data_key="timeseries_char_vx_target",
                    group_by=["timeseries_char_phase_kind"],
                    options={"y": "timeseries_char_power_total_w", "aggregate": True, "mode": "lines+markers"},
                    layout_group="power_curves",
                ),
            ],
        )

        builder = ReportBuilder(benchmark_dir=tmp_path, manifest=manifest)
        builder.build()

        content = (tmp_path / "report.html").read_text()
        assert "Characterization Report" in content
        assert "Power vs. Velocity Curves" in content
        assert "Mean Power" in content      # derived summary column label
        assert "W" in content               # unit-suffixed axis label
        note = (tmp_path / "report_manifest.yaml").read_text()
        assert "characterization" in note
