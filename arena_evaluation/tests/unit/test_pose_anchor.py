import pathlib

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pose_anchor import (
    anchor_odom,
    classify,
    fit_rigid,
    longest_segment,
    pair_samples,
    resolve_pose_source,
)
from arena_evaluation.storage.schemas import TopicBundle

DT_NS = 33_333_333


def _odom(x: np.ndarray, y: np.ndarray, yaw: np.ndarray, t: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"time_ns": t, "stamp_ns": t, "pos_x": x, "pos_y": y, "yaw": yaw, "vel_linear": np.zeros_like(x), "vel_angular": np.zeros_like(x)})


def _straight(n: int = 100, length: float = 10.0, t0: int = 0) -> pl.DataFrame:
    t = (np.arange(n) * DT_NS + t0).astype(np.int64)
    x = np.linspace(0.0, length, n)
    return _odom(x, np.zeros(n), np.zeros(n), t)


def _transform(odom: pl.DataFrame, rows: list[int], theta: float, tx: float, ty: float) -> pl.DataFrame:
    """Ground-truth samples built from the given odom rows through a known rigid transform."""
    sub = odom[rows]
    x = sub["pos_x"].to_numpy()
    y = sub["pos_y"].to_numpy()
    c, s = np.cos(theta), np.sin(theta)
    return pl.DataFrame(
        {
            "time_ns": sub["time_ns"],
            "stamp_ns_gt": sub["stamp_ns"],
            "pos_x_gt": tx + c * x - s * y,
            "pos_y_gt": ty + s * x + c * y,
            "yaw_gt": sub["yaw"].to_numpy() + theta,
            "frame_id": pl.Series("frame_id", ["env_0/jackal/odom"] * len(sub), dtype=pl.String),
        }
    )


def _bundle(odom: pl.DataFrame, tf_gt: pl.DataFrame | None) -> TopicBundle:
    return TopicBundle(odom=odom, tf_gt=tf_gt)


def test_one_pair_recovers_the_transform_exactly():
    odom = _straight()
    gt = _transform(odom, [10], 0.7, 3.0, -2.0)
    fit = fit_rigid(pair_samples(odom, gt, 0, len(odom)))
    assert fit.n == 1
    assert fit.theta == pytest.approx(0.7)
    assert (fit.tx, fit.ty) == pytest.approx((3.0, -2.0))
    assert fit.residual_m == 0.0


def test_noisy_pairs_recover_the_transform_to_under_a_centimetre():
    odom = _straight()
    gt = _transform(odom, [5, 50, 95], -1.2, 12.0, 4.5)
    noise = pl.Series("noise", [0.004, -0.003, 0.005])
    gt = gt.with_columns((pl.col("pos_x_gt") + noise).alias("pos_x_gt"))
    fit = fit_rigid(pair_samples(odom, gt, 0, len(odom)))
    assert fit.n == 3
    assert fit.theta == pytest.approx(-1.2, abs=1e-3)
    assert (fit.tx, fit.ty) == pytest.approx((12.0, 4.5), abs=0.01)
    assert 0.0 < fit.residual_m < 0.01


def test_pairs_are_matched_on_header_stamps_within_tolerance():
    # A sample stamped past the end of the track has no odom row within tolerance.
    odom = _straight()
    gt = _transform(odom, [99], 0.0, 0.0, 0.0)
    off = gt.with_columns((pl.col("stamp_ns_gt") + 400_000_000).alias("stamp_ns_gt"))
    assert len(pair_samples(odom, off, 0, len(odom))) == 0
    assert len(pair_samples(odom, gt.with_columns((pl.col("stamp_ns_gt") + 40_000_000).alias("stamp_ns_gt")), 0, len(odom))) == 1


def _teleported() -> tuple[pl.DataFrame, int]:
    """Odom with a short discarded segment, a 1 s gap and 20 m jump, then the surviving segment."""
    first = _straight(n=5, length=0.5)
    second = _straight(n=100, length=10.0, t0=int(first["time_ns"][-1]) + 1_000_000_000)
    second = second.with_columns((pl.col("pos_x") + 20.0).alias("pos_x"))
    return pl.concat([first, second]), len(first)


def test_longest_segment_drops_the_short_side_of_a_teleport():
    odom, split = _teleported()
    assert longest_segment(odom) == (split, len(odom))


def test_ground_truth_only_in_the_discarded_segment_falls_back_to_odom():
    odom, split = _teleported()
    gt = _transform(odom, [1], 0.4, 1.0, 1.0)
    frame, source = resolve_pose_source(_bundle(odom, gt), (None, None))
    assert frame is None
    assert (source.kind, source.samples, source.residual_m) == ("odom", 0, None)


def test_a_sample_in_the_discarded_segment_is_not_fitted():
    odom, split = _teleported()
    gt = pl.concat([_transform(odom, [1], 0.4, 1.0, 1.0), _transform(odom, [split + 40], 0.4, 1.0, 1.0)])
    pairs = pair_samples(odom, gt, *longest_segment(odom))
    assert len(pairs) == 1
    assert int(pairs["stamp_ns"][0]) == int(odom["stamp_ns"][split + 40])


def test_pairs_from_two_transforms_keep_the_raw_sparse_stream():
    odom = _straight()
    gt = pl.concat([_transform(odom, [10], 0.7, 3.0, -2.0), _transform(odom, [90], 0.7, 3.0, 4.0)])
    frame, source = resolve_pose_source(_bundle(odom, gt), (None, None))
    assert source.kind == "tf_sparse"
    assert source.samples == 2
    assert source.residual_m > 0.5
    assert frame.equals(gt)


def test_sparse_ground_truth_is_anchored_over_the_whole_odom_track():
    odom = _straight()
    gt = _transform(odom, [10, 90], 0.7, 3.0, -2.0)
    frame, source = resolve_pose_source(_bundle(odom, gt), (None, None))
    assert (source.kind, source.samples) == ("anchored", 2)
    assert source.residual_m == pytest.approx(0.0, abs=1e-9)
    assert len(frame) == len(odom)
    assert frame["frame_id"].unique().to_list() == ["anchored"]
    assert frame["pos_x_gt"][10] == pytest.approx(gt["pos_x_gt"][0])
    assert frame["pos_y_gt"][10] == pytest.approx(gt["pos_y_gt"][0])


def test_a_single_sample_anchors_without_a_residual():
    odom = _straight()
    frame, source = resolve_pose_source(_bundle(odom, _transform(odom, [10], 0.7, 3.0, -2.0)), (None, None))
    assert (source.kind, source.samples, source.residual_m) == ("anchored", 1, None)
    assert len(frame) == len(odom)


def test_the_episode_window_bounds_the_samples_that_are_fitted():
    odom = _straight()
    gt = pl.concat([_transform(odom, [10], 0.7, 3.0, -2.0), _transform(odom, [90], 2.0, 30.0, 30.0)])
    window = (int(odom["time_ns"][0]), int(odom["time_ns"][50]))
    _, source = resolve_pose_source(_bundle(odom, gt), window)
    assert (source.kind, source.samples, source.residual_m) == ("anchored", 1, None)


def test_a_dense_stream_is_passed_through_unchanged():
    odom = _straight()
    gt = _transform(odom, list(range(100)), 0.7, 3.0, -2.0)
    frame, source = resolve_pose_source(_bundle(odom, gt), (None, None))
    assert (source.kind, source.samples, source.residual_m) == ("tf", 100, None)
    assert frame.equals(gt)


def test_an_anchored_stream_reclassifies_as_dense():
    odom = _straight()
    frame, _ = resolve_pose_source(_bundle(odom, _transform(odom, [10, 90], 0.7, 3.0, -2.0)), (None, None))
    again, source = resolve_pose_source(_bundle(odom, frame), (None, None))
    assert source.kind == "tf"
    assert again.equals(frame)


def test_classify_reads_the_header_stamps_not_the_log_time():
    odom = _straight()
    dense = _transform(odom, list(range(100)), 0.0, 0.0, 0.0)
    assert classify(dense) == "dense"
    assert classify(dense.head(1)) == "sparse"
    assert classify(None) == "absent"
    assert classify(dense.head(0)) == "absent"
    # Log time every 33 ms, header stamps 1 s apart: sparse.
    stamped = dense.head(4).with_columns(pl.Series("stamp_ns_gt", [0, 1_000_000_000, 2_000_000_000, 3_000_000_000]))
    assert classify(stamped) == "sparse"


def test_anchored_yaw_wraps_into_the_principal_branch():
    n = 4
    t = (np.arange(n) * DT_NS).astype(np.int64)
    odom = _odom(np.zeros(n), np.zeros(n), np.full(n, np.pi - 0.1), t)
    frame = anchor_odom(odom, fit_rigid(pair_samples(odom, _transform(odom, [0], 0.5, 0.0, 0.0), 0, n)))
    yaw = frame["yaw_gt"].to_numpy()
    assert np.all(yaw >= -np.pi)
    assert np.all(yaw < np.pi)
    assert yaw[0] == pytest.approx(np.pi - 0.1 + 0.5 - 2 * np.pi)


def test_anchored_frame_round_trips_through_the_topic_store(tmp_path: pathlib.Path) -> None:
    odom = _straight()
    frame, _ = resolve_pose_source(_bundle(odom, _transform(odom, [10, 90], 0.7, 3.0, -2.0)), (None, None))
    TopicParquetStore.write({"env_0_jackal": TopicBundle(odom=odom, tf_gt=frame)}, tmp_path, overwrite=True)
    bundles = TopicParquetStore.read(tmp_path)
    read_back = bundles["env_0_jackal"].tf_gt.collect()
    assert read_back.schema == frame.schema
    assert read_back.equals(frame)


def test_frames_without_header_stamps_fall_back_to_log_time() -> None:
    gt = pl.DataFrame({"time_ns": [0, 1, 2], "pos_x_gt": [1.0, 1.5, 2.0], "pos_y_gt": [0.0, 0.0, 0.0]})
    assert classify(gt) == "dense"
    bundle = TopicBundle(odom=pl.DataFrame({"time_ns": [0, 1, 2], "pos_x": [0.0, 0.5, 1.0], "pos_y": [0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0]}), tf_gt=gt)
    frame, source = resolve_pose_source(bundle, (None, None))
    assert source.kind == "tf"
    assert frame.equals(gt)


def test_sparse_stream_without_yaw_is_kept_raw() -> None:
    gt = pl.DataFrame({"time_ns": [0, 5_000_000_000], "pos_x_gt": [1.0, 6.0], "pos_y_gt": [0.0, 0.0]})
    odom = pl.DataFrame({"time_ns": [0, 5_000_000_000], "pos_x": [0.0, 5.0], "pos_y": [0.0, 0.0], "yaw": [0.0, 0.0]})
    frame, source = resolve_pose_source(TopicBundle(odom=odom, tf_gt=gt), (None, None))
    assert source.kind == "tf_sparse"
    assert frame.equals(gt)
