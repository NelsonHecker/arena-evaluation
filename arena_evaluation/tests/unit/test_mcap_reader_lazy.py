import pathlib
import pytest
import tempfile
import polars as pl
from unittest import mock
from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.storage.schemas import TopicBundle

# Helper mock classes to simulate ROS2 decoders
class MockChannel:
    def __init__(self, topic):
        self.topic = topic

class MockMessage:
    def __init__(self, log_time):
        self.log_time = log_time

class MockVector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockQuaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

class MockPose:
    def __init__(self, px=0.0, py=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.position = MockVector3(px, py)
        self.orientation = MockQuaternion(qx, qy, qz, qw)

class MockPoseWithCovariance:
    def __init__(self, px=0.0, py=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.pose = MockPose(px, py, qx, qy, qz, qw)

class MockTwist:
    def __init__(self, lx=0.0, az=0.0):
        self.linear = MockVector3(lx)
        self.angular = MockVector3(z=az)

class MockTwistWithCovariance:
    def __init__(self, lx=0.0, az=0.0):
        self.twist = MockTwist(lx, az)

class MockOdomMsg:
    def __init__(self, px=0.0, py=0.0, lx=0.0, az=0.0):
        self.pose = MockPoseWithCovariance(px, py)
        self.twist = MockTwistWithCovariance(lx, az)


def test_mcap_reader_lazy_chunking():
    # Setup temporary directory for output parquet files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        mcap_file = tmp_path / "dummy.mcap"
        mcap_file.touch()

        # Generate 10,005 mock odom messages to trigger a chunk flush (threshold 10,000)
        mock_messages = []
        for i in range(10005):
            channel = MockChannel("/env_0/odom")
            message = MockMessage(i * 1000)
            ros_msg = MockOdomMsg(px=float(i), py=float(i * 2), lx=1.0, az=0.5)
            mock_messages.append((None, channel, message, ros_msg))

        with mock.patch("mcap.reader.NonSeekingReader") as mock_reader_cls:
            mock_reader_inst = mock_reader_cls.return_value
            mock_reader_inst.iter_decoded_messages.return_value = mock_messages

            reader = MCAPReader(mcap_file)
            bundle = reader.read()

            # Verify that the Parquet file was created in the inferred topics folder
            odom_parquet = tmp_path / "topics" / "odom.parquet"
            assert odom_parquet.exists()

            # Verify it returns a TopicBundle of LazyFrames
            assert isinstance(bundle, TopicBundle)
            assert isinstance(bundle.odom, pl.LazyFrame)

            # Collect lazy frame and verify contents
            odom_df = bundle.odom.collect()
            assert len(odom_df) == 10005
            assert odom_df["pos_x"][0] == 0.0
            assert odom_df["pos_x"][10004] == 10004.0
            assert odom_df["pos_y"][10004] == 20008.0
            assert odom_df["time_ns"][10004] == 10004000


class MockTransformHeader:
    def __init__(self, frame_id):
        self.frame_id = frame_id

class MockTranslation:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockTransform:
    def __init__(self, tx, ty, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.translation = MockTranslation(tx, ty)
        self.rotation = MockQuaternion(qx, qy, qz, qw)

class MockStampedTransform:
    def __init__(self, frame_id, child_frame_id, tx, ty, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.header = MockTransformHeader(frame_id)
        self.child_frame_id = child_frame_id
        self.transform = MockTransform(tx, ty, qx, qy, qz, qw)

class MockTFMsg:
    def __init__(self, transforms):
        self.transforms = transforms


def test_mcap_reader_tf_gt_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        mcap_file = tmp_path / "dummy.mcap"
        mcap_file.touch()

        # Create a mock /tf message that matches world-to-base_link ground truth conditions
        transforms = [
            MockStampedTransform(
                frame_id="map",
                child_frame_id="env_0/base_link",
                tx=5.5,
                ty=10.0,
                qw=1.0
            )
        ]
        mock_messages = [
            (
                None,
                MockChannel("/tf"),
                MockMessage(999999),
                MockTFMsg(transforms)
            )
        ]

        with mock.patch("mcap.reader.NonSeekingReader") as mock_reader_cls:
            mock_reader_inst = mock_reader_cls.return_value
            mock_reader_inst.iter_decoded_messages.return_value = mock_messages

            reader = MCAPReader(mcap_file)
            bundle = reader.read()

            tf_gt_parquet = tmp_path / "topics" / "tf_gt.parquet"
            assert tf_gt_parquet.exists()

            assert bundle.tf_gt is not None
            assert isinstance(bundle.tf_gt, pl.LazyFrame)

            tf_gt_df = bundle.tf_gt.collect()
            assert len(tf_gt_df) == 1
            assert tf_gt_df["time_ns"][0] == 999999
            assert tf_gt_df["pos_x_gt"][0] == 5.5
            assert tf_gt_df["pos_y_gt"][0] == 10.0
            assert tf_gt_df["yaw_gt"][0] == 0.0

