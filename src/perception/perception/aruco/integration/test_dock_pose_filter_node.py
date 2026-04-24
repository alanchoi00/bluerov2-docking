"""Integration test: dock_pose_filter node.

Publishes fused pose + sets up a static TF odom->camera_link, asserts
filtered pose and health appear on the output topics.
"""
import time
import threading

import pytest
import rclpy
import rclpy.parameter
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    TransformStamped,
)
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from interfaces.msg import FilterHealth

_BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_harness(FilterHealth):
    class _Harness(Node):
        def __init__(self):
            super().__init__("filter_test_harness")
            self._fused_pub = self.create_publisher(
                PoseWithCovarianceStamped, "/perception/aruco_dock_pose",
                _BEST_EFFORT_QOS,
            )
            self._filtered_received: list[PoseWithCovarianceStamped] = []
            self._health_received = []
            self.create_subscription(
                PoseWithCovarianceStamped,
                "/perception/dock_pose_filtered",
                lambda m: self._filtered_received.append(m),
                _BEST_EFFORT_QOS,
            )
            self.create_subscription(
                FilterHealth,
                "/perception/dock_pose_filtered/health",
                lambda m: self._health_received.append(m),
                _BEST_EFFORT_QOS,
            )
            self._tf = StaticTransformBroadcaster(self)
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "odom"
            tf.child_frame_id = "camera_link"
            tf.transform.rotation.w = 1.0
            self._tf.sendTransform(tf)

        def publish_fused(self, x: float, y: float, z: float, sigma: float = 0.01):
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.position.z = z
            msg.pose.pose.orientation.w = 1.0
            cov = [0.0] * 36
            for i in (0, 7, 14, 21, 28, 35):  # diagonal
                cov[i] = sigma ** 2
            msg.pose.covariance = cov
            self._fused_pub.publish(msg)

    return _Harness()


def test_initializes_on_first_fused_pose(ros_context):
    from perception.aruco.dock_pose_filter import DockPoseFilter
    node_under_test = DockPoseFilter()
    node_under_test.set_parameters([
        rclpy.parameter.Parameter(
            "healthy_max_position_std_m",
            rclpy.parameter.Parameter.Type.DOUBLE,
            0.1,
        )
    ])
    harness = _make_harness(FilterHealth)

    exec_ = SingleThreadedExecutor()
    exec_.add_node(node_under_test)
    exec_.add_node(harness)

    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [exec_.spin_once(timeout_sec=0.05) for _ in iter(lambda: not stop.is_set(), False)],
        daemon=True,
    )
    t.start()

    time.sleep(0.3)  # let TF propagate
    harness.publish_fused(1.0, 2.0, 3.0)
    time.sleep(0.1)
    harness.publish_fused(1.0, 2.0, 3.0)
    time.sleep(0.5)

    stop.set()
    t.join(timeout=1.0)

    assert len(harness._filtered_received) >= 1
    last = harness._filtered_received[-1]
    assert last.header.frame_id == "odom"
    assert abs(last.pose.pose.position.x - 1.0) < 0.1
    assert abs(last.pose.pose.position.y - 2.0) < 0.1
    assert abs(last.pose.pose.position.z - 3.0) < 0.1

    assert any(h.status == FilterHealth.HEALTHY for h in harness._health_received)

    node_under_test.destroy_node()
    harness.destroy_node()


def test_warming_up_before_any_input(ros_context):
    from perception.aruco.dock_pose_filter import DockPoseFilter
    node_under_test = DockPoseFilter()
    harness = _make_harness(FilterHealth)

    exec_ = SingleThreadedExecutor()
    exec_.add_node(node_under_test)
    exec_.add_node(harness)

    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [exec_.spin_once(timeout_sec=0.05) for _ in iter(lambda: not stop.is_set(), False)],
        daemon=True,
    )
    t.start()

    time.sleep(0.5)
    stop.set()
    t.join(timeout=1.0)

    for h in harness._health_received:
        assert h.status == FilterHealth.WARMING_UP

    node_under_test.destroy_node()
    harness.destroy_node()
