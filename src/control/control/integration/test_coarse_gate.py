"""Integration test: coarse node self-gates cmd_vel on /docking/state."""

import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from interfaces.msg import FilterHealth, DockingState

_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Harness(Node):
    def __init__(self):
        super().__init__("coarse_gate_harness")
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/perception/dock_pose_filtered", _RELIABLE
        )
        self._health_pub = self.create_publisher(
            FilterHealth, "/perception/dock_pose_filtered/health", _RELIABLE
        )
        self._state_pub = self.create_publisher(
            DockingState, "/docking/state", _RELIABLE
        )
        self.cmds: list[Twist] = []
        self.create_subscription(Twist, "/cmd_vel", self.cmds.append, _RELIABLE)
        self._tf = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "base_link"
        tf.transform.rotation.w = 1.0
        self._tf.sendTransform(tf)

    def feed(self, state):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.pose.pose.position.x = 3.0
        m.pose.pose.orientation.w = 1.0
        self._pose_pub.publish(m)
        h = FilterHealth()
        h.status = FilterHealth.HEALTHY
        self._health_pub.publish(h)
        s = DockingState()
        s.state = state
        self._state_pub.publish(s)


def _load_params():
    import os
    import yaml
    from ament_index_python.packages import get_package_share_directory
    from rclpy.parameter import Parameter

    cfg = os.path.join(
        get_package_share_directory("control"), "config", "coarse_pbvs.yaml"
    )
    with open(cfg) as f:
        values = yaml.safe_load(f)["coarse_approach"]["ros__parameters"]
    return [Parameter(name=k, value=v) for k, v in values.items() if k != "use_sim_time"]


def _run(node, harness, state, *, iterations=15, period=0.05):
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(harness)
    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [
            ex.spin_once(timeout_sec=0.02)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    t.start()
    for _ in range(iterations):
        harness.feed(state)
        time.sleep(period)
    stop.set()
    t.join(timeout=1.0)


def test_coarse_silent_when_fine_active(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    _run(node, harness, DockingState.FINE)
    assert harness.cmds == [], "coarse must not publish cmd_vel when FINE is active"
    node.destroy_node()
    harness.destroy_node()


def test_coarse_drives_when_coarse_active(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    _run(node, harness, DockingState.COARSE)
    assert any(c.linear.x > 0.0 for c in harness.cmds), "coarse should drive when active"
    node.destroy_node()
    harness.destroy_node()
