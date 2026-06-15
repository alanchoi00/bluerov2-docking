"""Integration test: coarse_approach node.

Publishes synthetic dock_pose_filtered + health + a static TF map->base_link,
asserts cmd_vel and CoarseApproachStatus behave per phase."""

import math
import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from interfaces.msg import FilterHealth, CoarseApproachStatus

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
        super().__init__("coarse_test_harness")
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/perception/dock_pose_filtered", _RELIABLE
        )
        self._health_pub = self.create_publisher(
            FilterHealth, "/perception/dock_pose_filtered/health", _RELIABLE
        )
        self.cmds: list[Twist] = []
        self.status: list[CoarseApproachStatus] = []
        self.create_subscription(Twist, "/cmd_vel", self.cmds.append, _RELIABLE)
        self.create_subscription(
            CoarseApproachStatus,
            "/control/coarse_approach/status",
            self.status.append,
            _RELIABLE,
        )
        self._tf = StaticTransformBroadcaster(self)

    def send_tf(self, x, y, z, qx, qy, qz, qw):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = float(z)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        self._tf.sendTransform(tf)

    def publish_dock(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        m = PoseWithCovarianceStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.pose.pose.position.x = float(x)
        m.pose.pose.position.y = float(y)
        m.pose.pose.position.z = float(z)
        m.pose.pose.orientation.x = float(qx)
        m.pose.pose.orientation.y = float(qy)
        m.pose.pose.orientation.z = float(qz)
        m.pose.pose.orientation.w = float(qw)
        self._pose_pub.publish(m)

    def publish_health(self, status):
        h = FilterHealth()
        h.header.stamp = self.get_clock().now().to_msg()
        h.status = status
        self._health_pub.publish(h)


def _spin(*nodes, seconds):
    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [
            ex.spin_once(timeout_sec=0.02)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    t.start()
    time.sleep(seconds)
    stop.set()
    t.join(timeout=1.0)


def _load_params():
    """Load coarse_pbvs.yaml (the single source of truth) as parameter overrides
    so the node -- which declares parameters by type with no in-code defaults --
    can be constructed in tests."""
    import os

    import yaml
    from ament_index_python.packages import get_package_share_directory
    from rclpy.parameter import Parameter

    cfg = os.path.join(
        get_package_share_directory("control"), "config", "coarse_pbvs.yaml"
    )
    with open(cfg) as f:
        values = yaml.safe_load(f)["coarse_approach"]["ros__parameters"]
    return [
        Parameter(name=k, value=v) for k, v in values.items() if k != "use_sim_time"
    ]


def test_blocked_and_zero_cmd_before_any_pose(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    _spin(node, harness, seconds=1.0)

    assert harness.cmds, "expected cmd_vel to be published on the timer"
    last = harness.cmds[-1]
    assert last.linear.x == 0.0 and last.angular.z == 0.0
    assert harness.status and harness.status[-1].phase == CoarseApproachStatus.BLOCKED

    node.destroy_node()
    harness.destroy_node()


def _params_overrides():
    import rclpy.parameter

    P = rclpy.parameter.Parameter
    return [
        P("standoff_distance_m", P.Type.DOUBLE, 1.0),
        P("ready_debounce_cycles", P.Type.INTEGER, 2),
    ]


def _spin_while_feeding(node, harness, feed, *, iterations, period):
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
        feed()
        time.sleep(period)
    stop.set()
    t.join(timeout=1.0)


def test_approaches_when_healthy_and_off_target(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    node.set_parameters(_params_overrides())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)

    def feed():
        harness.publish_dock(3.0, 0.0, 0.0)
        harness.publish_health(FilterHealth.HEALTHY)

    _spin_while_feeding(node, harness, feed, iterations=20, period=0.05)

    moving = [c for c in harness.cmds if c.linear.x > 0.0]
    assert moving, "expected positive surge toward the dock"
    assert any(s.phase == CoarseApproachStatus.APPROACHING for s in harness.status)
    node.destroy_node()
    harness.destroy_node()


def test_blocks_on_stale_health(ros_context):
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())
    harness = _Harness()
    harness.send_tf(0, 0, 0, 0, 0, 0, 1)

    def feed():
        harness.publish_dock(3.0, 0.0, 0.0)
        harness.publish_health(FilterHealth.STALE)

    _spin_while_feeding(node, harness, feed, iterations=15, period=0.05)

    assert harness.cmds, "expected cmd_vel to be published on the timer"
    assert harness.cmds[-1].linear.x == 0.0
    assert harness.status[-1].phase == CoarseApproachStatus.BLOCKED
    node.destroy_node()
    harness.destroy_node()
