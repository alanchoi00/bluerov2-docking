"""Integration test: coarse node self-gates cmd_vel on /docking/state."""

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

from interfaces.msg import CoarseApproachStatus, FilterHealth, DockingState

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

    def feed(self, state, dock_x=3.0):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.pose.pose.position.x = dock_x
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


def test_latch_resets_after_phase_gate(ros_context):
    """Regression: latch must clear when coarse is gated off and reset on re-entry.

    Geometry: dock at (3, 0, 0) with identity orientation -> standoff is at
    (3, -1, 0) in map (dock -Y axis). ROV placed exactly at standoff, rotated
    90 deg around Z so +x body faces world +y (towards the dock). This puts
    range_to_standoff=0, axis_offset=0, yaw_err=0, satisfying all tolerances.
    After ready_debounce_cycles ticks the latch fires (ready_for_handoff=True).
    Then FINE gates it off for several ticks. On COARSE re-entry the latch must
    have been cleared, so the first status messages are ready_for_handoff=False
    until the debounce accumulates again.
    """
    from control.coarse_approach_node import CoarseApproach

    node = CoarseApproach(parameter_overrides=_load_params())

    # ROV at standoff: (3, -1, 0), oriented 90 deg around Z (facing world +y).
    qz = math.sin(math.pi / 4)
    qw = math.cos(math.pi / 4)

    class _LatchHarness(Node):
        def __init__(self):
            super().__init__("latch_harness")
            self._pose_pub = self.create_publisher(
                PoseWithCovarianceStamped, "/perception/dock_pose_filtered", _RELIABLE
            )
            self._health_pub = self.create_publisher(
                FilterHealth, "/perception/dock_pose_filtered/health", _RELIABLE
            )
            self._state_pub = self.create_publisher(DockingState, "/docking/state", _RELIABLE)
            self.status: list[CoarseApproachStatus] = []
            self.create_subscription(
                CoarseApproachStatus,
                "/control/coarse_approach/status",
                self.status.append,
                _RELIABLE,
            )
            self._tf = StaticTransformBroadcaster(self)
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "map"
            tf.child_frame_id = "base_link"
            tf.transform.translation.x = 3.0
            tf.transform.translation.y = -1.0
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf.sendTransform(tf)

        def publish(self, state):
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

    harness = _LatchHarness()

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(harness)
    stop = threading.Event()
    spin_thread = threading.Thread(
        target=lambda: [
            ex.spin_once(timeout_sec=0.02)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    spin_thread.start()

    # Phase 1: run COARSE long enough for ready_debounce_cycles (10) to elapse.
    # 40 iterations at 0.05 s = 2 s >> 10/20 Hz = 0.5 s.
    for _ in range(40):
        harness.publish(DockingState.COARSE)
        time.sleep(0.05)

    assert any(s.ready_for_handoff for s in harness.status), \
        "latch must have fired at standoff before testing reset"

    # Phase 2: gate off with FINE for several ticks (clears latch via gate early-return).
    harness.status.clear()
    for _ in range(10):
        harness.publish(DockingState.FINE)
        time.sleep(0.05)

    # Phase 3: re-enter COARSE; collect a short burst of status messages.
    harness.status.clear()
    for _ in range(6):
        harness.publish(DockingState.COARSE)
        time.sleep(0.05)

    stop.set()
    spin_thread.join(timeout=1.0)

    # The first status messages must not have a stale True latch. The latch
    # needs to re-accumulate ready_debounce_cycles (10) ticks, so none of the
    # first 6 messages can be ready_for_handoff=True.
    assert harness.status, "expected status messages after COARSE re-entry"
    assert not any(s.ready_for_handoff for s in harness.status), (
        "latch must be reset on phase gate; ready_for_handoff must start False "
        "and require ready_debounce_cycles ticks before re-latching"
    )

    node.destroy_node()
    harness.destroy_node()
