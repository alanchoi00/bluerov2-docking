"""Integration test: docking FSM transitions + side effects.

Uses a fake VehicleIO to capture set_mode/set_arm, drives the node with
synthetic status messages, asserts the /docking/state sequence."""

import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from interfaces.msg import (
    CoarseApproachStatus, FineAlignStatus, FilterHealth, DockingState
)

_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
)


class _FakeIO:
    def __init__(self):
        self.modes = []
        self.arms = []

    def set_mode(self, mode):
        self.modes.append(mode)

    def set_arm(self, arm):
        self.arms.append(arm)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Harness(Node):
    def __init__(self):
        super().__init__("fsm_test_harness")
        self._coarse = self.create_publisher(
            CoarseApproachStatus, "/control/coarse_approach/status", _RELIABLE)
        self._fine = self.create_publisher(
            FineAlignStatus, "/control/fine_align/status", _RELIABLE)
        self._health = self.create_publisher(
            FilterHealth, "/perception/dock_pose_filtered/health", _RELIABLE)
        self.states = []
        self.create_subscription(
            DockingState, "/docking/state", lambda m: self.states.append(m.state),
            _RELIABLE)
        from std_msgs.msg import Bool
        self._engaged_pub = self.create_publisher(Bool, "/docking/engaged", _RELIABLE)

    def coarse(self, ready, health=FilterHealth.HEALTHY):
        m = CoarseApproachStatus()
        m.ready_for_handoff = ready
        m.dock_healthy = True
        self._coarse.publish(m)
        h = FilterHealth(); h.status = health; self._health.publish(h)

    def fine(self, seated, range_m=0.2, health=FilterHealth.HEALTHY):
        m = FineAlignStatus()
        m.seated = seated
        m.range_to_dock_m = range_m
        m.dock_healthy = True
        self._fine.publish(m)
        h = FilterHealth(); h.status = health; self._health.publish(h)

    def engage(self, on: bool):
        from std_msgs.msg import Bool
        self._engaged_pub.publish(Bool(data=on))


def _load_params():
    import os, yaml
    from ament_index_python.packages import get_package_share_directory
    from rclpy.parameter import Parameter
    cfg = os.path.join(
        get_package_share_directory("orchestrator"), "config", "docking_fsm.yaml")
    with open(cfg) as f:
        values = yaml.safe_load(f)["docking_fsm"]["ros__parameters"]
    params = [Parameter(name=k, value=v) for k, v in values.items() if k != "use_sim_time"]
    # viewer's timer/publisher races in-process teardown; off for tests
    params.append(Parameter("enable_viewer", Parameter.Type.BOOL, False))
    return params


def _spin(node, harness, feed, *, iterations, period=0.05):
    ex = SingleThreadedExecutor()
    ex.add_node(node); ex.add_node(harness)
    stop = threading.Event()
    t = threading.Thread(target=lambda: [
        ex.spin_once(timeout_sec=0.02)
        for _ in iter(lambda: not stop.is_set(), False)], daemon=True)
    t.start()
    for i in range(iterations):
        feed(i)
        time.sleep(period)
    stop.set(); t.join(timeout=1.0)


def test_starts_idle_in_poshold(ros_context):
    from orchestrator.docking_fsm_node import DockingFSM
    io = _FakeIO()
    node = DockingFSM(vehicle_io=io, parameter_overrides=_load_params())
    harness = _Harness()
    _spin(node, harness, lambda i: harness.engage(False), iterations=10)
    assert harness.states and harness.states[0] == DockingState.IDLE
    assert "POSHOLD" in io.modes
    assert True in io.arms, "IDLE must re-arm so manual control works after DOCKED"
    node.destroy_node(); harness.destroy_node()


def test_engage_enters_coarse_althold(ros_context):
    from orchestrator.docking_fsm_node import DockingFSM
    io = _FakeIO()
    node = DockingFSM(vehicle_io=io, parameter_overrides=_load_params())
    harness = _Harness()
    _spin(node, harness, lambda i: (harness.engage(True), harness.coarse(ready=False)),
          iterations=15)
    assert DockingState.COARSE in harness.states
    assert "ALT_HOLD" in io.modes
    node.destroy_node(); harness.destroy_node()


def test_disengage_returns_to_idle(ros_context):
    from orchestrator.docking_fsm_node import DockingFSM
    io = _FakeIO()
    node = DockingFSM(vehicle_io=io, parameter_overrides=_load_params())
    harness = _Harness()

    def feed(i):
        harness.engage(i < 10)          # engage first, then release
        harness.coarse(ready=False)

    _spin(node, harness, feed, iterations=25)
    assert DockingState.COARSE in harness.states
    assert harness.states[-1] == DockingState.IDLE
    node.destroy_node(); harness.destroy_node()


def test_coarse_to_fine_on_handoff(ros_context):
    from orchestrator.docking_fsm_node import DockingFSM
    io = _FakeIO()
    node = DockingFSM(vehicle_io=io, parameter_overrides=_load_params())
    harness = _Harness()
    _spin(node, harness, lambda i: (harness.engage(True), harness.coarse(ready=True)),
          iterations=20)
    assert DockingState.FINE in harness.states
    # FINE drops the autopilot depth-hold so the controller owns the descent
    assert "STABILIZE" in io.modes
    node.destroy_node(); harness.destroy_node()


def test_fine_to_docked_disarms(ros_context):
    from orchestrator.docking_fsm_node import DockingFSM
    io = _FakeIO()
    node = DockingFSM(vehicle_io=io, parameter_overrides=_load_params())
    harness = _Harness()

    def feed(i):
        harness.engage(True)
        if i < 8:
            harness.coarse(ready=True)
        else:
            harness.fine(seated=True)

    _spin(node, harness, feed, iterations=30)
    assert DockingState.DOCKED in harness.states
    assert io.arms and io.arms[-1] is False
    node.destroy_node(); harness.destroy_node()


def test_demote_does_not_flap(ros_context):
    from orchestrator.docking_fsm_node import DockingFSM
    io = _FakeIO()
    node = DockingFSM(vehicle_io=io, parameter_overrides=_load_params())
    harness = _Harness()

    def feed(i):
        harness.engage(True)
        if i < 8:
            harness.coarse(ready=True)          # reach FINE
        else:
            harness.fine(seated=False, range_m=5.0)  # far away -> drift demote
        # deliberately publish NO fresh coarse ready after demote

    _spin(node, harness, feed, iterations=40)
    # after demote it must settle in COARSE, not flap back to FINE
    tail = harness.states[-5:]
    assert tail and all(s == DockingState.COARSE for s in tail), \
        f"expected to settle in COARSE, got {tail}"
    node.destroy_node(); harness.destroy_node()
