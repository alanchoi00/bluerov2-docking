#!/usr/bin/env python3
"""docking_fsm: YASMIN orchestrator for the BlueROV2 docking mission.

Sequences COARSE -> FINE -> DOCKED. Has no velocity authority: it only
publishes the active phase on /docking/state (which the phase controllers
self-gate on) and drives flight mode / arming via VehicleIO. Transition policy
lives in the pure orchestrator.transitions helpers; this node is the ROS + YASMIN
wiring around them.

YASMIN 5.x API quick reference:
  from yasmin import State, StateMachine, Blackboard
  class MyState(State):
      def __init__(self): super().__init__(outcomes=["done"])
      def execute(self, blackboard) -> str: ...; return "done"
  sm = StateMachine(outcomes=["docked"])
  sm.add_state("NAME", MyState(), transitions={"done": "NEXT"})
  sm.set_start_state("NAME")
  outcome = sm(Blackboard())            # blocks until the machine finishes
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from yasmin import State, StateMachine, Blackboard
from yasmin_viewer import YasminViewerPub

from interfaces.msg import (
    CoarseApproachStatus,
    FineAlignStatus,
    FilterHealth,
    DockingState as DockingStateMsg,
)
from orchestrator import transitions as tr
from orchestrator.vehicle_io import VehicleIO


class Outcome:
    HANDOFF = "handoff"
    SEATED = "seated"
    DEMOTE = "demote"
    DONE = "done"
    FINISHED = "finished"


class StateName:
    COARSE = "coarse"
    FINE = "fine"
    DOCKED = "docked"


class DockingFSM(Node):
    def __init__(self, vehicle_io=None, **kwargs):
        super().__init__("docking_fsm", **kwargs)

        # Fail fast if the pure mirrors drift from the generated message.
        assert tr.WARMING_UP == FilterHealth.WARMING_UP
        assert tr.HEALTHY == FilterHealth.HEALTHY
        assert tr.DEGRADED == FilterHealth.DEGRADED
        assert tr.STALE == FilterHealth.STALE

        ptype = Parameter.Type
        self.declare_parameter("tick_rate_hz", ptype.DOUBLE)
        self.declare_parameter("loss_timeout_cycles", ptype.INTEGER)
        self.declare_parameter("drift_timeout_cycles", ptype.INTEGER)
        self.declare_parameter("demote_range_m", ptype.DOUBLE)
        self.declare_parameter("alt_hold_mode", ptype.STRING)

        # Injected in tests (a fake); a real MAVROS adapter otherwise.
        self.vehicle_io = vehicle_io if vehicle_io is not None else VehicleIO(self)

        # Latest telemetry, written by subscriptions, read by the FSM states.
        self._coarse_ready = False
        self._fine_seated = False
        self._fine_range = 0.0
        self._health = FilterHealth.HEALTHY

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_state = self.create_publisher(DockingStateMsg, "/docking/state", qos)
        self.create_subscription(
            CoarseApproachStatus,
            "/control/coarse_approach/status",
            self._on_coarse,
            qos,
        )
        self.create_subscription(
            FineAlignStatus, "/control/fine_align/status", self._on_fine, qos
        )
        self.create_subscription(
            FilterHealth, "/perception/dock_pose_filtered/health", self._on_health, qos
        )

        self.get_logger().info("docking_fsm ready")
        self._start_fsm()

    def publish_state(self, state: int, label: str) -> None:
        msg = DockingStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.label = label
        self._pub_state.publish(msg)

    def param_int(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def param_double(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def param_str(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _on_coarse(self, msg: CoarseApproachStatus) -> None:
        self._coarse_ready = bool(msg.ready_for_handoff)

    def _on_fine(self, msg: FineAlignStatus) -> None:
        self._fine_seated = bool(msg.seated)
        self._fine_range = float(msg.range_to_dock_m)

    def _on_health(self, msg: FilterHealth) -> None:
        self._health = int(msg.status)

    def _start_fsm(self) -> None:
        self._stop = threading.Event()
        sm = StateMachine(outcomes=[Outcome.FINISHED])
        sm.add_state(
            StateName.COARSE,
            CoarseState(self),
            transitions={Outcome.HANDOFF: StateName.FINE},
        )
        sm.add_state(
            StateName.FINE,
            FineState(self),
            transitions={
                Outcome.SEATED: StateName.DOCKED,
                Outcome.DEMOTE: StateName.COARSE,
            },
        )
        sm.add_state(
            StateName.DOCKED,
            DockedState(self),
            transitions={Outcome.DONE: Outcome.FINISHED},
        )
        sm.set_start_state(StateName.COARSE)

        def _run():
            # A state loop that exits on shutdown falls off to a None outcome;
            # swallow that here so the daemon thread ends quietly during teardown.
            try:
                sm(Blackboard())
            except Exception as exc:
                self.get_logger().debug(f"FSM thread ended: {exc}")

        self._fsm_thread = threading.Thread(target=_run, daemon=True)
        self._fsm_thread.start()
        YasminViewerPub(sm, "DOCKING", node=self)

    def destroy_node(self):
        # Stop the FSM thread before tearing down ROS objects: otherwise a state
        # loop can publish on a destroyed node during shutdown and segfault.
        if getattr(self, "_stop", None) is not None:
            self._stop.set()
        if getattr(self, "_fsm_thread", None) is not None:
            self._fsm_thread.join(timeout=2.0)
        super().destroy_node()


class CoarseState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=[Outcome.HANDOFF])
        self._node = node

    def execute(self, blackboard) -> str:  # type: ignore
        self._node.vehicle_io.set_mode(self._node.param_str("alt_hold_mode"))
        self._node.publish_state(DockingStateMsg.COARSE, "COARSE")
        period = 1.0 / self._node.param_double("tick_rate_hz")
        while rclpy.ok() and not self._node._stop.is_set():
            if self._node._coarse_ready:
                return Outcome.HANDOFF
            time.sleep(period)


class FineState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=["seated", "demote"])
        self._node = node

    def execute(self, blackboard) -> str:  # type: ignore
        self._node.publish_state(DockingStateMsg.FINE, "FINE")
        loss_counter = 0
        drift_counter = 0
        loss_to = self._node.param_int("loss_timeout_cycles")
        drift_to = self._node.param_int("drift_timeout_cycles")
        demote_range = self._node.param_double("demote_range_m")
        period = 1.0 / self._node.param_double("tick_rate_hz")
        while rclpy.ok() and not self._node._stop.is_set():
            if self._node._fine_seated:
                return Outcome.SEATED
            loss_counter, lost = tr.sustained(
                loss_counter, tr.is_loss(self._node._health), loss_to
            )
            drift_counter, drifted = tr.sustained(
                drift_counter,
                tr.is_drift(self._node._fine_range, demote_range),
                drift_to,
            )
            if lost or drifted:
                return Outcome.DEMOTE
            time.sleep(period)


class DockedState(State):
    def __init__(self, node: DockingFSM) -> None:
        super().__init__(outcomes=[Outcome.DONE])
        self._node = node

    def execute(self, blackboard) -> str:
        self._node.publish_state(DockingStateMsg.DOCKED, "DOCKED")
        self._node.vehicle_io.set_arm(False)
        return Outcome.DONE


def main(args=None):
    rclpy.init(args=args)
    node = DockingFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
