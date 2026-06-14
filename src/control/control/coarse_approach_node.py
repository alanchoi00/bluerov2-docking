#!/usr/bin/env python3
"""coarse_approach: PBVS coarse-approach controller node.

Drives the BlueROV2 to a standoff point on the dock entry axis using the
filtered dock pose (#34). Publishes body-frame cmd_vel + CoarseApproachStatus.
Fixed-rate timer always emits a command (zero when BLOCKED) so ardusub_bridge
never re-sends a stale command."""

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener, TransformException

from control.pbvs import CoarsePbvsController, CoarsePbvsParams
from control import guidance as guidance_lib
from control import health_gate as hg
from interfaces.msg import FilterHealth, CoarseApproachStatus


class CoarseApproach(Node):
    def __init__(self):
        super().__init__("coarse_approach")

        # Fail fast if the pure mirrors drift from the generated message.
        assert hg.HEALTHY == FilterHealth.HEALTHY
        assert hg.STALE == FilterHealth.STALE
        assert hg.AT_STANDOFF == CoarseApproachStatus.AT_STANDOFF
        assert hg.BLOCKED == CoarseApproachStatus.BLOCKED

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("aim_offset_in_dock", [0.0, 0.310, 0.042])
        self.declare_parameter("standoff_distance_m", 1.0)
        self.declare_parameter("position_tol_m", 0.10)
        self.declare_parameter("axis_offset_tol_m", 0.10)
        self.declare_parameter("heading_tol_rad", 0.10)
        self.declare_parameter("ready_debounce_cycles", 10)
        self.declare_parameter("degraded_gain_scale", 0.5)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_pose_age_s", 1.0)
        for name, default in (
            ("kp_surge", 1.0), ("kp_sway", 0.8), ("kd_sway", 0.3),
            ("kp_heave", 0.8), ("kd_heave", 0.3), ("kp_yaw", 1.0), ("kd_yaw", 0.3),
            ("handoff_range_m", 0.0), ("surge_taper_range_m", 0.25),
            ("v_max_surge", 0.5), ("v_max_sway", 0.3),
            ("v_max_heave", 0.3), ("v_max_yaw", 0.5),
        ):
            self.declare_parameter(name, default)

        self._controller = CoarsePbvsController(self._params())
        self._ready_counter = 0
        self._latest_pose: PoseWithCovarianceStamped | None = None
        self._latest_pose_t: float | None = None
        self._latest_health: int | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", qos)
        self._pub_status = self.create_publisher(
            CoarseApproachStatus, "/control/coarse_approach/status", qos
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/perception/dock_pose_filtered",
            self._on_pose,
            qos,
        )
        self.create_subscription(
            FilterHealth, "/perception/dock_pose_filtered/health", self._on_health, qos
        )

        rate = self.get_parameter("control_rate_hz").get_parameter_value().double_value
        self._dt = 1.0 / rate
        self.create_timer(self._dt, self._tick)
        self.get_logger().info("coarse_approach ready")

    def _params(self) -> CoarsePbvsParams:
        g = lambda n: self.get_parameter(n).get_parameter_value().double_value
        return CoarsePbvsParams(
            kp_surge=g("kp_surge"), kp_sway=g("kp_sway"), kd_sway=g("kd_sway"),
            kp_heave=g("kp_heave"), kd_heave=g("kd_heave"),
            kp_yaw=g("kp_yaw"), kd_yaw=g("kd_yaw"),
            handoff_range_m=g("handoff_range_m"),
            surge_taper_range_m=g("surge_taper_range_m"),
            v_max_surge=g("v_max_surge"), v_max_sway=g("v_max_sway"),
            v_max_heave=g("v_max_heave"), v_max_yaw=g("v_max_yaw"),
        )

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._latest_pose = msg
        self._latest_pose_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_health(self, msg: FilterHealth) -> None:
        self._latest_health = int(msg.status)

    def _publish_zero(self, phase: int) -> None:
        self._pub_cmd.publish(Twist())
        st = CoarseApproachStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.dock_healthy = False
        st.ready_for_handoff = False
        self._pub_status.publish(st)

    def _tick(self) -> None:
        # Task 8 fills in the regulating path. Until then: always BLOCKED + zero.
        self._controller.reset()
        self._ready_counter = 0
        self._publish_zero(CoarseApproachStatus.BLOCKED)


def main(args=None):
    rclpy.init(args=args)
    node = CoarseApproach()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
