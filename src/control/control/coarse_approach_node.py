#!/usr/bin/env python3
"""coarse_approach: PBVS coarse-approach controller node.

Drives the BlueROV2 to a standoff point on the dock entry axis using the
filtered dock pose (#34). Publishes body-frame cmd_vel + CoarseApproachStatus.
Fixed-rate timer always emits a command (zero when BLOCKED) so ardusub_bridge
never re-sends a stale command."""

import rclpy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

from control.pbvs import CoarsePbvsController, CoarsePbvsParams
from control import guidance as guidance_lib
from control import health_gate as hg
from interfaces.msg import FilterHealth, CoarseApproachStatus


class CoarseApproach(Node):
    def __init__(self):
        super().__init__("coarse_approach")

        # Fail fast if the pure mirrors drift from the generated message.
        assert hg.WARMING_UP == FilterHealth.WARMING_UP
        assert hg.HEALTHY == FilterHealth.HEALTHY
        assert hg.DEGRADED == FilterHealth.DEGRADED
        assert hg.STALE == FilterHealth.STALE
        assert hg.APPROACHING == CoarseApproachStatus.APPROACHING
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

        # Gains are snapshotted here at construction; changing a gain parameter
        # at runtime requires a node restart. Tolerances (see _tolerances) are
        # re-read live each tick so they can be tuned in-mission.
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

    def _tolerances(self) -> hg.Tolerances:
        gi = lambda n: self.get_parameter(n).get_parameter_value().integer_value
        gd = lambda n: self.get_parameter(n).get_parameter_value().double_value
        return hg.Tolerances(
            position_m=gd("position_tol_m"),
            axis_offset_m=gd("axis_offset_tol_m"),
            heading_rad=gd("heading_tol_rad"),
            debounce_cycles=gi("ready_debounce_cycles"),
        )

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._latest_pose = msg
        self._latest_pose_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_health(self, msg: FilterHealth) -> None:
        self._latest_health = int(msg.status)

    def _pose_too_old(self) -> bool:
        if self._latest_pose_t is None:
            return True
        age = self.get_clock().now().nanoseconds * 1e-9 - self._latest_pose_t
        max_age = self.get_parameter("max_pose_age_s").get_parameter_value().double_value
        # Negative age means the clock jumped backwards (e.g. sim reset): treat
        # the cached pose as untrustworthy and block until a fresh one arrives.
        return age < 0.0 or age > max_age

    def _publish_zero(self, phase: int) -> None:
        self._pub_cmd.publish(Twist())
        st = CoarseApproachStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.dock_healthy = False
        st.ready_for_handoff = False
        self._pub_status.publish(st)

    def _block(self) -> None:
        # reset() clears the PD derivative memory so a resumed approach does not
        # see a spurious error jump across the blocked gap.
        self._controller.reset()
        self._ready_counter = 0
        self._publish_zero(CoarseApproachStatus.BLOCKED)

    def _tick(self) -> None:
        if (
            self._latest_pose is None
            or self._latest_health is None
            or self._pose_too_old()
        ):
            self._block()
            return

        scale = (
            self.get_parameter("degraded_gain_scale")
            .get_parameter_value()
            .double_value
        )
        gate = hg.gate_for_health(self._latest_health, scale)
        if gate.blocked:
            self._block()
            return

        target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        try:
            # Time() = latest available transform; ROV TF staleness is
            # acceptable for coarse approach (avoids a measurement-stamp deadlock).
            tf = self._tf_buffer.lookup_transform(target_frame, "base_link", Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {target_frame}->base_link unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            self._block()
            return

        p = self._latest_pose.pose.pose
        rov = tf.transform
        aim_offset = (
            self.get_parameter("aim_offset_in_dock")
            .get_parameter_value()
            .double_array_value
        )
        standoff = (
            self.get_parameter("standoff_distance_m").get_parameter_value().double_value
        )

        g = guidance_lib.compute_guidance(
            dock_pos=(p.position.x, p.position.y, p.position.z),
            dock_quat_xyzw=(
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ),
            rov_pos=(rov.translation.x, rov.translation.y, rov.translation.z),
            rov_quat_xyzw=(
                rov.rotation.x,
                rov.rotation.y,
                rov.rotation.z,
                rov.rotation.w,
            ),
            aim_offset_in_dock=list(aim_offset),
            standoff_distance_m=standoff,
        )

        cmd = self._controller.step(g.rel_pos_body, g.yaw_err, self._dt)

        twist = Twist()
        twist.linear.x = cmd.surge * gate.gain_scale
        twist.linear.y = cmd.sway * gate.gain_scale
        twist.linear.z = cmd.heave * gate.gain_scale
        twist.angular.z = cmd.yaw_rate * gate.gain_scale
        self._pub_cmd.publish(twist)

        tol = self._tolerances()
        within_pos, within_head = hg.within_tolerances(
            g.range_to_standoff_m, g.axis_offset_m, g.yaw_err, tol
        )
        phase, ready, self._ready_counter = hg.decide_phase(
            blocked=False,
            within_pos=within_pos,
            within_head=within_head,
            healthy=gate.dock_healthy,
            ready_counter=self._ready_counter,
            tol=tol,
        )

        st = CoarseApproachStatus()
        st.header.stamp = self.get_clock().now().to_msg()
        st.phase = phase
        st.range_to_standoff_m = g.range_to_standoff_m
        st.axis_offset_m = g.axis_offset_m
        st.vertical_error_m = g.vertical_error_m
        st.heading_error_rad = g.yaw_err
        st.within_position_tol = within_pos
        st.within_heading_tol = within_head
        st.dock_healthy = gate.dock_healthy
        st.ready_for_handoff = ready
        self._pub_status.publish(st)


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
