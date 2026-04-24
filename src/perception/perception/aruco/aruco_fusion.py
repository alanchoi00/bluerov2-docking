#!/usr/bin/env python3
"""aruco_fusion ROS2 node per-frame multi-marker spatial fusion."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from perception.aruco.lib.dock_layout import (
    EXPECTED_MARKER_IDS,
    MARKER_POSE_IN_DOCK,
    MARKER_SIZE,
)
from perception.aruco.lib.geometry import quat_inverse, quat_multiply
from perception.aruco.lib.noise_model import (
    marker_position_covariance,
    marker_rotation_covariance,
)
from perception.aruco.lib.pose_fusion import MarkerMeasurement, fuse_markers
from perception.aruco.lib.spatial_consensus import MarkerCandidate, filter_consistent


def _is_empty(pose_stamped: PoseStamped) -> bool:
    """aruco_relay publishes PoseStamped() with default identity orientation when a marker
    is absent. Position is all-zero; a real detection always has non-zero z in camera frame.
    """
    p = pose_stamped.pose.position
    return p.x == 0.0 and p.y == 0.0 and p.z == 0.0


def _pose_to_numpy(ps: PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    p = ps.pose.position
    o = ps.pose.orientation
    return np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])


def _compute_implied_dock_origin(
    marker_position_cam: np.ndarray,
    marker_orientation_cam: np.ndarray,
    marker_position_in_dock: np.ndarray,
    marker_orientation_in_dock: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a marker's pose in camera and in dock, compute the dock-origin pose in camera.

    Rotates the marker's in-dock position vector by q_dock_cam (dock->camera rotation)
    using scipy.Rotation.apply the q*[v,0]*q^-1 trick requires Hamilton-product math
    on pure-imaginary quaternions, which our scipy-wrapped quat_multiply doesn't do
    (it normalizes any 4-tuple input and treats it as a unit rotation).
    """
    q_dock_cam = quat_multiply(
        marker_orientation_cam, quat_inverse(marker_orientation_in_dock)
    )
    rotated = Rotation.from_quat(q_dock_cam).apply(marker_position_in_dock)
    p_dock_cam = marker_position_cam - rotated
    return p_dock_cam, q_dock_cam


class ArucoFusion(Node):
    def __init__(self):
        super().__init__("aruco_fusion")

        self.declare_parameter("consensus_threshold_deg", 8.0)
        self.declare_parameter("noise_scale_alpha", 2.0)
        # Rotation-noise scale for the PnP-error kernel sigma_rot = alpha * r / s.
        # Derived from the same pixel-noise physics as position (same alpha).
        self.declare_parameter("noise_scale_alpha_rot", 0.001)
        self.declare_parameter("min_markers_for_consensus", 3)

        self._pub = self.create_publisher(
            PoseWithCovarianceStamped, "/perception/aruco_dock_pose", 10
        )

        self._subs = [
            Subscriber(self, PoseStamped, f"/perception/aruco_{mid}")
            for mid in EXPECTED_MARKER_IDS
        ]
        self._sync = ApproximateTimeSynchronizer(self._subs, queue_size=10, slop=0.05)
        self._sync.registerCallback(self._on_synced)

        self.get_logger().info("aruco_fusion ready")

    def _on_synced(self, *pose_messages: PoseStamped) -> None:
        threshold_rad = math.radians(
            self.get_parameter("consensus_threshold_deg")
            .get_parameter_value()
            .double_value
        )
        alpha = (
            self.get_parameter("noise_scale_alpha").get_parameter_value().double_value
        )
        alpha_rot = (
            self.get_parameter("noise_scale_alpha_rot")
            .get_parameter_value()
            .double_value
        )
        min_for_check = (
            self.get_parameter("min_markers_for_consensus")
            .get_parameter_value()
            .integer_value
        )

        candidates: list[MarkerCandidate] = []
        measurements: list[MarkerMeasurement] = []
        first_non_empty_stamp = None

        for mid, ps in zip(EXPECTED_MARKER_IDS, pose_messages):
            if _is_empty(ps):
                continue
            if first_non_empty_stamp is None:
                first_non_empty_stamp = ps.header.stamp
            p_cam, q_cam = _pose_to_numpy(ps)
            p_dock, q_dock = MARKER_POSE_IN_DOCK[mid]
            p_origin, q_origin = _compute_implied_dock_origin(
                p_cam, q_cam, np.array(p_dock), np.array(q_dock)
            )

            candidates.append(
                MarkerCandidate(marker_id=mid, position=p_origin, orientation=q_origin)
            )
            r_m = float(np.linalg.norm(p_cam))
            size_m = MARKER_SIZE[mid]
            pos_cov = marker_position_covariance(
                range_m=r_m, marker_size_m=size_m, alpha=alpha
            )
            rot_cov = marker_rotation_covariance(
                range_m=r_m, marker_size_m=size_m, alpha=alpha_rot
            )
            measurements.append(
                MarkerMeasurement(
                    marker_id=mid,
                    position=p_origin,
                    orientation=q_origin,
                    position_covariance=pos_cov,
                    orientation_covariance=rot_cov,
                )
            )

        if not candidates:
            return

        surviving = filter_consistent(candidates, threshold_rad, min_for_check)
        surviving_ids = {c.marker_id for c in surviving}
        measurements = [m for m in measurements if m.marker_id in surviving_ids]

        if not measurements:
            return

        fused = fuse_markers(measurements)

        out = PoseWithCovarianceStamped()
        out.header.stamp = first_non_empty_stamp
        out.header.frame_id = "camera_link"
        out.pose.pose.position.x = float(fused.position[0])
        out.pose.pose.position.y = float(fused.position[1])
        out.pose.pose.position.z = float(fused.position[2])
        out.pose.pose.orientation.x = float(fused.orientation[0])
        out.pose.pose.orientation.y = float(fused.orientation[1])
        out.pose.pose.orientation.z = float(fused.orientation[2])
        out.pose.pose.orientation.w = float(fused.orientation[3])

        cov6 = np.zeros((6, 6))
        cov6[:3, :3] = fused.position_covariance
        cov6[3:, 3:] = fused.orientation_covariance
        out.pose.covariance = cov6.flatten().tolist()

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
