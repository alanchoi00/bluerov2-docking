"""Integration test: aruco_fusion node round-trip.

Launches aruco_fusion, publishes 9 PoseStamped messages on the per-marker
topics, asserts a fused PoseWithCovarianceStamped appears on the output
topic with reasonable values.
"""

import time
import threading

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


MARKER_IDS = (201, 202, 301, 302, 303, 304, 305, 401, 402)


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Harness(Node):
    """Publishes per-marker poses, subscribes to fused output."""

    def __init__(self):
        super().__init__("aruco_fusion_test_harness")
        self._pub_map = {
            mid: self.create_publisher(PoseStamped, f"/perception/aruco_{mid}", 10)
            for mid in MARKER_IDS
        }
        self._received: list[PoseWithCovarianceStamped] = []
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/perception/aruco_dock_pose",
            lambda m: self._received.append(m),
            10,
        )

    def publish_all_empty(self):
        stamp = self.get_clock().now().to_msg()
        for pub in self._pub_map.values():
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "camera_link"
            # Default PoseStamped() has orientation.w=1.0 (identity) and position all-zero
            # this matches aruco_relay's empty-marker convention.
            pub.publish(msg)

    def publish_marker(
        self,
        mid: int,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
    ):
        stamp = self.get_clock().now().to_msg()
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_link"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._pub_map[mid].publish(msg)

    @property
    def received(self):
        return list(self._received)


def test_empty_markers_no_output(ros_context):
    from perception.aruco.aruco_fusion import ArucoFusion

    node_under_test = ArucoFusion()
    harness = _Harness()

    exec_ = SingleThreadedExecutor()
    exec_.add_node(node_under_test)
    exec_.add_node(harness)

    stop = threading.Event()

    def _spin():
        while not stop.is_set():
            exec_.spin_once(timeout_sec=0.05)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()

    for _ in range(5):
        harness.publish_all_empty()
        time.sleep(0.05)

    time.sleep(0.2)
    stop.set()
    t.join(timeout=1.0)

    assert harness.received == []
    node_under_test.destroy_node()
    harness.destroy_node()


def test_three_agreeing_markers_produces_fused_output(ros_context):
    from perception.aruco.aruco_fusion import ArucoFusion

    node_under_test = ArucoFusion()
    harness = _Harness()

    exec_ = SingleThreadedExecutor()
    exec_.add_node(node_under_test)
    exec_.add_node(harness)

    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [
            exec_.spin_once(timeout_sec=0.05)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    t.start()

    # Scenario: dock is at (0, 0, 1.0) in camera_link, facing the camera.
    # Camera axis: X=right, Y=down, Z=forward. Dock axis: X=right, Y=into-page, Z=up.
    # Dock-to-camera rotation takes dock +Y->camera +Z and dock +Z->camera -Y,
    # which is -90deg about camera's X axis: q_dock_cam = (-0.7071, 0, 0, 0.7071).
    #
    # Each marker has q_marker_dock = (+0.7071, 0, 0, 0.7071) (see dock_layout),
    # so q_marker_cam = q_dock_cam * q_marker_dock = identity (w=1).
    #
    # Marker camera position = dock_cam + R(-90deg_about_X) * offset_in_dock,
    # where R(-90deg_about_X)(x,y,z) = (x, z, -y). For each marker:
    #   301 offset (0, 0.310, -0.03675)   -> (0,       -0.03675, -0.310)
    #         -> p_cam = (0, -0.03675, 0.690)
    #   401 offset (-0.02625, 0.310, 0.042) -> (-0.02625, 0.042,   -0.310)
    #         -> p_cam = (-0.02625, 0.042, 0.690)
    #   402 offset ( 0.02625, 0.310, 0.042) -> ( 0.02625, 0.042,   -0.310)
    #         -> p_cam = ( 0.02625, 0.042, 0.690)
    for _ in range(5):
        for mid in MARKER_IDS:
            if mid not in (301, 401, 402):
                stamp = harness.get_clock().now().to_msg()
                empty = PoseStamped()
                empty.header.stamp = stamp
                empty.header.frame_id = "camera_link"
                harness._pub_map[mid].publish(empty)

        harness.publish_marker(301, 0.00000, -0.03675, 0.690)
        harness.publish_marker(401, -0.02625, 0.04200, 0.690)
        harness.publish_marker(402, 0.02625, 0.04200, 0.690)
        time.sleep(0.05)

    time.sleep(0.3)
    stop.set()
    t.join(timeout=1.0)

    assert len(harness.received) >= 1
    fused = harness.received[-1]
    assert fused.header.frame_id == "camera_link"
    # Fused dock origin should be near (0, 0, 1.0).
    assert abs(fused.pose.pose.position.x - 0.0) < 0.05
    assert abs(fused.pose.pose.position.y - 0.0) < 0.05
    assert abs(fused.pose.pose.position.z - 1.0) < 0.05
    # Fused dock orientation should be near q_dock_cam = (-0.7071, 0, 0, 0.7071).
    # Sign ambiguity in quaternions allows the equivalent (+0.7071, 0, 0, -0.7071),
    # so compare |components| rather than raw signs.
    qx, qy, qz, qw = (
        fused.pose.pose.orientation.x,
        fused.pose.pose.orientation.y,
        fused.pose.pose.orientation.z,
        fused.pose.pose.orientation.w,
    )
    assert abs(abs(qx) - 0.7071) < 0.01
    assert abs(qy) < 0.01
    assert abs(qz) < 0.01
    assert abs(abs(qw) - 0.7071) < 0.01

    node_under_test.destroy_node()
    harness.destroy_node()
