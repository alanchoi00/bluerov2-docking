#!/usr/bin/env python3
import rclpy
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.publisher import Publisher as rclpyPublisher


class ArucoRelay(Node):
    def __init__(self):
        super().__init__(node_name="aruco_relay")

        self.declare_parameter("reference_marker_size", 0.1)
        self.declare_parameter("marker_sizes", [""])

        self._size_cache: dict[int, float] = {}
        self._pose_publishers: dict[int, rclpyPublisher] = {}

        raw = (
            self.get_parameter("marker_sizes").get_parameter_value().string_array_value
        )
        for entry in raw:
            if ":" in entry:
                id_str, size_str = entry.split(":", 1)
                mid = int(id_str)
                self._size_cache[mid] = float(size_str)
                topic = f"/perception/aruco_{mid}"
                self._pose_publishers[mid] = self.create_publisher(
                    PoseStamped, topic, 10
                )
                self.get_logger().info(
                    f"Registered marker {mid} ({size_str}m) on {topic}"
                )

        self.create_subscription(
            MarkerArray,
            "/arucos/markers",
            self._on_detections,
            10,
        )

    def _scale(self, mid: int) -> float:
        ref = (
            self.get_parameter("reference_marker_size")
            .get_parameter_value()
            .double_value
        )
        return self._size_cache.get(mid, ref) / ref

    def _on_detections(self, msg: MarkerArray) -> None:
        detected: dict[int, PoseStamped] = {}

        for marker in msg.markers:
            mid = marker.id
            if mid not in self._pose_publishers:
                continue

            s = self._scale(mid)
            p = marker.pose.pose.position

            pose = PoseStamped()
            pose.header = marker.header
            pose.pose.position.x = p.x * s
            pose.pose.position.y = p.y * s
            pose.pose.position.z = p.z * s
            pose.pose.orientation = marker.pose.pose.orientation
            detected[mid] = pose

        for mid, pub in self._pose_publishers.items():
            if mid in detected:
                pub.publish(detected[mid])
            else:
                empty = PoseStamped()
                empty.header = msg.header
                pub.publish(empty)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
