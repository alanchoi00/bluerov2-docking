#!/usr/bin/env python3
import rclpy
from aruco_msgs.msg import ArucoDetection
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class ArucoRelay(Node):
    def __init__(self):
        super().__init__("aruco_relay")

        self._publishers: dict[int, rclpy.publisher.Publisher] = {}

        self.create_subscription(
            ArucoDetection,
            "/aruco/slope/detections",
            self._on_detections,
            10,
        )
        self.create_subscription(
            ArucoDetection,
            "/aruco/backplate/detections",
            self._on_detections,
            10,
        )

    def _on_detections(self, msg: ArucoDetection) -> None:
        for marker in msg.markers:
            mid = marker.id
            if mid not in self._publishers:
                topic = f"/perception/aruco/{mid}"
                self._publishers[mid] = self.create_publisher(
                    PoseStamped, topic, 10
                )
                self.get_logger().info(f"New marker detected, publishing on {topic}")

            out = PoseStamped()
            out.header = marker.pose.header
            out.pose = marker.pose.pose.pose
            self._publishers[mid].publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
