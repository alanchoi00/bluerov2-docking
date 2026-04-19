#!/usr/bin/env python3
import rclpy
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.publisher import Publisher as rclpyPublisher


class ArucoRelay(Node):
    def __init__(self):
        super().__init__(node_name="aruco_relay")

        self._pose_publishers: dict[int, rclpyPublisher] = {}

        self.create_subscription(
            MarkerArray,
            "/aruco/slope/detections",
            self._on_detections,
            10,
        )
        self.create_subscription(
            MarkerArray,
            "/aruco/backplate/detections",
            self._on_detections,
            10,
        )

    def _on_detections(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            mid = marker.id
            if mid not in self._pose_publishers:
                topic = f"/perception/aruco/{mid}"
                self._pose_publishers[mid] = self.create_publisher(
                    PoseStamped, topic, 10
                )
                self.get_logger().info(f"New marker detected, publishing on {topic}")

            out = PoseStamped()
            out.header = marker.header
            out.pose = marker.pose.pose
            self._pose_publishers[mid].publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
