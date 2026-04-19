#!/usr/bin/env python3
import rclpy
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.publisher import Publisher as rclpyPublisher
from tf2_ros import TransformBroadcaster


class ArucoRelay(Node):
    def __init__(self):
        super().__init__(node_name="aruco_relay")

        self._pose_publishers: dict[int, rclpyPublisher] = {}
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            MarkerArray,
            "/aruco/detections",
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

            tf = TransformStamped()
            tf.header = marker.header
            tf.child_frame_id = f"aruco_{mid}"
            tf.transform.translation.x = marker.pose.pose.position.x
            tf.transform.translation.y = marker.pose.pose.position.y
            tf.transform.translation.z = marker.pose.pose.position.z
            tf.transform.rotation = marker.pose.pose.orientation
            self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
