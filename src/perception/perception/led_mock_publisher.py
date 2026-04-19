#!/usr/bin/env python3
import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformException  # type: ignore[attr-defined]
import tf2_geometry_msgs

from perception.utils.transforms import apply_dock_pose


_LED_LINK_NAMES_DEFAULT = [
    "led_top_left",
    "led_top_right",
    "led_bottom_left",
    "led_bottom_right",
]


class LedMockPublisher(Node):
    def __init__(self):
        super().__init__("led_mock_publisher")

        self.declare_parameter("noise_stddev", 0.01)
        self.declare_parameter("detection_distance", 10.0)
        self.declare_parameter("gz_world_name", "ocean_world")
        self.declare_parameter("dock_model_name", "docking_station")
        self.declare_parameter("led_link_names", _LED_LINK_NAMES_DEFAULT)
        self.declare_parameter("robot_odom_topic", "/model/bluerov2_heavy/odometry")

        self._led_world_positions = self._fetch_led_poses_from_gz()
        if self._led_world_positions is None:
            self.get_logger().warn(
                "Could not fetch LED poses from Gazebo — using zero positions"
            )
            self._led_world_positions = [[0.0, 0.0, 0.0]] * 4

        self._rng = np.random.default_rng()
        self._camera_info = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._pub = self.create_publisher(PoseArray, "/perception/leds", 10)
        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self._on_image, 10
        )
        self._info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self._on_camera_info, 10
        )
        odom_topic = (
            self.get_parameter("robot_odom_topic").get_parameter_value().string_value
        )
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 10
        )

    def _fetch_led_poses_from_gz(self):
        """Query Gazebo for LED link world positions via gz transport.

        Subscribes to /world/<gz_world_name>/pose/info and collects:
        - the dock model's world pose (position + orientation)
        - the 4 LED link positions in dock-local frame

        Applies apply_dock_pose to return LED positions in world frame.
        Times out after 10 seconds. Returns list of 4 [x, y, z] or None.
        """
        try:
            import gz.transport13 as transport
            from gz.msgs10.pose_v_pb2 import Pose_V  # type: ignore[import]
        except ImportError:
            self.get_logger().warn("gz.transport13 not available")
            return None

        world = self.get_parameter("gz_world_name").get_parameter_value().string_value
        model = self.get_parameter("dock_model_name").get_parameter_value().string_value
        link_names = list(
            self.get_parameter("led_link_names")
            .get_parameter_value()
            .string_array_value
        )
        topic = f"/world/{world}/pose/info"

        dock_pose_xyzrpy = {}
        led_offsets = {}
        all_found = threading.Event()

        def _on_pose_v(msg):
            for pose in msg.pose:
                if pose.name == model and model not in dock_pose_xyzrpy:
                    p = pose.position
                    q = pose.orientation
                    sinr = 2.0 * (q.w * q.x + q.y * q.z)
                    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                    roll = math.atan2(sinr, cosr)
                    sinp = 2.0 * (q.w * q.y - q.z * q.x)
                    pitch = math.asin(max(-1.0, min(1.0, sinp)))
                    siny = 2.0 * (q.w * q.z + q.x * q.y)
                    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    yaw = math.atan2(siny, cosy)
                    dock_pose_xyzrpy[model] = [p.x, p.y, p.z, roll, pitch, yaw]
                elif pose.name in link_names and pose.name not in led_offsets:
                    p = pose.position
                    led_offsets[pose.name] = [p.x, p.y, p.z]
            if model in dock_pose_xyzrpy and all(n in led_offsets for n in link_names):
                all_found.set()

        gz_node = transport.Node()
        gz_node.subscribe(Pose_V, topic, _on_pose_v)
        all_found.wait(timeout=10.0)

        missing = [n for n in link_names if n not in led_offsets]
        if missing or model not in dock_pose_xyzrpy:
            not_found = missing + ([] if model in dock_pose_xyzrpy else [model])
            self.get_logger().warn(f"Entities not found in Gazebo: {not_found}")
            return None

        dp = dock_pose_xyzrpy[model]
        offsets = np.array([led_offsets[n] for n in link_names], dtype=float)
        return apply_dock_pose(dp[:3], dp[3:], offsets).tolist()

    def _on_camera_info(self, msg: CameraInfo):
        self._camera_info = msg

    def _on_odom(self, msg: Odometry):
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = msg.child_frame_id
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self._tf_broadcaster.sendTransform(t)

    def _on_image(self, msg):
        try:
            transform = self._tf_buffer.lookup_transform(
                "camera_link",
                "map",
                msg.header.stamp,
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return

        noise_stddev = (
            self.get_parameter("noise_stddev").get_parameter_value().double_value
        )
        max_dist = (
            self.get_parameter("detection_distance").get_parameter_value().double_value
        )

        info = self._camera_info
        if info is None:
            return

        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        w, h = info.width, info.height

        poses = []
        for wp in self._led_world_positions:
            pt = PointStamped()
            pt.header.frame_id = "map"
            pt.header.stamp = msg.header.stamp
            pt.point.x = float(wp[0])
            pt.point.y = float(wp[1])
            pt.point.z = float(wp[2])

            pt_cam = tf2_geometry_msgs.do_transform_point(pt, transform)
            x, y, z = pt_cam.point.x, pt_cam.point.y, pt_cam.point.z

            if z <= 0.0:
                continue
            if (x * x + y * y + z * z) > max_dist * max_dist:
                continue

            u = fx * x / z + cx
            v = fy * y / z + cy
            if not (0.0 <= u < w and 0.0 <= v < h):
                continue

            noise = self._rng.normal(0.0, noise_stddev, 3)
            pose = Pose()
            pose.position.x = x + float(noise[0])
            pose.position.y = y + float(noise[1])
            pose.position.z = z + float(noise[2])
            pose.orientation.w = 1.0
            poses.append(pose)

        if not poses:
            return

        pa = PoseArray()
        pa.header.frame_id = "camera_link"
        pa.header.stamp = msg.header.stamp
        pa.poses = poses
        self._pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = LedMockPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
