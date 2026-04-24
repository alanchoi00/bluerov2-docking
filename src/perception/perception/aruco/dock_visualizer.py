#!/usr/bin/env python3
"""Publishes a Marker of the dock mesh in the dock_filtered TF frame.

RViz renders the mesh at the filtered dock pose by following the TF tree
(dock_filtered is broadcast by dock_pose_filter after the KF initializes).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker


class DockVisualizer(Node):
    def __init__(self):
        super().__init__("dock_visualizer")

        self.declare_parameter(
            "mesh_resource",
            "package://description/models/docking_station/meshes/dock_station.dae",
        )
        self.declare_parameter("mesh_offset_z", -0.190)
        self.declare_parameter("dock_frame", "dock_filtered")
        self.declare_parameter("publish_rate_hz", 5.0)
        # True: render the DAE with its own embedded materials/textures
        # (proper-looking ArUco patterns and dock body colors). False: render
        # as a uniform-coloured blob using the fallback_color_* parameters
        # useful for diagnosing when embedded materials fail to load.
        self.declare_parameter("use_embedded_materials", True)
        self.declare_parameter("fallback_color_r", 0.8)
        self.declare_parameter("fallback_color_g", 0.8)
        self.declare_parameter("fallback_color_b", 1.0)
        self.declare_parameter("fallback_color_a", 0.8)

        # RELIABLE + TRANSIENT_LOCAL so late-joining RViz sees the marker.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub = self.create_publisher(Marker, "/perception/dock_visualization", qos)
        rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info("dock_visualizer ready")

    def _publish(self) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = (
            self.get_parameter("dock_frame").get_parameter_value().string_value
        )
        marker.ns = "dock_filtered"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = (
            self.get_parameter("mesh_offset_z").get_parameter_value().double_value
        )
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        use_embedded = (
            self.get_parameter("use_embedded_materials")
            .get_parameter_value()
            .bool_value
        )
        marker.mesh_use_embedded_materials = use_embedded
        if use_embedded:
            # All zeros tells RViz "do not apply any tint at all render the
            # mesh with its own materials and textures faithfully".
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.0
        else:
            marker.color.r = (
                self.get_parameter("fallback_color_r")
                .get_parameter_value()
                .double_value
            )
            marker.color.g = (
                self.get_parameter("fallback_color_g")
                .get_parameter_value()
                .double_value
            )
            marker.color.b = (
                self.get_parameter("fallback_color_b")
                .get_parameter_value()
                .double_value
            )
            marker.color.a = (
                self.get_parameter("fallback_color_a")
                .get_parameter_value()
                .double_value
            )

        marker.mesh_resource = (
            self.get_parameter("mesh_resource").get_parameter_value().string_value
        )
        self._pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = DockVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
