#!/usr/bin/env python3
"""autonomy_deadman: relay /cmd_vel_auto -> /cmd_vel only while a joystick
deadman button is held.

The blue_teleop joystick already self-gates manual /cmd_vel on its own deadman
(axis 2 / left trigger), so /cmd_vel is silent unless the operator is actively
flying. This node adds a SECOND, independent deadman for autonomy: while the
configured button is held, the docking controllers' commands (published to
/cmd_vel_auto) are relayed to /cmd_vel; on release the relay stops and one zero
Twist is sent so autonomy halts immediately and ALT_HOLD holds depth. Hold the
autonomy button OR the manual trigger -- never both.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class AutonomyDeadman(Node):
    def __init__(self, **kwargs):
        super().__init__("autonomy_deadman", **kwargs)

        self.declare_parameter("deadman_button", 5)
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("input_topic", "/cmd_vel_auto")
        self.declare_parameter("output_topic", "/cmd_vel")

        self._button = (
            self.get_parameter("deadman_button").get_parameter_value().integer_value
        )
        joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        in_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        out_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )

        self._held = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub = self.create_publisher(Twist, out_topic, qos)
        self._pub_engaged = self.create_publisher(Bool, "/docking/engaged", qos)
        self.create_subscription(Joy, joy_topic, self._on_joy, qos)
        self.create_subscription(Twist, in_topic, self._on_auto, qos)
        self.get_logger().info(
            f"autonomy_deadman ready: relay {in_topic} -> {out_topic} "
            f"while button {self._button} held"
        )

    def _on_joy(self, msg: Joy) -> None:
        held = self._button < len(msg.buttons) and msg.buttons[self._button] == 1
        if self._held and not held:
            # release edge: stop autonomy immediately so the vehicle does not
            # coast on the last relayed command.
            self._pub.publish(Twist())
        self._held = held
        self._pub_engaged.publish(Bool(data=held))

    def _on_auto(self, msg: Twist) -> None:
        if self._held:
            self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomyDeadman()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
