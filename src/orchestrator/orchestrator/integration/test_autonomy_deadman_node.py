"""Integration test: autonomy_deadman relays /cmd_vel_auto -> /cmd_vel only
while the deadman button is held, and zeroes on the release edge."""

import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
)

_BUTTON = 5


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class _Harness(Node):
    def __init__(self):
        super().__init__("deadman_harness")
        self._joy = self.create_publisher(Joy, "/joy", _RELIABLE)
        self._auto = self.create_publisher(Twist, "/cmd_vel_auto", _RELIABLE)
        self.out: list[Twist] = []
        self.create_subscription(Twist, "/cmd_vel", self.out.append, _RELIABLE)

    def hold(self, held: bool):
        j = Joy()
        j.buttons = [0] * 8
        j.buttons[_BUTTON] = 1 if held else 0
        self._joy.publish(j)

    def auto(self, surge: float):
        t = Twist()
        t.linear.x = surge
        self._auto.publish(t)


def _params():
    return [Parameter("deadman_button", Parameter.Type.INTEGER, _BUTTON)]


def _spin(node, harness, feed, *, iterations, period=0.05):
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(harness)
    stop = threading.Event()
    t = threading.Thread(
        target=lambda: [
            ex.spin_once(timeout_sec=0.02)
            for _ in iter(lambda: not stop.is_set(), False)
        ],
        daemon=True,
    )
    t.start()
    for i in range(iterations):
        feed(i)
        time.sleep(period)
    stop.set()
    t.join(timeout=1.0)


def test_relays_when_held(ros_context):
    from orchestrator.autonomy_deadman_node import AutonomyDeadman

    node = AutonomyDeadman(parameter_overrides=_params())
    harness = _Harness()

    def feed(i):
        harness.hold(True)
        harness.auto(0.3)

    _spin(node, harness, feed, iterations=10)
    assert any(c.linear.x == 0.3 for c in harness.out), "should relay while held"
    node.destroy_node()
    harness.destroy_node()


def test_silent_when_released(ros_context):
    from orchestrator.autonomy_deadman_node import AutonomyDeadman

    node = AutonomyDeadman(parameter_overrides=_params())
    harness = _Harness()

    def feed(i):
        harness.hold(False)
        harness.auto(0.3)

    _spin(node, harness, feed, iterations=10)
    assert all(c.linear.x != 0.3 for c in harness.out), \
        "must not relay autonomy while released"
    node.destroy_node()
    harness.destroy_node()


def test_zero_on_release_edge(ros_context):
    from orchestrator.autonomy_deadman_node import AutonomyDeadman

    node = AutonomyDeadman(parameter_overrides=_params())
    harness = _Harness()

    def feed(i):
        # hold for the first half, release for the second half
        harness.hold(i < 5)
        harness.auto(0.3)

    _spin(node, harness, feed, iterations=10)
    # after release there must be a zero Twist (the stop command)
    assert any(
        c.linear.x == 0.0 for c in harness.out
    ), "expected a zero Twist on the release edge"
    node.destroy_node()
    harness.destroy_node()
