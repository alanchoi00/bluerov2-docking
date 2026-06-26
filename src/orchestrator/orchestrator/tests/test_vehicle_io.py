"""Smoke test: VehicleIO constructs its clients without a running MAVROS."""

import pytest
import rclpy
from rclpy.node import Node

from orchestrator.vehicle_io import VehicleIO


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_vehicle_io_constructs_and_skips_when_services_absent(ros_context):
    node = Node("vehicle_io_test")
    io = VehicleIO(node)
    # services are not up; these must not raise (they warn and return)
    io.set_mode("ALT_HOLD")
    io.set_arm(False)
    node.destroy_node()
