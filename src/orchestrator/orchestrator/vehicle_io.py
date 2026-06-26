"""Adapter wrapping the MAVROS mode/arming services so the FSM depends on a
small local interface, not MAVROS directly. Swappable for hardware, mockable
in tests."""

from __future__ import annotations

from mavros_msgs.srv import SetMode, CommandBool


class VehicleIO:
    def __init__(self, node) -> None:
        self._node = node
        self._set_mode = node.create_client(SetMode, "/mavros/set_mode")
        self._arming = node.create_client(CommandBool, "/mavros/cmd/arming")

    def set_mode(self, mode: str) -> None:
        """Request a flight-mode change (async, fire-and-forget with a log)."""
        req = SetMode.Request()
        req.custom_mode = mode
        if not self._set_mode.service_is_ready():
            self._node.get_logger().warn(f"/mavros/set_mode not ready; mode={mode} skipped")
            return
        fut = self._set_mode.call_async(req)
        fut.add_done_callback(
            lambda f: self._node.get_logger().info(f"set_mode({mode}) -> {f.result()}")
        )

    def set_arm(self, arm: bool) -> None:
        """Arm/disarm the vehicle (async)."""
        req = CommandBool.Request()
        req.value = bool(arm)
        if not self._arming.service_is_ready():
            self._node.get_logger().warn(f"/mavros/cmd/arming not ready; arm={arm} skipped")
            return
        fut = self._arming.call_async(req)
        fut.add_done_callback(
            lambda f: self._node.get_logger().info(f"set_arm({arm}) -> {f.result()}")
        )
