"""Closed-loop simulation: CoarsePbvsController <-> BlueRovSim plant.

Each tick: compute the dock pose in the body frame from BlueRovSim's world pose,
ask the controller for a command, map it to BlueRovSim's 6-DOF command vector
(roll/pitch left to passive stability), step the plant, and log.
"""

import _paths  # noqa: F401  (sets sys.path)

from dataclasses import dataclass

import numpy as np
from bluerov_model import BlueRovModel

from control.pbvs import CoarsePbvsController, CoarsePbvsParams
from dock_signal import dock_pose_in_body
from scenarios import Scenario


@dataclass
class Trajectory:
    t: np.ndarray
    forward: np.ndarray  # body-frame range to dock (m); target = handoff_range_m
    left: np.ndarray  # lateral error (m); target 0
    up: np.ndarray  # vertical error (m); target 0
    yaw_err: np.ndarray  # heading error (rad); target 0
    cmd_surge: np.ndarray
    cmd_sway: np.ndarray
    cmd_heave: np.ndarray
    cmd_yaw: np.ndarray


def run(
    params: CoarsePbvsParams,
    scenario: Scenario,
    handoff_range_m: float,
    dt: float = 0.01,
    t_max: float = 15.0,
) -> Trajectory:
    model = BlueRovModel(dt=dt)
    model.eta = np.array(scenario.eta0, dtype=float).copy()

    controller = CoarsePbvsController(params)
    controller.reset()

    # Coarse approach servoes to a STANDOFF point handoff_range_m in front of the
    # dock along the approach axis (the vehicle holds dock_heading), then hands
    # off to fine alignment -- it does not drive into the dock. Mirrors the
    # production guidance, which feeds the controller range-to-standoff. The
    # standoff sits handoff_range_m back from the dock along the heading vector.
    approach = np.array(
        [np.cos(scenario.dock_heading), np.sin(scenario.dock_heading), 0.0]
    )
    standoff_pos_world = (
        np.asarray(scenario.dock_pos_world, dtype=float) - handoff_range_m * approach
    )

    n = int(round(t_max / dt))
    cols = {k: np.zeros(n) for k in
            ("t", "fwd", "left", "up", "yaw", "cs", "csw", "ch", "cy")}

    for i in range(n):
        rel, yaw_err = dock_pose_in_body(
            model.eta, scenario.dock_pos_world, scenario.dock_heading
        )
        rel_standoff, _ = dock_pose_in_body(
            model.eta, standoff_pos_world, scenario.dock_heading
        )
        cmd = controller.step(rel_standoff, yaw_err, dt)
        model.step(
            np.array([cmd.surge, cmd.sway, cmd.heave, 0.0, 0.0, cmd.yaw_rate])
        )

        cols["t"][i] = i * dt
        # Log the true range-to-dock; it converges to handoff_range_m (forward),
        # while lateral/vertical converge to 0 on the approach axis.
        cols["fwd"][i], cols["left"][i], cols["up"][i] = rel
        cols["yaw"][i] = yaw_err
        cols["cs"][i], cols["csw"][i] = cmd.surge, cmd.sway
        cols["ch"][i], cols["cy"][i] = cmd.heave, cmd.yaw_rate

    return Trajectory(
        cols["t"], cols["fwd"], cols["left"], cols["up"], cols["yaw"],
        cols["cs"], cols["csw"], cols["ch"], cols["cy"],
    )
