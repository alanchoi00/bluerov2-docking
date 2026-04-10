# bluerov2-docking

Autonomous docking of a BlueROV2 to a free-floating underwater station using vision-based guidance. UNSW Thesis research project.

## Overview

The system uses a two-level visual guidance pipeline to guide the ROV through a sequence of docking behaviours:

| Level | Guidance | Range | Method |
|-------|----------|-------|--------|
| I | Coarse approach | Long range | LED cue detection (YOLO) -> centroid-based IBVS |
| II | Fine alignment | Close range | ArUco/AprilTag PnP -> 6-DOF pose error PID |

A state machine manages mode transitions: `SEARCHING -> COARSE_APPROACH -> FINE_ALIGNMENT -> ENGAGE -> DOCKED` (with `ABORT` fallback).

The free-floating dock is modelled with sinusoidal heave/sway motion in simulation to emulate a cable-suspended structure. Pose estimates are temporally filtered (Kalman or complementary) to account for update-rate limitations and dropout.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension (`ms-vscode-remote.remote-containers`)
- For NVIDIA GPU: `nvidia-container-toolkit` installed on the host
- For Nouveau/Intel GPU: no extra GPU setup required

The dev containers pull a pre-built image from GHCR that includes ROS2 Humble, Gazebo Harmonic, and the [blue](https://github.com/Robotic-Decision-Making-Lab/blue) BlueROV2 packages. No local ROS2 install is needed.

## Dev container setup (Recommended)

This is the recommended way to develop. The container mounts the repo into `/home/blue/ws/src/bluerov2-docking` and runs `rosdep install` on first launch.

1. Clone the repo:
   ```bash
   git clone https://github.com/alanchoi00/bluerov2-docking.git
   cd bluerov2-docking
   ```

2. Open in VS Code:
   ```bash
   code .
   ```

3. When prompted, select **Reopen in Container** and choose your GPU variant:
   - `bluerov2-docking (Nouveau)`: AMD/Intel GPU or no GPU
   - `bluerov2-docking (NVIDIA)`: NVIDIA GPU

   VS Code will pull the image, build the container, and run `rosdep install` automatically.

4. Open VSCode terminal and build the workspace inside the container:

   ```bash
   cd /home/blue/ws
   colcon build
   source install/setup.bash
   ```

To run commands from an external terminal, add this alias to your `~/.bashrc` or `~/.zshrc` to avoid hassle of finding the docker image name each time:

```bash
alias docking-shell='docker exec -it $(docker ps --format "{{.Image}} {{.Names}}" | grep bluerov2-docking | awk "{print \$2}") bash'
```

### Rebuilding the container

If the base image has been updated:
```bash
# From VS Code command palette:
Dev Containers: Rebuild Container Without Cache
```

## Manual setup (without dev container)

> Requires ROS2 Humble and Gazebo Harmonic installed on the host.

```bash
git clone https://github.com/alanchoi00/bluerov2-docking.git

mkdir -p ~/ws/src
cd ~/ws/src
ln -s /path/to/bluerov2-docking/src bluerov2-docking

cd ~/ws
rosdep install -y --from-paths src --ignore-src --rosdistro humble \
  --skip-keys="gz-transport12 gz-sim7 gz-math7 gz-msgs9 gz-plugin2"

colcon build
source install/setup.bash
```

## Simulation

The Gazebo simulation environment is provided by [alanchoi00/blue-sim-humble](https://github.com/alanchoi00/blue-sim-humble), a ROS2 Humble fork of the upstream [blue](https://github.com/Robotic-Decision-Making-Lab/blue) package. Pre-built Docker images are published to GHCR and are used as the base for the dev containers in this repo.

### Launch the vehicle

```bash
# BlueROV2 Heavy (8 thrusters) in Gazebo
ros2 launch blue_bringup bluerov2_heavy.launch.yaml use_sim:=true

# Standard BlueROV2 (6 thrusters)
ros2 launch blue_bringup bluerov2.launch.yaml use_sim:=true
```

Wait for the line `[mavros.param]: PR: parameters list received` before sending commands.

### Key launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_sim` | `false` | Enable Gazebo + ArduSub SITL |
| `use_sim_time` | `false` | Use Gazebo clock |
| `use_rviz` | `false` | Open RViz |
| `use_camera` | `false` | Enable camera stream |
| `localization_source` | `gazebo` | Pose source: `gazebo`, `mocap`, or `camera` |
| `gazebo_world_file` | `underwater.world` | Gazebo world to load |

### Launch with controllers

Terminal 1:
```bash
ros2 launch blue_demos bluerov2_demo.launch.yaml use_sim:=true
```

Terminal 2 (after MAVROS connects):
```bash
ros2 launch blue_demos bluerov2_controllers.launch.py use_sim:=true
```

Verify:
```bash
ros2 control list_controllers
```

### Teleoperation

Keyboard:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Gamepad:
```bash
ros2 launch blue_demos joy_teleop.launch.yaml
```

### Upstream documentation

Full tutorials from the upstream `blue` package:

- [Simulation](https://robotic-decision-making-lab.github.io/blue/tutorials/simulation)
- [Control](https://robotic-decision-making-lab.github.io/blue/tutorials/control)
- [Teleoperation](https://robotic-decision-making-lab.github.io/blue/tutorials/teleop/)

## Related

- [alanchoi00/blue-sim-humble](https://github.com/alanchoi00/blue-sim-humble): Gazebo simulation (ROS2 Humble fork)
- [Robotic-Decision-Making-Lab/blue](https://github.com/Robotic-Decision-Making-Lab/blue): upstream blue package
- [Project board](https://github.com/users/alanchoi00/projects/7)
