# bluerov2-docking :anchor:

ROS2 simulation and perception stack for autonomous BlueROV2 underwater docking using ArUco markers and LED detection. UNSW Thesis research project.

<img src="assets/bluerov2_docking_demo.gif" alt="bluerov2_docking_demo_gif"/>

## Overview

The system drives the BlueROV2 onto a dock through a perception-to-control pipeline:

1. **Perception** fuses multiple ArUco markers into a single 6-DOF dock pose and smooths it with a Kalman filter, publishing the filtered pose plus a health signal. A simulated LED point cloud provides a complementary long-range cue.
2. **Coarse approach** (PBVS) servos on the filtered dock pose to drive the vehicle to a standoff point on the dock's entry axis, ready for handoff.
3. **Fine alignment** (impedance control, planned) takes over at close range for final docking.

### Packages

| Package | Description |
|---------|-------------|
| `description` | Docking station Gazebo model, world file, RViz + Foxglove configs |
| `sim` | Simulation launch file |
| `perception` | ArUco detection, multi-marker fusion + Kalman filter, LED mock publisher |
| `control` | Coarse approach PBVS controller |
| `interfaces` | Custom messages (filter health, coarse approach status) |

## Prerequisites

- **x86_64 (amd64) host required** — the base image and ArduSub SITL have no ARM builds. Mac (Apple Silicon) and ARM VMs are not supported.
- [Docker](https://docs.docker.com/get-docker/)
- NVIDIA GPU: `nvidia-container-toolkit` on the host
- Nouveau/Intel/AMD: no extra GPU setup required

## Setup

Clone the repo:

```bash
git clone https://github.com/alanchoi00/bluerov2-docking.git
cd bluerov2-docking
```

### Dev container (recommended)

Requires [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

1. Open the repo in VS Code and select **Reopen in Container** when prompted.
2. Choose your GPU variant:
   - `bluerov2-docking (Nouveau)`: AMD/Intel or no GPU
   - `bluerov2-docking (NVIDIA)`: NVIDIA GPU

VS Code will pull the image, build the container, and run `rosdep install` + `colcon build` automatically. The workspace is mounted so edits in VS Code are reflected inside the container immediately.

To update: pull the new image with `docker pull`, then run **Dev Containers: Rebuild Container** from the command palette.

### Docker Compose

1. Start the container:

   ```bash
   # Nouveau/Intel/AMD
   docker compose -f .docker/compose/nouveau-desktop.yaml up -d

   # NVIDIA
   docker compose -f .docker/compose/nvidia-desktop.yaml up -d
   ```

2. Open a shell and build:

   ```bash
   docker exec -it <container_name> bash
   cd /home/ubuntu/ws_docking
   rosdep install -y --from-paths src --ignore-src --rosdistro jazzy \
     --skip-keys='gz-transport13 gz-sim8 gz-math7 gz-msgs10 gz-plugin2'
   colcon build --symlink-install
   source install/setup.bash
   ```

3. Stop when done:

   ```bash
   docker compose -f .docker/compose/nouveau-desktop.yaml down
   ```

To update the base image:

```bash
docker pull ghcr.io/alanchoi00/blue-sim:jazzy-desktop        # Nouveau
docker pull ghcr.io/alanchoi00/blue-sim:jazzy-desktop-nvidia  # NVIDIA
```

## Simulation

The Gazebo environment is built on [alanchoi00/blue-sim](https://github.com/alanchoi00/blue-sim) (ROS2 Jazzy). Pre-built Docker images are published to GHCR.

### Launch

```bash
ros2 launch sim sim.launch.py
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_ardusub` | `true` | Start ArduSub SITL + MAVROS bridge |
| `flight_mode` | `POSHOLD` | ArduSub flight mode |
| `use_joy` | `false` | Joystick teleop |
| `use_key` | `false` | Keyboard teleop |
| `use_mock_led` | `true` | Publish simulated LED point cloud |
| `use_aruco` | `true` | Run ArUco marker detection |
| `use_docking_rviz` | `false` | Open RViz with docking config |
| `use_foxglove` | `false` | Start `foxglove_bridge` (WebSocket on port 8765) |
| `use_control` | `false` | Start the coarse approach PBVS controller (`control` package) |

E.g., launch without ArduSub for faster startup:

```bash
ros2 launch sim sim.launch.py use_ardusub:=false use_docking_rviz:=true
```

> The coarse controller (`use_control:=true`) assumes the vehicle owns horizontal control, i.e. `ALT_HOLD`. The default `POSHOLD` holds position and fights its sway/surge commands, so run it with `flight_mode:=ALT_HOLD`.

### Foxglove

[Foxglove](https://foxglove.dev/) (free tier) or [Lichtblick](https://github.com/lichtblick-suite/lichtblick) gives a single window for live 3D, camera, plots, logs, and MCAP replay — sharing the same WebSocket protocol, so either viewer works.

```bash
ros2 launch sim sim.launch.py use_foxglove:=true
```

Connect the viewer to `ws://localhost:8765` (open connection -> **Foxglove WebSocket**; the dev container runs with `--network=host`), then open the layout `src/description/foxglove/docking.json` from the app.

## Related

- [alanchoi00/blue-sim](https://github.com/alanchoi00/blue-sim): Gazebo simulation base
- [Robotic-Decision-Making-Lab/blue](https://github.com/Robotic-Decision-Making-Lab/blue): upstream blue package
