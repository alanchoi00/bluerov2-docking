# bluerov2-docking :anchor:

ROS2 simulation and perception stack for autonomous BlueROV2 underwater docking using ArUco markers and LED detection. UNSW Thesis research project.

## Overview

The system uses a two-level visual guidance pipeline:

| Level | Guidance | Range | Method |
|-------|----------|-------|--------|
| I | Coarse approach | Long range | LED cluster detection -> centroid-based IBVS |
| II | Fine alignment | Close range | ArUco PnP -> 6-DOF pose error |

### Packages

| Package | Description |
|---------|-------------|
| `description` | Docking station Gazebo model, world file, RViz config |
| `sim` | Simulation launch file |
| `perception` | LED mock publisher, ArUco detection relay |

## Prerequisites

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

E.g., launch without ArduSub for faster startup:

```bash
ros2 launch sim sim.launch.py use_ardusub:=false use_docking_rviz:=true
```

## Related

- [alanchoi00/blue-sim](https://github.com/alanchoi00/blue-sim): Gazebo simulation base
- [Robotic-Decision-Making-Lab/blue](https://github.com/Robotic-Decision-Making-Lab/blue): upstream blue package
