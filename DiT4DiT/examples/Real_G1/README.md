# Teleop, Training & Deployment Tutorial

A deployment toolchain wrapping NVIDIA GR00T Whole-Body Control to drive a real Unitree G1: data collection via teleoperation, replay, training and closed-loop policy inference.

---

## 0. Table of Contents

- [1. System Overview](#1-system-overview)
- [2. Hardware & Network](#2-hardware--network)
- [3. Installation](#3-installation)
- [4. Teleop Data Collection](#4-teleop-data-collection)
- [5. Data Replay](#5-data-replay)
- [6. Training on Real-Robot Data](#6-training-on-real-robot-data)
- [7. Closed-loop Inference](#7-closed-loop-inference)
- [8. Pico Controller Reference](#8-pico-controller-reference)

---

## 1. System Overview

The full stack runs across two machines:

| Role | Machine | Main processes |
|---|---|---|
| **Local host** | x86 workstation | G1 control loop, teleop policy, camera viewer, inference client, inference ZMQ server (default port `5556`) |
| **Robot side (Jetson Orin)** | Mounted on the G1, IP `192.168.123.164` | ALOHA gripper DDS bridge, RealSense camera server |
---

## 2. Hardware & Network

### Required Hardware
- **Unitree G1** humanoid (29-DoF, with ALOHA gripper)
- **Local host**
- **Jetson Orin** mounted on the robot, IP set to `192.168.123.164`
- **Pico 4 / Pico 4 Pro** VR headset (teleop controller)
- **ALOHA2 gripper**

### Network Setup
- Local host: static IP `192.168.123.222`, netmask `255.255.255.0`, on the same subnet as the G1
- Pico headset on the same WiFi as the local host

See the [Unitree G1 Developer Guide](https://support.unitree.com/home/en/G1_developer) for the robot networking details.

---

## 3. Installation

The project ships as Docker containers. Local host and Jetson Orin each have their own image.

### 3.1 Common Prerequisites
```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
```
Docker + the NVIDIA Container Toolkit must be available on the host (`run_docker.sh` will install the toolkit automatically if missing).

### 3.2 Local Host Setup

```bash
mkdir -p ~/Projects && cd ~/Projects
git clone https://github.com/Mondo-Robotics/DiT4DiT.git
cd decoupled_wbc

# Pull the prebuilt image and enter the container (recommended)
./docker/run_docker.sh --install --root

# Re-enter the container later
./docker/run_docker.sh --root
```

To rebuild locally (only needed when you change Dockerfile or dependencies):
```bash
./docker/run_docker.sh --build --root
```

### 3.3 Jetson Orin (Robot Side) Setup

```bash
ssh unitree@192.168.123.164
mkdir -p ~/Projects && cd ~/Projects
git clone https://github.com/Mondo-Robotics/DiT4DiT.git
cd decoupled_wbc

./docker/run_docker_jetson.sh --root
```

### 3.4 Sanity Check

Inside the container:
```bash
python -c "import gr00t_wbc, torch, rclpy; print('OK')"
```
If it prints `OK`, the environment is ready.

---

## 4. Teleop Data Collection

### 4.1 One-shot Launch (Recommended)

On the local host:
```bash
cd ~/Projects/decoupled_wbc
./docker/run_docker.sh --root

# inside the container
python scripts/deploy_g1_real_robot.py --data_collection
```

This script automatically:
1. SSHs to the Jetson and starts the ALOHA gripper bridge and RealSense camera server
2. Starts the G1 control loop locally (DDS to the G1)
3. Starts the camera viewer
4. Starts the data exporter

Recorded data lands in `g1_real_data/<task>/data/chunk-XXX/episode_XXXXXX.parquet`.

### 4.2 Manual Launch (Step-by-step)
**(1) Jetson Orin: gripper server**
```bash
ssh unitree@192.168.123.164
cd decoupled_wbc
python aloha_feedback_server/aloha_gripper_dds_bridge.py
```

**(2) Jetson Orin: camera server**
```bash
ssh unitree@192.168.123.164
docker rm -f gr00t_wbc_jetson
./decoupled_wbc/docker/run_docker_jetson.sh --root

python gr00t_wbc/control/sensor/composed_camera.py \
    --ego_view_camera realsense \
    --port 5555 \
    --server
```

**(3) Local host: G1 control loop**
```bash
cd ~/Projects/decoupled_wbc
./docker/run_docker.sh --root

python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
    --interface real \
    --with_hands \
    --hand_type aloha
```

**(4) Local host: teleop policy + data exporter**
```bash
# Teleop policy (Pico-driven)
python gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
    --hand_control_device pico --body_control_device pico

# Data exporter (records observations + actions to parquet)
python gr00t_wbc/control/main/teleop/run_g1_data_exporter.py \
    --camera_host 192.168.123.164 --camera_port 5555 \
    --data_collection_frequency 30 \
    --root_output_dir ./g1_real_data
```

**(5) Local host (optional): camera viewer**
```bash
python gr00t_wbc/control/main/teleop/run_camera_viewer.py \
    --camera-host 192.168.123.164 \
    --camera-port 5555
```

---

## 5. Data Replay
1. **Start the gripper server**:
   ```bash
   ssh unitree@192.168.123.164
   cd decoupled_wbc
   python aloha_feedback_server/aloha_gripper_dds_bridge.py
   ```

2. **Start the G1 control loop**:
   ```bash
   ./docker/run_docker.sh --root
   python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
       --interface real --with_hands --hand_type aloha
   ```

3. **Replay a `.parquet` episode**:
   ```bash
   ./docker/run_docker.sh --root

   # Upper-body replay (arms + hands)
   python gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
       --lerobot_replay_path g1_real_data/test/data/chunk-000/episode_000000.parquet

   # Whole-body replay (with waist + legs)
   python gr00t_wbc/control/main/teleop/run_teleop_policy_loop.py \
       --compact_replay_path g1_real_data/test/data/chunk-000/episode_000000.parquet
   ```

---

## 6. Training on Real-Robot Data

After collecting episodes via teleop (Section 4), train a DiT4DiT policy on them with `examples/Real_G1/train_files/run_decoupled_wbc.sh`. This must be run from a checkout of the upstream `DiT4DiT` project (where the `DiT4DiT` Python package and training entry point live), not from the `decoupled_wbc` repo.

### 6.1 Prerequisites
- A working DiT4DiT training environment (conda env + `pip install -e .` in the DiT4DiT repo root).
- Cosmos-Predict2.5-2B base VLM weights downloaded locally.
- Dataset converted to the LeRobot parquet layout produced by Section 4 (`g1_real_data/<task>/data/chunk-XXX/episode_XXXXXX.parquet`).
- (Optional) A pretrained DiT4DiT checkpoint to warm-start from.

### 6.2 Register Your Dataset in the Mixture

The script trains on the mixture named `g1_decoupled_wbc`. Open `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py` and replace the placeholder entry with your own dataset name(s):

```python
"g1_decoupled_wbc": [
    ("<your_g1_dataset_name>", 1.0, "g1_body29_aloha_full_body"),
    # add more (name, weight, robot_config) tuples to mix datasets
],
```

The first field is the directory name under `data_root_dir`; the third (`g1_body29_aloha_full_body`) is the robot config and should match the embodiment used during data collection.

### 6.3 Edit the Launch Script

Open `examples/Real_G1/train_files/run_decoupled_wbc.sh` and fill in the placeholders:

| Variable | Meaning |
|---|---|
| `WANDB_API_KEY` | Your W&B key (or leave `WANDB_MODE=offline`) |
| `base_vlm` | Path to the Cosmos-Predict2.5-2B checkpoint |
| `data_root_dir` | Parent directory holding your collected episode datasets |
| `data_mix` | Mixture name in `mixtures.py` (default `g1_decoupled_wbc`) |
| `pretrained_ckpt` | Path to a DiT4DiT checkpoint to warm-start from (or empty to train from scratch) |
| `run_root_dir` / `run_id` | Where to write checkpoints and logs |
| `wandb_entity` | Your W&B entity (only needed when `WANDB_MODE` is online) |

### 6.4 Launch Training

From the DiT4DiT project root:

```bash
# Make sure your conda env is active and `pip install -e .` has been run.
cd /path/to/DiT4DiT
bash examples/Real_G1/train_files/run_decoupled_wbc.sh
```

The script copies itself into `${run_root_dir}/${run_id}/` for reproducibility. Checkpoints are written under the same directory and can be passed to `server_zmq.sh` (see Section 7) as the inference checkpoint.

---

## 7. Closed-loop Inference

Inference requires several processes running concurrently. A multi-pane tmux layout works well.

### 7.1 Window Layout

| Location | Role | Command |
|---|---|---|
| Inference server | Model ZMQ server (any GPU machine, local or remote) | `server_zmq.sh` (default port 5556) |
| Jetson - Gripper | ALOHA gripper bridge | `aloha_gripper_dds_bridge.py` |
| Jetson - Camera | Camera server | `composed_camera.py --server` |
| Local host - Viewer | Live camera preview | `run_camera_viewer.py` |
| Local host - Control | G1 control loop | `run_g1_control_loop.py` |
| Local host - Client | Inference client | `run_dit4dit_inference_policy_loop.py` |

### 7.2 Startup Sequence

**(1) Inference server**

Bring up the DiT4DiT ZMQ server on any GPU machine you have available. The launch script `server_zmq.sh` and the entry point `deployment/model_server/server_policy_zmq.py` are included in this repo for reference, but their Python imports (`DiT4DiT.*`, `deployment.model_server.tools.*`) live in the upstream `DiT4DiT` project, so you must run them from a checkout of that repo where the full package is installed.

```bash
# Inside the upstream DiT4DiT project root (with the conda env active and `pip install -e .` done):
# 1. Edit server_zmq.sh and set ckpt_path to point at your trained checkpoint.
# 2. Launch the server:
bash server_zmq.sh
```

The server listens on port 5556 by default. Verify it is up:
```bash
ss -tlnp | grep 5556
```

**(2) Jetson: gripper server**
```bash
ssh unitree@192.168.123.164
cd decoupled_wbc
python aloha_feedback_server/aloha_gripper_dds_bridge.py
```

**(3) Jetson: camera server**
```bash
ssh unitree@192.168.123.164
docker rm -f gr00t_wbc_jetson
./decoupled_wbc/docker/run_docker_jetson.sh --root

python gr00t_wbc/control/sensor/composed_camera.py \
    --ego_view_camera realsense --port 5555 --server
```

**(4) Local host: camera viewer**
```bash
cd ~/Projects/decoupled_wbc
./docker/run_docker.sh --root

python gr00t_wbc/control/main/teleop/run_camera_viewer.py \
    --camera-host 192.168.123.164 \
    --camera-port 5555 \
    --infer_video_path infer_video/dit4dit/test
```

**(5) Local host: G1 control loop**
```bash
./docker/run_docker.sh --root
python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
    --interface real --with_hands --hand_type aloha
```

**(6) Local host: inference client (DiT4DiT)**
```bash
./docker/run_docker.sh --root
python gr00t_wbc/control/main/teleop/run_dit4dit_inference_policy_loop.py \
    --inference_host localhost \
    --inference_port 5556 \
    --hand_type aloha \
    --enable_wbc \
    --language_instruction "place each can of cola on the table into the box on the conveyor belt." \
    --camera_host 192.168.123.164 \
    --camera_port 5555 \
    --action_horizon 50 \
    --action_execution_horizon 32 \
    --inference_frequency 30.0
```
---

## 8. Pico Controller Reference

Install the [XR-Robotics teleop app](https://github.com/XR-Robotics) on the Pico headset and connect it to the same WiFi as the local host.

**Policy Control:**
- `Left menu + Left trigger (>0.98) + Left grip (>0.98)`: toggle lower-body policy
- `Left menu + Right grip (>0.98)`: toggle upper-body teleop activation

**Navigation (lower body):**
- `Left stick Y-axis`: forward/backward (max ±0.5 m/s)
- `Left stick X-axis`: left/right strafe (max ±0.5 m/s)
- `Right stick X-axis`: yaw rotation (max ±1.0 rad/s)

**Height Control (hold to keep adjusting, step 0.01 m / frame):**
- `X (right controller)`: lower base height (min 0.24 m)
- `Y (right controller)`: raise base height (max 0.74 m)

**Arm Control (IK):** left/right controller 6DoF pose → left/right wrist IK target.

**Hand / Gripper (Left menu NOT pressed):**
- `Trigger > 0.5`: index finger
- `Trigger > 0.5 + Grip > 0.5`: middle finger
- `Grip > 0.5`: ring finger
- Thumb is always open by default

**Data Recording:**

Starting / saving an episode is a **two-step operation** (source: [`teleop_policy.py`](gr00t_wbc/control/policy/teleop_policy.py#L197)):

1. **Press `A`**: teleop is disabled immediately; the robot smoothly returns to its initial pose
2. **Press `Left menu + Right grip (>0.98)`**: re-enable teleop. Only at this moment is the `toggle_data_collection` signal actually sent
   - 1st full cycle = start recording
   - 2nd full cycle = stop and save the current episode

Delete the most recently saved episode (single-step shortcut):
- **`Y + Left menu`** pressed together: directly emits `toggle_delete_episode` to delete the previous episode

> ℹ️ The intent is to prevent accidental triggers: pressing `A` first repositions the robot, the operator confirms safety, and only then `Left menu + Right grip` re-activates teleop and actually toggles the recording state.

---

## License

See `LICENSE`. This repository extends NVIDIA `gr00t_wbc` and retains the original Apache 2.0 license.
