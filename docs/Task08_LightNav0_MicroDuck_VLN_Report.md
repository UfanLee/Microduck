# Task 08 — LightNav-0 Visual-Language Navigation with MicroDuck

**Project:** Datawhale Every-Embodied — Task 08  
**Platform:** Local WSL2 Ubuntu 24.04 + Vast.ai remote NVIDIA RTX 5090  
**Status:** End-to-end integration achieved; MicroDuck visibly moved under VLN control  
**Date:** 2026-11-01 (Australia/Brisbane)

## 1. Objective

Integrate a vision-language navigation (VLN) model with a physically simulated MicroDuck robot. The system should accept a natural-language instruction, use first-person camera frames to infer a local trajectory, convert the trajectory into velocity commands via model predictive control (MPC), and execute those commands through the MicroDuck walking policy in MuJoCo.

**Success criterion:** The robot should physically move in the simulation following the VLN control pipeline, rather than merely displaying predicted waypoints.

## 2. System architecture

```text
User instruction + MuJoCo first-person frames (local WSL2)
                │
                ▼
Local vln-mujoco client (127.0.0.1:8000)
                │
                ▼
Encrypted SSH port forwarding (local 8050 → cloud 8050)
                │
                ▼
LightNav-0 VLN inference server (Vast.ai RTX 5090)
                │
                ▼
Predicted local trajectory (10 waypoints)
                │
                ▼
Local MPC controller → linear/angular velocity commands
                │
                ▼
MicroDuck ONNX walking policy (61 observations → 14 joint actions)
                │
                ▼
MuJoCo physics and robot motion
```

The inference model runs remotely, while MuJoCo rendering, MPC, and joint-level robot control run locally. The remote GPU does not automatically accelerate local MuJoCo rendering.

## 3. Environment

| Component | Setup / observation |
|---|---|
| Local machine | WSL2 Ubuntu 24.04; NVIDIA GeForce RTX 3060 Laptop GPU (6 GiB) |
| Local graphics | MuJoCo EGL; accelerated via Mesa D3D12 and NVIDIA GPU selection |
| Cloud instance | Vast.ai RTX 5090 (32 GiB), instance 58552195 |
| Cloud disk | 98 GiB writable filesystem confirmed with `df -h /workspace` |
| Cloud Python | Python 3.11.17 in `/workspace/lightnav-venv` |
| ML runtime | `torch==2.10.0`, `transformers==5.8.0`, `vllm==0.19.1` |
| Model | `LightOriginsHQ/LightNav-0`, downloaded locally on cloud host |
| Robot model | `robot_allcollisions.xml` (official demo-recommended variant) |
| Walking policy | `BEST_alpha_walking.onnx` (locally available, 61→14 interface) |
| Communication | SSH local port forwarding to remote port 8050 |

The cloud instance was listed at approximately **US$0.50/hour**, excluding any additional storage or related charges. This is a rental rate, not a measured total experiment cost.

## 4. Deployment and implementation

### 4.1 Cloud Python environment

```bash
cd /workspace
git clone https://github.com/lightorigins/LightNav-0.git
uv python install 3.11
uv venv --python 3.11 /workspace/lightnav-venv
cd /workspace/LightNav-0
uv pip install --python /workspace/lightnav-venv/bin/python -e '.[vllm,video]'
```

Installation completed successfully with 180 packages installed.

### 4.2 Download official model

```bash
/workspace/lightnav-venv/bin/hf download \
  LightOriginsHQ/LightNav-0 \
  --local-dir /workspace/LightNav-0/checkpoints/LightNav-0
```

The download finished successfully, fetching 20 files; the download log reported approximately 9.71 GB after reconstruction.

### 4.3 Start the VLN inference server

```bash
PORT=8050 /workspace/lightnav-venv/bin/lightnav-serve \
  --task vln \
  --model_path /workspace/LightNav-0/checkpoints/LightNav-0 \
  --backend vllm_local
```

Verified server output:

```text
[lightnav-ws] READY host=0.0.0.0 port=8050 task=vln K=256 H=10 backend=vllm_local
```

Other recorded initialization observations:

- Model checkpoint loading reported **8.7 GiB** GPU memory for model weights, not total final process VRAM.
- The model configured **64 history frames** and **10 predicted waypoints per chunk**.
- The one-shot warmup took approximately **22.19 seconds**. This must not be interpreted as steady-state inference latency.
- vLLM initialized successfully on the RTX 5090 with BF16 model execution.

### 4.4 SSH tunnel

After registering an SSH public key with Vast.ai and enabling it on the instance, use the following **local WSL2** command:

```bash
ssh -N -L 8050:127.0.0.1:8050 -p 14943 root@180.189.55.43
```

This host/port was the instance's reported SSH endpoint during the experiment; it may change if the instance is recreated. Keep the tunnel terminal open. Never upload the SSH private key to the cloud dashboard.

### 4.5 Launch the local MicroDuck client

```bash
cd ~/projects/LightNav-0/mujoco_demo
GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/vln-mujoco \
  --robot microduck \
  --robot-model ~/projects/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml \
  --walking-policy ~/projects/microduck-simulator/app/public/policies/BEST_alpha_walking.onnx \
  --vln-server ws://127.0.0.1:8050 \
  --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, connect to `ws://127.0.0.1:8050` if necessary, provide a language instruction, and select **Start VLN**.

## 5. Debugging: severe simulation slowdown

### 5.1 Symptom

The VLN client connected to the server and displayed a `RUNNING` state and **10 waypoints**, but the robot initially appeared stationary. Manual `W` control also showed little apparent displacement.

### 5.2 Initial diagnostics

The initial local simulation advanced about **0.2 simulated seconds during 10 seconds of wall-clock time**:

\[
\mathrm{RTF} = \frac{0.2}{10} \approx 0.02
\]

The `vln-mujoco` process showed approximately **900% CPU usage** in one sample, equivalent to about nine logical CPU cores, with approximately 3.1 GiB resident memory. The WSL2 instance reported 16 logical CPUs, approximately 7.6 GiB system RAM, and no swap use.

Tests with both `robot_groundcontact.xml` and `robot_allcollisions.xml`, and with restricted OpenMP / BLAS threads, did not remove the severe slowdown. A TurtleBot control test (without the MicroDuck ONNX walking policy) also progressed at about 0.2 simulated seconds per 10 real seconds. Therefore, the shared simulation/rendering path became the main suspect rather than MicroDuck's specific policy or collision geometry.

### 5.3 Identify software rendering

The first local EGL context reported:

```text
OpenGL Renderer: llvmpipe (LLVM 20.1.2, 256 bits)
OpenGL Vendor: Mesa
```

`llvmpipe` is a CPU software renderer. This provided direct evidence that the tested EGL context was not GPU accelerated.

### 5.4 Select the NVIDIA-backed WSL2 D3D12 renderer

The following environment variables were tested:

```bash
GALLIUM_DRIVER=d3d12
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
MUJOCO_GL=egl
```

After these were applied, the renderer query reported:

```text
Renderer: D3D12 (NVIDIA GeForce RTX 3060 Laptop GPU)
Vendor: Microsoft Corporation
```

With these variables applied to the TurtleBot simulator, simulation time advanced approximately **2 seconds per 10 real seconds**, a **10× improvement** over the initial measurement:

\[
\mathrm{RTF}_{\mathrm{after}} \approx 0.2
\]

This was a single approximate observation, not a repeated performance benchmark. Rendering remained slower than real time.

## 6. Observed outcome

**Confirmed by user observation:** After relaunching MicroDuck with the D3D12 renderer and connecting the client to LightNav-0, **MicroDuck visibly moved**.

The end-to-end pipeline therefore demonstrated more than waypoint generation: it produced visible robot motion with a running VLN model and a locally simulated robot.

**Limits of the available evidence:** No quantitative MicroDuck displacement, final task success rate, per-step latency distribution, or repeat-run RTF was recorded. The earlier UI showed a latency value of approximately 353 ms and 10 waypoints, but this alone is insufficient to establish steady-state inference performance or successful completion of a particular destination-reaching task.

## 7. Key technical lessons

1. **Modular autonomy:** Language-guided perception, waypoint prediction, MPC, policy-based actuation, and physical simulation form separate layers. A failure in robot motion is not automatically a failure of VLN inference.
2. **Validate bottom-up:** Check manual movement, simulation clock rate, and rendering before attributing poor navigation to the model.
3. **Rendering matters:** Running `MUJOCO_GL=egl` does not by itself guarantee hardware acceleration inside WSL2. Inspect the actual OpenGL renderer.
4. **Offload strategically:** The cloud RTX 5090 enabled the larger VLN model; local rendering still required independent GPU configuration.
5. **Distinguish predictions from execution:** `RUNNING` plus waypoints proves only part of the pipeline; visible physical motion is a stronger integration result, though not equivalent to quantified navigation-task success.

## 8. Follow-up experiments

- Record a full VLN episode including instruction, camera frames, trajectories, actual robot pose, and final outcome.
- Benchmark local MicroDuck simulation RTF with the accelerated renderer over multiple runs; separate physics, ONNX, MPC, and rendering costs.
- Compare `robot_allcollisions.xml` and `robot_groundcontact.xml` while holding other settings fixed.
- Measure model inference and end-to-end control latency over multiple commands.
- Test navigation to a concrete, visible indoor target, then repeat with multiple instructions to estimate task completion rate.
- Consider running the simulator on the cloud GPU only if local rendering remains too slow; verify that the remote host supports graphical rendering before migration.

## 9. References and provenance

- Official LightNav-0 repository: https://github.com/lightorigins/LightNav-0
- MuJoCo demo instructions: https://github.com/lightorigins/LightNav-0/tree/main/mujoco_demo
- Official model: https://huggingface.co/LightOriginsHQ/LightNav-0
- Measurements, configuration details, troubleshooting observations, and final movement outcome come from terminal logs and user observations recorded during this Task 08 session.

---

**Conclusion:** Task 08 achieved a working LightNav-0 → local MPC → MicroDuck walking-policy → MuJoCo motion pipeline. The most important engineering obstacle was WSL2's fallback to CPU software rendering (`llvmpipe`); forcing a D3D12-backed NVIDIA renderer restored visible motion and improved the observed TurtleBot simulation speed by approximately 10×. Full navigation-task success and real-time performance remain future measurement goals.
