# Task 07 — MicroDuck RDK Edge & Web Deployment

## 1. Objective

Reproduce the Datawhale Task 07 workflow from browser-based ONNX inference to a separate TCP policy server and MuJoCo closed-loop simulation. The current tests use **WSL2 Ubuntu and a local CPU inference server, not physical RDK X5 hardware**.

## 2. Hardware and software

- Host: Windows + WSL2 Ubuntu 24.04
- Project: `datawhalechina/every-embodied`, Task `07-RDK端侧与网页部署`
- Robot definition: `pollen-robotics/microduck_rl`, `robot_groundcontact.xml`
- ONNX model: `BEST_alpha_walking.onnx`
- ONNX SHA256: `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`
- Contract: `[1,61]` observation to `[1,14]` joint action
- Model runner: ONNX Runtime 1.31.1, CPU (reported by completed simulation)
- Simulator: MuJoCo 3.17.0 (reported by completed simulation)
- Server: `rdk_policy_server.py`, MDP1 persistent TCP on `127.0.0.1:8765`
- Rendering: EGL; video encoding: FFmpeg; actuator approximation: `--no-bam` (MuJoCo position actuators)

## 3. Browser deployment validation

Cloned the official MicroDuck Sandbox with Git LFS, installed Linux Node.js in WSL2, ran the Vite development server on port 5175, and tested the browser simulation. Walking, turning, left/right kicking, sitting, standing, and resetting worked in the user's manual inspection.

All nine policy files successfully passed one-step finite-value inference with `obs [1,61] -> actions [1,14]`:

- `BEST_alpha_sitstand.onnx`
- `BEST_alpha_stand.onnx`
- `BEST_alpha_walking.onnx`
- `BEST_roller.onnx`
- `BEST_roller_crouch.onnx`
- `alpha_ground_pick.onnx`
- `ball_kick_left.onnx`
- `ball_kick_right.onnx`
- `roulade.onnx`

**Caution:** One-step finite-value inference verifies shape and computational validity, not safe or successful robot behavior in unseen conditions.

## 4. Local inference and TCP benchmarks

| Metric | CPU-only ONNX | Local TCP + ONNX |
|---|---:|---:|
| Mean | 0.020 ms | 0.141 ms |
| P95 | 0.027 ms | 0.203 ms |
| P99 | 0.038 ms | 0.263 ms |
| Maximum | 0.184 ms | 0.718 ms |
| Samples over 20 ms | 0 / 1000 | 0 / 1000 |

The CPU-only test used 20 warmups then 1,000 timed inferences. The local TCP test used 1,000 round trips over a persistent loopback connection. The mean additional cost was approximately 0.121 ms. Neither benchmark measures actual RDK X5 CPU/BPU performance, real network conditions, actuator latency, or end-to-end hard real-time behavior.

## 5. Single-MicroDuck closed-loop validation (completed)

The client used genuine MuJoCo observations to request an action from the standalone CPU policy server on each simulation control step, applied the returned action, advanced the physics engine, and wrote a video and JSON report.

**Command (from the Task 07 source directory):**

```bash
MUJOCO_GL=egl uv run --no-project --with numpy --with mujoco --with onnxruntime --with pillow python rdk_microduck_video.py \
  --microduck-repo ~/projects/microduck_rl \
  --policy-host 127.0.0.1 --policy-port 8765 \
  --output ~/projects/task07_tcp_duck.mp4 \
  --duration 5 --fps 30 --no-bam
```

| Measure | Result |
|---|---:|
| Number of ducks | 1 |
| Simulated time | 5.0 s |
| Control frequency | 50 Hz |
| Control requests | 250 |
| Wall time | 97.8316 s |
| Realtime factor | 0.0511 |
| Mean policy batch/roundtrip | 0.8730 ms |
| P50 policy batch/roundtrip | 0.8509 ms |
| P90 policy batch/roundtrip | 1.4178 ms |
| P99 policy batch/roundtrip | 2.3212 ms |
| Max policy batch/roundtrip | 3.4209 ms |
| End x displacement | +0.60079 m |
| End y displacement | -0.11181 m |
| Minimum trunk height | 0.11301 m |
| Video | 150 frames; 960 × 540; 30 fps |

Local files created on the user's WSL system:
- `~/projects/task07_tcp_duck.mp4`
- `~/projects/task07_tcp_duck.json`

Movement was recorded, but a full visual review of the MP4 is not yet documented; displacement alone is not a definitive walking-success measure.

### Interpretation

Policy roundtrip latency was below the 20 ms nominal control period in this experiment. **However, the overall simulation ran at approximately 0.05× real time**, likely due to physics simulation, offscreen rendering and video encoding overhead (breakdown not yet profiled). Thus, this is a valid simulated closed-loop experiment, not a demonstration of complete real-time operation.

The script's JSON field `"platform": "RDK X5"` is a static label from the upstream script. **No physical RDK X5 was used**.

The `--no-bam` setting disables the detailed BAM actuator dynamics, so these motion results must not be used to claim real-servo fidelity or successful Sim-to-Real deployment.

## 6. 3 × 3 formation evaluation (completed)

The WSL2 client used the same MDP1 policy server and the identical walking ONNX model to control nine independent robot instances in one MuJoCo scene. Each 50 Hz control step sent a batch of nine 61-dimensional observations and received nine 14-dimensional actions. There were **250 batch requests and 2,250 individual model inferences**, not 2,250 TCP round trips. This test is multi-instance inference, **not multi-agent coordination or formation control**.

**Command (from the Task 07 source directory):**

```bash
MUJOCO_GL=egl uv run --no-project --with numpy --with mujoco --with onnxruntime --with pillow python rdk_microduck_video.py \
  --microduck-repo ~/projects/microduck_rl \
  --policy-host 127.0.0.1 --policy-port 8765 \
  --output ~/projects/task07_tcp_duck_3x3.mp4 \
  --rows 3 --cols 3 --duration 5 --fps 30 \
  --camera-distance 1.65 --no-bam
```

| Measure | Observed result |
|---|---:|
| Formation | 3 rows × 3 columns; 9 distinct physics instances |
| Simulated duration | 5.0 s |
| Control frequency | 50 Hz |
| Batch requests | 250 |
| Individual policy inferences | 2,250 |
| Wall time | 839.7532 s |
| Realtime factor | 0.005954 |
| Mean batch roundtrip | 1.5205 ms |
| P50 batch roundtrip | 1.1648 ms |
| P90 batch roundtrip | 2.8406 ms |
| P99 batch roundtrip | 4.2615 ms |
| Maximum batch roundtrip | 14.8739 ms |
| 3×3 video | 150 frames; 960 × 540; 30 fps; 2,135,566 bytes |
| ONNX SHA256 | `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c` |

**Per-robot displacement (metres, simulated 5 s):**

| Robot | Δx | Δy | Mean forward speed (m/s) |
|---|---:|---:|---:|
| duck0 | +0.600373 | -0.111352 | 0.120075 |
| duck1 | +0.594899 | -0.108867 | 0.118980 |
| duck2 | +0.601192 | -0.112379 | 0.120238 |
| duck3 | +0.600779 | -0.111713 | 0.120156 |
| duck4 | +0.601161 | -0.111426 | 0.120232 |
| duck5 | +0.600784 | -0.111768 | 0.120157 |
| duck6 | +0.600992 | -0.111900 | 0.120198 |
| duck7 | +0.601165 | -0.111393 | 0.120233 |
| duck8 | +0.601209 | -0.111388 | 0.120242 |

All nine reported the same minimum trunk height, approximately **0.113014 m**. The forward displacements cluster around 0.60 m, consistent with independent instances receiving the same policy and command, but this is not evidence of formation maintenance or collision avoidance. Visual stability remains to be verified from the user's MP4.

**Comparison to the one-duck run:**

| Measure | 1 duck | 9 ducks |
|---|---:|---:|
| Mean policy call roundtrip | 0.8730 ms | 1.5205 ms |
| P99 | 2.3212 ms | 4.2615 ms |
| Max | 3.4209 ms | 14.8739 ms |
| Simulated 5 s wall time | 97.8316 s | 839.7532 s |
| Realtime factor | 0.051108 | 0.005954 |

The nine-duck **batch roundtrip** is ~1.74× the one-duck mean for nine model evaluations, illustrating modest relative inference/transport overhead. However, the full simulation is ~8.59× more expensive in wall time. Physics, rendering, and video encoding were not profiled separately, so assigning a precise dominant bottleneck is not justified. Both simulations ran far below real time. The maximum observed batch roundtrip (14.87 ms) was below the nominal 20 ms budget, but neither the CPU-on-WSL2 result nor an off-line simulation proves hard real-time or RDK X5 performance.

User WSL artifacts:
- `~/projects/task07_tcp_duck_3x3.mp4`
- `~/projects/task07_tcp_duck_3x3.json`

As with the single-duck test, `"platform": "RDK X5"` is an upstream hardcoded report label: the test **did not run on an RDK board**. The `--no-bam` flag substituted generic MuJoCo position actuators, limiting physical fidelity.

## 7. Troubleshooting log

1. Windows npm invoked through WSL (UNC path / `CMD.EXE`): switched to Linux Node.js via nvm.
2. Missing `pip`: used `uv run --no-project --with ...` instead of modifying system Python.
3. MuJoCo OpenGL import failure: supplied `MUJOCO_GL=egl`.
4. Missing `PIL`: added `--with pillow`.
5. Unavailable PyPI requirement `bam-mujoco`: removed it and used `--no-bam` for a simplified simulation.
6. Missing FFmpeg executable: installed OS `ffmpeg`.

## 8. Conclusions and future work

The browser successfully loaded and executed nine distinct ONNX policies, and a local TCP policy server was able to control a real MuJoCo simulation through 61-dimensional observations and 14-dimensional actions. The baseline and local loopback latency distributions are documented. The 3 × 3 experiment is now completed and its measured results are documented. Remaining work is to (a) visually inspect the recorded single- and nine-duck videos, (b) profile full-loop timing and evaluate physically realistic actuator dynamics, and (c) eventually benchmark on a real RDK X5 and implement robot-specific safety checks.

## References

- [Datawhale — Task 07: RDK 端侧与网页部署](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/05-MicroDuck%E8%BF%90%E5%8A%A8%E7%AD%96%E7%95%A5%E4%B8%8E%E4%BB%BB%E5%8A%A1%E5%BC%80%E5%8F%91%E4%B8%93%E9%A2%98/01-%E4%BB%BB%E5%8A%A1%E8%B5%84%E6%96%99/07-RDK%E7%AB%AF%E4%BE%A7%E4%B8%8E%E7%BD%91%E9%A1%B5%E9%83%A8%E7%BD%B2)
- [Pollen Robotics — microduck_rl](https://github.com/pollen-robotics/microduck_rl)
- [Pollen Robotics — MicroDuck Simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
