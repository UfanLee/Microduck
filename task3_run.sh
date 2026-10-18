#!/usr/bin/env bash
# Datawhale Every-Embodied / MicroDuck Task 3
# Reproduce 2m stilts, run 2cm PPO smoke test, and write a factual GitHub report.
set -uo pipefail

ROOT="$HOME/projects/microduck-stilts"
DEST="/mnt/c/Users/LXH20/Desktop/Microduck"
MEDIA="$DEST/media"
DOCS="$DEST/docs"
LOGS="$DEST/logs"
MODEL="$ROOT/policies/stilts/200cm"
VIDEO="$MODEL/videos/play/rl-video-step-0.mp4"
OUTPUT_VIDEO="$MEDIA/MicroDuck_Stilts_200cm.mp4"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Project not found: $ROOT"
  exit 1
fi
mkdir -p "$MEDIA" "$DOCS" "$LOGS" "$ROOT/artifacts"
exec > >(tee "$LOGS/Task3_Execution.log") 2>&1
cd "$ROOT"

echo "=== MicroDuck Task 3 — $(date -Iseconds) ==="
echo "Project: $ROOT"
echo "Windows output: $DEST"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

video_status="FAILED"
training_status="FAILED"
onnx_status="NOT CHECKED"
video_desc="2m video not successfully produced"
train_desc="PPO smoke test not successfully completed"
video_link="2 m 视频尚未成功生成；请查看 Task3_200cm_Export.log。"

echo ""
echo "=== 1 / 4 Download the public 2m checkpoint ==="
if [[ -s "$MODEL/policy.onnx" && -s "$MODEL/checkpoint.pt" && -s "$MODEL/manifest.json" ]]; then
  echo "2m model files already exist; skipping download."
  download_ok=1
else
  if uv run hf download HannesVonEssen/microduck-stilts \
      200cm/policy.onnx 200cm/checkpoint.pt 200cm/manifest.json \
      --local-dir "$ROOT/policies/stilts"; then
    download_ok=1
  else
    download_ok=0
    echo "WARNING: Hugging Face download failed."
  fi
fi
ls -lh "$MODEL" 2>/dev/null || true

echo ""
echo "=== 2 / 4 ONNX interface check ==="
if [[ "$download_ok" -eq 1 && -s "$MODEL/policy.onnx" ]]; then
  if uv run python - <<'PY'
import numpy as np
import onnxruntime as ort
p = "policies/stilts/200cm/policy.onnx"
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
inp, output = s.get_inputs()[0], s.get_outputs()[0]
result = s.run(None, {inp.name: np.zeros((1, 61), dtype=np.float32)})[0]
print("ONNX:", inp.name, inp.shape, "->", output.name, output.shape, "actual", result.shape)
assert result.shape == (1, 14), f"Unexpected action shape: {result.shape}"
PY
  then
    onnx_status="PASSED"
  else
    onnx_status="FAILED"
  fi
fi

echo ""
echo "=== 3 / 4 2m MuJoCo rollout (500 frames @ 50Hz) ==="
if [[ "$download_ok" -eq 1 && -s "$MODEL/checkpoint.pt" ]]; then
  export MICRODUCK_STILT_HEIGHT_CM=200
  export MICRODUCK_STILT_BLEND=0.5
  marker="$(mktemp "$ROOT/.task3_export_start_XXXXXXXX")"
  if uv run python scripts/export.py Mjlab-Stilt-Flat-MicroDuck \
      --checkpoint-file policies/stilts/200cm/checkpoint.pt \
      --device cpu \
      --num-envs 1 \
      --video True \
      --video-length 500 \
      --video-height 720 \
      --video-width 1280 \
      --viewer auto \
      --running-speed 0.15 \
      --onnx-file artifacts/stilts_200cm.onnx \
      2>&1 | tee "$LOGS/Task3_200cm_Export.log"; then
    if [[ -s "$VIDEO" && "$VIDEO" -nt "$marker" ]]; then
      if cp -f "$VIDEO" "$OUTPUT_VIDEO"; then
        video_status="PASSED"
        video_desc="2m checkpoint rollout generated, new nonempty MP4 copied to Windows Desktop"
        video_link="[2 m 本地回放视频](../media/MicroDuck_Stilts_200cm.mp4)"
      fi
    else
      echo "WARNING: Export returned success but a newly generated MP4 was not found: $VIDEO"
    fi
  else
    echo "WARNING: 2m rollout command failed; consult $LOGS/Task3_200cm_Export.log"
  fi
  rm -f "$marker"
else
  echo "SKIPPED: 2m checkpoint is missing."
fi

if [[ "$video_status" == "PASSED" ]]; then
  uv run python - <<'PY' || true
import imageio.v3 as iio
p = "policies/stilts/200cm/videos/play/rl-video-step-0.mp4"
try:
    meta = iio.immeta(p, plugin="ffmpeg")
    print("Recorded video metadata:", {k: meta.get(k) for k in ("duration", "fps", "size", "codec")})
except Exception as e:
    print("Metadata could not be checked:", e)
PY
fi

echo ""
echo "=== 4 / 4 PPO smoke test: 2cm stilts, 64 envs, 5 iterations ==="
export MICRODUCK_STILT_HEIGHT_CM=2
export MICRODUCK_STILT_BLEND=0
export WANDB_MODE=disabled

if uv run python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  if uv run train Mjlab-Stilt-Flat-MicroDuck \
      --env.scene.num-envs 64 \
      --agent.max-iterations 5 \
      2>&1 | tee "$LOGS/Task3_PPO_Smoke.log"; then
    training_status="PASSED"
    train_desc="64 parallel environments, 5 PPO iterations; process exited successfully"
  else
    train_desc="PPO command returned an error; inspect Task3_PPO_Smoke.log (e.g. device memory or configuration)"
    echo "WARNING: PPO smoke test failed; the 2m video (if generated) remains intact."
  fi
else
  training_status="SKIPPED"
  train_desc="No CUDA-compatible PyTorch device detected in this environment"
  echo "WARNING: CUDA unavailable. Training smoke test skipped."
fi

# Optionally save an existing 25cm video without falsely claiming it was reproduced.
VIDEO25="$ROOT/policies/stilts/25cm/videos/play/rl-video-step-0.mp4"
if [[ -s "$VIDEO25" ]]; then
  cp -f "$VIDEO25" "$MEDIA/MicroDuck_Stilts_25cm.mp4"
  note25="Found and copied an existing 25cm MP4 (creation conditions should be confirmed separately)."
  link25="[25 cm 视频](../media/MicroDuck_Stilts_25cm.mp4)"
else
  note25="25cm output video was not found by this script; earlier steps should be verified separately."
  link25="25 cm 视频待确认"
fi

export TASK3_VIDEO_STATUS="$video_status" TASK3_TRAINING_STATUS="$training_status"
export TASK3_ONNX_STATUS="$onnx_status"
export TASK3_VIDEO_DESC="$video_desc" TASK3_TRAIN_DESC="$train_desc"
export TASK3_VIDEO_LINK="$video_link" TASK3_NOTE25="$note25" TASK3_LINK25="$link25"
export TASK3_ROOT="$ROOT" TASK3_DEST="$DEST"

python3 - <<'PY'
from pathlib import Path
import os
from datetime import datetime, timezone
root=Path(os.environ["TASK3_DEST"])
report=root/"docs"/"Task3_Stilt_Walking.md"
commit="unknown"
try:
    import subprocess
    commit=subprocess.check_output(["git","-C",os.environ["TASK3_ROOT"],"rev-parse","HEAD"],text=True).strip()
except Exception:
    pass
s=f"""# MicroDuck Task 3 — Stilt Walking Reinforcement Learning Reproduction

**Datawhale Every-Embodied · 高跷行走强化学习复现**  
**Recorded:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}  
**Code version:** `{commit}`  
**Platform:** Windows + WSL2 Ubuntu, Python 3.12, uv, MuJoCo / mjlab

## 1. 摘要

本任务以公开的 MicroDuck 高跷行走强化学习项目为基础，复现其策略推理与物理仿真流程，并执行一次小规模 PPO 训练冒烟测试。本文将**公开 checkpoint 的推理回放**与**本机新的 PPO 训练验证**分开记录，避免把预训练模型误写为从零训练的成果。

## 2. 高跷形态及课程学习

官方方案将高跷作为真实参与 MuJoCo 接触计算的机器人部件，而非仅改变画面外观。高跷高度、质量、惯量、碰撞体与足端接触位置改变了控制难度。

官方课程先在 2 cm 宽支撑形态学习，再逐步收窄支撑，最终逐级增加高度，直至约 2 m 的仿真高跷。不同高度配合不同策略，不能假定 25 cm 模型直接兼容 2 m 形态。

策略通常采用 61 维本体感觉观测，输出 14 维关节动作，控制频率 50 Hz。训练利用 PPO 及 actor–critic，部署时主要使用 actor。

## 3. 实验环境

- **实验目录：** `~/projects/microduck-stilts`
- **项目任务：** `Mjlab-Stilt-Flat-MicroDuck`
- **先前配置测试：** 23 passed（用户终端观察，耗时 30.38 秒）
- **策略来源：** [HannesVonEssen/microduck-stilts](https://huggingface.co/HannesVonEssen/microduck-stilts)
- **训练与仿真源码：** [Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground)

## 4. 2 m 策略推理与回放（复现）

设置：

```bash
export MICRODUCK_STILT_HEIGHT_CM=200
export MICRODUCK_STILT_BLEND=0.5
uv run python scripts/export.py Mjlab-Stilt-Flat-MicroDuck \\
  --checkpoint-file policies/stilts/200cm/checkpoint.pt \\
  --device cpu --num-envs 1 \\
  --video True --video-length 500 \\
  --video-width 1280 --video-height 720 \\
  --viewer auto --running-speed 0.15 \\
  --onnx-file artifacts/stilts_200cm.onnx
```

- **ONNX 接口检查：** {os.environ['TASK3_ONNX_STATUS']}
- **2 m 视频生成：** {os.environ['TASK3_VIDEO_STATUS']}
- **执行记录：** {os.environ['TASK3_VIDEO_DESC']}
- **视频证据：** {os.environ['TASK3_VIDEO_LINK']}
- **复现日志：** [Task3_200cm_Export.log](../logs/Task3_200cm_Export.log)

**解释范围：** 视频生成只能证明公开 actor 能在本地物理环境执行并产生帧；是否有稳定步态、跌倒、明显异常，仍应打开视频进行视觉检查。该视频不代表从零训练出了 2 m 策略，也不代表真机可安全使用。

## 5. PPO 训练冒烟测试（新执行的优化过程）

设置：

```bash
export MICRODUCK_STILT_HEIGHT_CM=2
export MICRODUCK_STILT_BLEND=0
export WANDB_MODE=disabled
uv run train Mjlab-Stilt-Flat-MicroDuck \\
  --env.scene.num-envs 64 \\
  --agent.max-iterations 5
```

- **PPO smoke test：** {os.environ['TASK3_TRAINING_STATUS']}
- **执行记录：** {os.environ['TASK3_TRAIN_DESC']}
- **训练日志：** [Task3_PPO_Smoke.log](../logs/Task3_PPO_Smoke.log)

这里训练的是**2 cm、宽支撑形态**，不是 2 m 策略。5 次迭代仅证明训练与优化器链路能否运行，不能说明重新学会高跷行走，也不能据此报告策略收敛。

## 6. 25 cm 历史步骤

- {os.environ['TASK3_NOTE25']}
- 视频：{os.environ['TASK3_LINK25']}

## 7. 局限性与后续研究

1. 2 m 复现使用公开 checkpoint，而非原始 PPO 的完整 optimizer / critic 中断状态。
2. 5 次 PPO 更新无法代替完整的课程学习训练。
3. 尚未在统一初始条件下统计 2 m 的跌倒率、速度跟踪误差或鲁棒性。
4. 2 m 高跷模型仅用于物理仿真；未经过真实机器人安全验证。

未来可比较 25 cm 与 2 m 的足端接触行为、身体倾角和运动轨迹，并设计逐级增加高度的课程，记录学习曲线。

## 8. GitHub 提交说明

把本文件以及 `media/`、`logs/` 文件夹上传至公开的 GitHub 仓库（例如 [UfanLee/Microduck](https://github.com/UfanLee/Microduck)）。不要把 `C:\\\\...`、`/home/...` 或 `127.0.0.1` 等本地路径当作助教可访问的提交链接。公开模型、源码与教程需保留原作者及许可证信息。

## References

- [Datawhale Every-Embodied — Task 3](https://github.com/datawhalechina/every-embodied)
- [Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground)
- [Hugging Face: MicroDuck Stilts](https://huggingface.co/HannesVonEssen/microduck-stilts)
- [Pollen Robotics MicroDuck](https://github.com/pollen-robotics/microduck)
"""
report.write_text(s, encoding="utf-8")
print(f"Report created: {report}")

summary=root/"logs"/"Task3_Result_Summary.txt"
summary.write_text(
    f"2m ONNX: {os.environ['TASK3_ONNX_STATUS']}\n"
    f"2m video: {os.environ['TASK3_VIDEO_STATUS']}\n"
    f"PPO smoke: {os.environ['TASK3_TRAINING_STATUS']}\n"
    f"Markdown report: {report}\n",
    encoding="utf-8"
)
print(summary.read_text(encoding="utf-8"))
PY

echo "=== ALL ATTEMPTS FINISHED ==="
echo "Windows folder: $DEST"
echo "Report: $DOCS/Task3_Stilt_Walking.md"
echo "Logs: $LOGS"
if [[ "$video_status" == "PASSED" && "$training_status" == "PASSED" && "$onnx_status" == "PASSED" ]]; then
  echo "SUCCESS: all three checks passed; review the MP4 before submitting."
  exit 0
fi
echo "INCOMPLETE: check log files. Completed outputs are preserved."
exit 1
