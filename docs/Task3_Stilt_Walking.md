# MicroDuck Task 3 — Stilt Walking Reinforcement Learning Reproduction

**Datawhale Every-Embodied · 高跷行走强化学习复现**  
**Recorded:** 2026-10-19 00:32:43 +1000  
**Code version:** `c5fcc50219fef01ac9931d0079c583ccbb29b689`  
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
uv run python scripts/export.py Mjlab-Stilt-Flat-MicroDuck \
  --checkpoint-file policies/stilts/200cm/checkpoint.pt \
  --device cpu --num-envs 1 \
  --video True --video-length 500 \
  --video-width 1280 --video-height 720 \
  --viewer auto --running-speed 0.15 \
  --onnx-file artifacts/stilts_200cm.onnx
```

- **ONNX 接口检查：** PASSED
- **2 m 视频生成：** PASSED
- **执行记录：** 2m checkpoint rollout generated, new nonempty MP4 copied to Windows Desktop
- **视频证据：** [2 m 本地回放视频](../media/MicroDuck_Stilts_200cm.mp4)
- **复现日志：** [Task3_200cm_Export.log](../logs/Task3_200cm_Export.log)

**解释范围：** 视频生成只能证明公开 actor 能在本地物理环境执行并产生帧；是否有稳定步态、跌倒、明显异常，仍应打开视频进行视觉检查。该视频不代表从零训练出了 2 m 策略，也不代表真机可安全使用。

## 5. PPO 训练冒烟测试（新执行的优化过程）

设置：

```bash
export MICRODUCK_STILT_HEIGHT_CM=2
export MICRODUCK_STILT_BLEND=0
export WANDB_MODE=disabled
uv run train Mjlab-Stilt-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5
```

- **PPO smoke test：** PASSED
- **执行记录：** 64 parallel environments, 5 PPO iterations; process exited successfully
- **训练日志：** [Task3_PPO_Smoke.log](../logs/Task3_PPO_Smoke.log)

这里训练的是**2 cm、宽支撑形态**，不是 2 m 策略。5 次迭代仅证明训练与优化器链路能否运行，不能说明重新学会高跷行走，也不能据此报告策略收敛。

## 6. 25 cm 历史步骤

- Found and copied an existing 25cm MP4 (creation conditions should be confirmed separately).
- 视频：[25 cm 视频](../media/MicroDuck_Stilts_25cm.mp4)

## 7. 局限性与后续研究

1. 2 m 复现使用公开 checkpoint，而非原始 PPO 的完整 optimizer / critic 中断状态。
2. 5 次 PPO 更新无法代替完整的课程学习训练。
3. 尚未在统一初始条件下统计 2 m 的跌倒率、速度跟踪误差或鲁棒性。
4. 2 m 高跷模型仅用于物理仿真；未经过真实机器人安全验证。

未来可比较 25 cm 与 2 m 的足端接触行为、身体倾角和运动轨迹，并设计逐级增加高度的课程，记录学习曲线。

## 8. GitHub 提交说明

把本文件以及 `media/`、`logs/` 文件夹上传至公开的 GitHub 仓库（例如 [UfanLee/Microduck](https://github.com/UfanLee/Microduck)）。不要把 `C:\\...`、`/home/...` 或 `127.0.0.1` 等本地路径当作助教可访问的提交链接。公开模型、源码与教程需保留原作者及许可证信息。

## References

- [Datawhale Every-Embodied — Task 3](https://github.com/datawhalechina/every-embodied)
- [Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground)
- [Hugging Face: MicroDuck Stilts](https://huggingface.co/HannesVonEssen/microduck-stilts)
- [Pollen Robotics MicroDuck](https://github.com/pollen-robotics/microduck)
