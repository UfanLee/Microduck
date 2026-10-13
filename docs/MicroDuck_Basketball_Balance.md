# MicroDuck Basketball Balance
## 基于 PPO 与 MuJoCo-Warp 的机器人篮球平衡强化学习实验

**项目类型：** Embodied AI / Reinforcement Learning / Robotics Simulation  
**学习项目：** Datawhale Every-Embodied — Task 1  
**实验日期：** 2026 年 10 月 13 日  
**实验平台：** Windows 11 + WSL2 Ubuntu + NVIDIA RTX 3060 Laptop GPU

---

## 1. 项目摘要（Abstract）

本实验基于开源项目 `microduck-playground`，复现 MicroDuck 双足机器人在篮球上保持平衡的强化学习任务，重点研究 PPO（Proximal Policy Optimization）、Actor-Critic 架构、LSTM 策略网络以及 MuJoCo-Warp GPU 物理仿真。

实验没有从零开始训练策略，而是下载公开预训练模型，在本地 NVIDIA RTX 3060 Laptop GPU（6GB 显存）上进行小规模 PPO Fine-tuning，验证强化学习训练流程。

实验成功完成以下工作：

- 配置 WSL2 Ubuntu、Python、CUDA、PyTorch 和 MuJoCo-Warp 环境。
- 下载并加载公开篮球平衡预训练 checkpoint。
- 使用 16 个并行环境进行 2 次 PPO 继续训练更新。
- 生成新的 `model_7000.pt` checkpoint。
- 对原始模型与 Fine-tuned 模型进行初步对照评测。
- 修复 PyTorch InferenceMode 与 LSTM 隐状态重置的兼容性问题。
- 导出 ONNX 策略模型，并生成本地 MuJoCo 仿真 MP4 视频。

在 16 个并行环境、10 秒评测条件下，原始模型和继续训练模型均达到 100% 存活率。原始模型的平均前向速度跟踪误差为 0.0953 m/s，而继续训练模型为 0.1170 m/s。

结果表明，本地硬件能够运行该具身智能强化学习流程，但仅凭两次 PPO 更新与小规模评测，不能证明 Fine-tuning 对模型性能产生了稳定改善。

## 2. 研究背景与问题定义

### 2.1 具身智能与强化学习

具身智能（Embodied AI）研究智能体如何通过感知、决策与动作，与物理环境进行交互。

与传统的图像分类或文本预测不同，机器人控制需要持续处理动态状态、动作约束和环境反馈。

本实验研究的核心问题是：

**如何通过强化学习，使 MicroDuck 机器人在不稳定的篮球表面保持平衡，并根据速度指令调整自身运动？**

该问题涉及动态平衡、关节控制、物理仿真和序列决策。

### 2.2 强化学习问题建模

将任务建模为马尔可夫决策过程（MDP）：

\[
\mathcal{M}=(S,A,P,R,\gamma)
\]

其中：

- \(S\)：机器人和环境的状态空间。
- \(A\)：机器人执行的动作空间。
- \(P\)：状态转移规律，由物理仿真决定。
- \(R\)：奖励函数，用于衡量动作质量。
- \(\gamma\)：未来奖励折扣因子。

每个控制时间步中，机器人接收观测 \(o_t\)，由策略网络产生动作 \(a_t\)，然后物理环境更新状态并返回奖励。

训练目标是最大化期望累计奖励：

\[
J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{T}\gamma^t r_t\right]
\]

## 3. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 / WSL2 Ubuntu 24.04 |
| GPU | NVIDIA RTX 3060 Laptop GPU |
| 显存 | 6 GiB |
| Python | 3.12 |
| 包管理 | uv |
| 深度学习框架 | PyTorch |
| 物理仿真 | MuJoCo-Warp |
| Warp | 1.12.0 |
| CUDA Toolkit（Warp 日志） | 12.9 |
| GPU 驱动支持的 CUDA 版本 | 13.1 |
| 强化学习算法 | PPO |
| 任务 | Mjlab-Basketball-MicroDuck |

### 3.1 开源资源

训练项目：

https://github.com/Vottivott/microduck-playground

Datawhale 学习项目：

https://github.com/datawhalechina/every-embodied

预训练权重：

https://huggingface.co/HannesVonEssen/microduck-basketball

本实验建立在以上开源资源之上。训练环境、机器人模型、奖励配置和预训练策略并非本人从零开发。

## 4. 强化学习算法与策略网络

### 4.1 PPO：Proximal Policy Optimization

PPO 是一种基于策略梯度的强化学习算法。

相比无限制地更新策略参数，PPO 通过裁剪目标函数限制每次策略变化的幅度，以提高训练稳定性。

其典型裁剪目标为：

\[
L^{CLIP}(\theta)
=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
\]

其中：

\[
r_t(\theta)=
\frac{\pi_\theta(a_t|o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|o_t)}
\]

- \(\pi_\theta\)：当前策略。
- \(\pi_{\theta_{\mathrm{old}}}\)：更新前的策略。
- \(\hat A_t\)：优势估计。
- \(\epsilon\)：裁剪范围参数。

PPO 的核心思想并不是每次都尽可能大幅度地提高奖励，而是在相对受限的策略变化范围内进行优化。

### 4.2 Actor-Critic 架构

本实验使用 Actor-Critic 训练思想。

**Actor（策略网络）** 根据机器人观测生成关节控制动作。

\[
a_t\sim \pi_\theta(a_t|o_t)
\]

**Critic（价值网络）** 估计当前状态的未来累计奖励：

\[
V_\phi(s_t)
\]

训练阶段，Critic 可利用比 Actor 更丰富的状态信息，帮助估计优势函数；实际部署时只需要策略网络及其必要的观测处理与记忆状态。

这种训练设计有助于在仿真环境中利用额外状态信息，同时尽量维持机器人实际运行时的可用性。

### 4.3 61 维观测空间

MicroDuck 篮球平衡策略的 Actor 接收 61 维观测。

| 观测组成 | 维度 |
|---|---:|
| 机器人角速度 | 3 |
| 投影重力方向 | 3 |
| 关节位置 | 14 |
| 关节速度 | 14 |
| 上一步动作 | 14 |
| 运动指令 | 3 |
| 头部状态预留维度 | 4 |
| 身体状态预留维度 | 6 |
| **总计** | **61** |

\[
o_t\in\mathbb R^{61}
\]

本任务的公开策略配置中，Actor 不直接接收篮球完整物理状态，而是主要根据机器人本体感知和运动反馈执行控制。

这与实际机器人中依赖 IMU、关节编码器等传感器进行闭环控制的思想一致。

### 4.4 14 维动作空间

策略输出 14 维动作，对应 14 个受控关节的动作命令。

\[
a_t\in\mathbb R^{14}
\]

动作经过仿真环境中的处理与约束后，转换为实际关节执行器控制信号。

动作空间设计直接影响机器人能执行哪些姿态调整，以及控制策略的可学习性。

### 4.5 LSTM 时序记忆

本任务的策略网络使用 LSTM 结构，其循环状态包含 Hidden State 与 Cell State。

\[
(h_t,c_t)=\operatorname{LSTM}(o_t,h_{t-1},c_{t-1})
\]

机器人平衡控制具有明显的时序特性。仅观察单一时刻，并不一定足以准确推断机器人的运动趋势。

LSTM 能够利用历史观测形成内部状态，辅助策略进行连续决策。

这也意味着在策略推理中必须正确管理循环隐状态，并在环境结束或重置时执行相应的状态重置。

## 5. 奖励函数分析

篮球平衡任务不仅要求机器人不摔倒，还需要兼顾运动指令、身体姿态和控制动作的稳定性。

本次训练日志包含以下奖励项：

| 奖励项 | 作用 |
|---|---|
| `track_linear_velocity` | 鼓励跟踪线速度指令 |
| `track_angular_velocity` | 鼓励跟踪角速度指令 |
| `pose` | 鼓励合理的身体和关节姿态 |
| `body_ang_vel` | 惩罚不必要的身体旋转 |
| `angular_momentum` | 约束角动量相关行为 |
| `dof_pos_limits` | 惩罚接近或超出关节位置限制 |
| `action_rate_l2` | 惩罚动作变化过快 |
| `self_collisions` | 约束自碰撞 |
| `upright` | 鼓励机器人保持直立 |
| `centered` | 鼓励机器人保持在球体中心附近 |
| `feet_on_ball` | 鼓励机器人双脚与篮球保持接触 |
| `height` | 鼓励适当的身体高度 |
| `ball_speed` | 约束篮球的运动速度 |

整体奖励可以抽象表示为：

\[
R_t=\sum_i w_i r_i(t)
\]

其中 \(r_i\) 为各项奖励或惩罚，\(w_i\) 为其对应权重。

不同奖励之间可能存在竞争关系。例如，快速调整姿态有利于防止摔倒，但动作变化过大又可能产生惩罚。

因此，机器人平衡并不是单一指标最大化问题，而是一个多目标控制优化问题。

## 6. 实验设计与过程

### 6.1 下载公开预训练模型

使用 Hugging Face 下载原始 checkpoint：

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('HannesVonEssen/microduck-basketball', local_dir='artifacts/basketball')"
```

获得主要文件：

- `checkpoint.pt`：公开预训练权重。
- `policy.onnx`：公开 ONNX 策略。
- `replay.mp4`：项目提供的参考演示视频。

### 6.2 PPO 继续训练

为了避免 6GB 显存下运行过大规模的训练任务，本实验采用小规模 Smoke Test。

```bash
WANDB_MODE=disabled uv run python scripts/finetune_basketball.py \
  artifacts/basketball/checkpoint.pt \
  --run-name basketball-resume-smoke \
  --num-envs 16 \
  --iterations 2 \
  --learning-rate 2e-5 \
  --push-interval-s 1.5 3 \
  --save-interval 2
```

训练顺利完成，终端输出：

```text
FINETUNE_DONE
```

并保存新的 checkpoint：

```text
model_7000.pt
```

该模型是在公开权重基础上继续更新的结果，并非从随机初始化训练所得。

### 6.3 训练指标

记录到的最后一次训练迭代指标如下：

| 指标 | 结果 |
|---|---:|
| Total steps（日志所示） | 768 |
| Steps per second | 269 |
| Mean reward | 2.15 |
| Mean value loss | 0.0299 |
| Mean surrogate loss | -0.0582 |
| Mean entropy loss | 2.6221 |
| Mean action std | 0.31 |
| Mean episode length | 23.50 |
| Fell termination | 0.0000 |
| NaN state termination | 0.0000 |

其中 `fell = 0` 与 `nan_state = 0` 表明该统计窗口没有记录到这些终止类型，但不能推导出策略在所有环境条件下都不会失败。

## 7. 模型对照评测

### 7.1 评测设置

本实验分别评测：

**Baseline：** 公开 `checkpoint.pt`

**Fine-tuned：** 本次生成的 `model_7000.pt`

评测脚本为：

```text
scripts/eval_basketball_checkpoint.py
```

评测条件：

- 并行环境数：16。
- 测试时长：10 秒。
- 指标：存活率、首次摔倒时间、平均前向速度跟踪误差。

### 7.2 实验结果

| 指标 | Baseline | Fine-tuned |
|---|---:|---:|
| 环境数量 | 16 | 16 |
| 评测时长 | 10 s | 10 s |
| 存活率 | 100% | 100% |
| 首次摔倒时间 | 无摔倒记录 | 无摔倒记录 |
| 前向速度跟踪误差 | 0.0953 m/s | 0.1170 m/s |

速度跟踪误差的相对变化：

\[
\frac{0.1170032-0.0953082}{0.0953082}\times100\%
\approx22.8\%
\]

继续训练模型在当前测试中的速度跟踪误差约增加 22.8%。

### 7.3 结果讨论

两个模型在本次测试中都保持了 100% 的存活率，说明在测试范围内没有观察到平衡失败。

但是，Fine-tuned 模型的前向速度跟踪误差高于 Baseline。

可能的原因包括：

1. PPO 继续训练更新次数不足，尚未形成稳定的策略改善。
2. 多奖励目标之间存在权衡，优化累计奖励不一定能降低单项速度误差。
3. 两次评测可能受到不同随机初始状态或运动指令的影响。
4. 16 个环境、10 秒的测试规模不足以形成高置信度结论。

因此，本实验不能证明继续训练提高了模型性能，也不能仅凭当前结果认定继续训练必然导致策略退化。

后续需要固定随机种子、统一测试条件、增加环境数量和评测时间，并进行多次重复实验。

## 8. ONNX 导出、视频录制与故障调试

### 8.1 ONNX 导出

本实验成功将 Fine-tuned checkpoint 导出为：

```text
artifacts/basketball/finetuned_7000.onnx
```

文件大小约为 2.4 MB。

ONNX 是用于表示和交换神经网络计算图的开放格式，为后续跨运行环境推理及部署提供了基础。

本次实验成功完成 ONNX 导出，但尚未在真实机器人硬件上进行部署与验证。

### 8.2 视频录制

使用 `scripts/export.py` 加载 checkpoint，在 MuJoCo 仿真环境中运行策略并录制 MP4。

成功生成的视频文件为：

```text
logs/rsl_rl/basketball/2026-10-13_12-52-59_basketball-resume-smoke/videos/play/rl-video-step-0.mp4
```

已验证视频文件存在。正式 10 秒版本的实际时长及画面效果仍需进一步核实。

### 8.3 关键调试问题：LSTM 与 InferenceMode

最初在视频录制过程中遇到：

```text
RuntimeError:
Inplace update to inference tensor outside InferenceMode is not allowed.
```

问题发生在策略的循环隐状态重置阶段。

原始实现的相关逻辑为：

```python
with torch.inference_mode():
    actions = policy(obs)

obs, _, dones, _ = env.step(actions)

if hasattr(policy, "reset"):
    policy.reset(dones)
```

LSTM 在 `InferenceMode` 内产生隐状态，但后续重置操作在该模式之外执行，导致 PyTorch 拒绝进行原地修改。

修复方式是在状态重置操作外增加 `torch.inference_mode()`：

```python
with torch.inference_mode():
    actions = policy(obs)

obs, _, dones, _ = env.step(actions)

with torch.inference_mode():
    if hasattr(policy, "reset"):
        policy.reset(dones)
```

修复后成功运行视频录制和 ONNX 导出。

这一问题说明，部署循环策略网络时，除了神经网络本身的推理计算，还必须正确处理内部记忆状态的生命周期。

## 9. 实验局限

本实验主要用于学习与验证完整的强化学习工程流程，存在以下局限：

- 未从零训练机器人平衡策略。
- Fine-tuning 仅进行 2 次 PPO 更新。
- 评测环境数量和时长较小。
- Baseline 与 Fine-tuned 的随机条件尚未严格配对。
- 未进行多种随机种子的统计评估。
- 尚未验证导出 ONNX 的数值一致性。
- 尚未在真实机器人平台部署。
- 正式 10 秒演示视频仍需确认实际视频时长和视觉表现。

这些局限应在项目展示时明确说明。

## 10. 后续研究计划

**阶段一：评测规范化**

固定随机种子，增加环境数和仿真时长，测量模型在更长时间和不同指令下的平衡稳定性。

**阶段二：PPO 超参数实验**

比较不同学习率、训练迭代数和奖励权重对策略表现的影响。

**阶段三：模型结构与感知研究**

进一步研究 LSTM 时序建模、机器人本体感知、隐状态管理以及模拟与现实之间的差异。

**阶段四：具身智能平台扩展**

将本实验中的 PPO、物理仿真、模型部署和闭环控制经验，逐步迁移至轮式机器人、无人机及其他自主机器人平台。

## 11. 实验总结

本实验在消费级 NVIDIA RTX 3060 Laptop GPU 上，完成了 MicroDuck 篮球平衡强化学习的本地仿真、预训练权重加载、PPO 继续训练、模型保存、对照评测、ONNX 导出和视频录制。

实验的主要成果不是证明新模型优于公开模型，而是成功建立并验证了一条可复现的具身智能强化学习工程流程：

**环境配置 → 预训练模型加载 → PPO Fine-tuning → 模型评测 → 调试 → ONNX 导出 → 仿真视频录制。**

同时，实验说明了强化学习研究中严格评测的重要性：模型能够成功完成训练，并不代表策略性能一定提高。性能判断必须建立在公平的实验设置和足够的统计证据之上。

---

## References

1. Vottivott, *microduck-playground*. https://github.com/Vottivott/microduck-playground
2. Datawhale, *every-embodied*. https://github.com/datawhalechina/every-embodied
3. HannesVonEssen, *microduck-basketball*. https://huggingface.co/HannesVonEssen/microduck-basketball
4. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. https://arxiv.org/abs/1707.06347
5. Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997.