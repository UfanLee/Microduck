# Task 6 — MicroDuck Ladder Climbing: PPO Dual-Policy Reproduction

> Datawhale Every-Embodied · MicroDuck 运动策略与任务开发专题 · Task 6  
> 实验日期：2026-10-25  
> 实验性质：**已发布 checkpoint 的本地复现、继续训练 smoke test 与评估**；不是从零训练出攀爬策略。  
> 环境：WSL2 Ubuntu、NVIDIA RTX 3060 Laptop GPU（6 GB）；实际运行 CUDA/MuJoCo Warp。  
> 主要参考：[Datawhale Task 6](https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/05-MicroDuck%E8%BF%90%E5%8A%A8%E7%AD%96%E7%95%A5%E4%B8%8E%E4%BB%BB%E5%8A%A1%E5%BC%80%E5%8F%91%E4%B8%93%E9%A2%98/01-%E4%BB%BB%E5%8A%A1%E8%B5%84%E6%96%99/06-MicroDuck%E6%A2%AF%E9%9D%A2%E6%94%80%E7%88%AC%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%AF%BC%E8%AF%BB) · [Vottivott desk-climb](https://github.com/Vottivott/microduck-playground/tree/main/experiments/desk-climb) · [HannesVonEssen models](https://huggingface.co/HannesVonEssen/microduck-climb)

## 1. 目标与系统结构

本实验复现 MicroDuck 沿斜梯逐级攀爬、接触桌面、交接给起身策略的流程。系统不是单一端到端模型，而是两个 PPO actor 加一个仿真监督器：

1. **Climber**：61 维本体观测 → 14 维关节动作，攀爬梯子。
2. **Supervisor**：利用 MuJoCo 仿真中的真实桌面接触和根部位置，判断是否满足 `supported_root` 交接条件。
3. **Get-up**：61 维观测 → 14 维动作，在桌面上尝试恢复站立。

两个 actor 的 13 个 command slots 为零。Climber 不添加动作滤波；Get-up 使用分头部与腿部的滤波，并使用相对攀爬增益 0.8 的配置。监督器使用特权仿真信息，**不能直接声称已经具备实体机器人上自主检测交接的能力**。

本地日志确认 Climber actor 网络为 `61→512→256→128→14`，Critic 输入 76 维、输出 1 维。

## 2. 环境、模型与运行步骤

为防止覆盖前几项任务，本实验独立安装在：

```text
~/projects/microduck-desk-climb/experiments/desk-climb/source
```

使用官方仓库的锁定环境及模型下载脚本：

```bash
git clone https://github.com/Vottivott/microduck-playground.git ~/projects/microduck-desk-climb
cd ~/projects/microduck-desk-climb/experiments/desk-climb/source
uv sync --frozen
uv run ../training/download_models.py
```

官方继续训练 smoke test 的入口：

```bash
uv run ../training/run.py climber --envs 64 --iterations 5 --out /tmp/desk-climber-smoke
uv run ../training/run.py getup --envs 64 --iterations 5 --out /tmp/desk-getup-smoke
```

**说明：** smoke test 只用于验证读取 checkpoint、计算、梯度更新和保存链路，不是攀爬技能的收敛证明。本次对话没有收到上述 smoke test 的独立训练指标，故不报告其奖励或性能数值。

64 环境固定随机种子评估：

```bash
uv run ../training/run.py evaluate \
  --envs 64 --seconds 60 --seed 19923 \
  --out logs/desk-eval
```

### 本地评估 manifest（以实际输出为准）

| 配置 | 实际记录 |
|---|---|
| Parallel environments | 64 |
| Seed | 19923 |
| Duration | 60 s |
| Control dt | 0.02 s (50 Hz) |
| Actor dimension | 61 |
| Command slots zero | True |
| Ladder treads | 30 |
| Riser | 0.024 m |
| Ladder angle | 62° |
| Desktop height | 0.74 m |
| Source checkpoint resumed iteration | 56500 |
| Optimizer restoration | `full_critic_and_Adam_exact=True` |
| External wrench | 0 |

**版本注意：** 此次本地评估使用的几何参数（30 根横档、0.74 m 桌面）与上游 README 展示的保留场景（27 根横档、0.66 m 桌面）不同。不能把二者的评估结果直接当作同一几何配置下的对照。

## 3. 64 环境评估结果

数据来源：`logs/desk-eval/eval.json`、`switches.json`、`handoff-contract.json`、`compiled-table-check.json`、`manifest.json`；以下统计由本地终端输出提供。

| 指标 | 结果 | 解读 |
|---|---:|---|
| Trials | 64 | 并行评估环境数量 |
| Unique environments with handoff | 52/64 (**81.25%**) | 满足交接触发条件，不等于任务成功 |
| No handoff | 12/64 (18.75%) | 从未记录策略交接 |
| Survivors | 0/64 | 没有满足评估器完整时域存活条件的环境 |
| First failure: `fallen` | 61/64 (**95.31%**) | 最普遍的首次失败原因 |
| First failure: `overstep` | 3/64 (**4.69%**) | 其余首次失败原因 |
| First failure before handoff | 52/52 | **所有交接环境在交接前已记录首次失败** |
| First failure at/after handoff | 0 | 未观测到此类首次失败 |
| Mean first-failure time, later handed off | 10.86 s | 52 个环境 |
| Median first-failure time, later handed off | 10.80 s | 52 个环境 |
| Mean first-failure time, no handoff | 7.57 s | 12 个环境 |
| Median first-failure time, no handoff | 10.39 s | 12 个环境 |
| Mean handoff time | 14.10 s | 52 次交接事件 |
| Min / max handoff time | 10.42 / 39.48 s | 交接时间跨度较大 |
| Nonfinite | False | 未报告 NaN/Inf |
| Full-table collision check | True | 桌面碰撞体尺寸校验通过 |

### 攀爬进度分布

`highest_treads` 记录的值：

| 进度编号 | 环境数 |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 1 |
| 25 | 7 |
| 31 | 53 |

`31` 是**评估器进度编号**，不是“第 31 根物理横档”（此场景仅 30 根横档）；不可单凭此值判定稳定登桌成功。

### 交接条件审计

52 条交接事件来自 52 个不同环境，全部为：

```text
trigger_mode = supported_root
feet_on_desk = [False, True]
```

即交接时检测到**右脚接触桌面**，并符合监督器规定的根部位置门槛。交接条件使用仿真器可见的接触与位置，不是 actor 自身推理出的视觉/接触信息。

### 核心结论

**81.25% 是 handoff trigger rate，而不是稳定攀爬成功率。** 在 52 个触发交接的环境中，首次失败均发生在交接之前；因为评估器在失败标志触发后继续推进仿真，机器人仍可能随后接触桌面并触发 Get-up。当前数据不支持宣称“81.25% 完整成功登桌”，也不能把全部失败归咎于 Get-up。

`survivors=0` 反映本次评估器的完整时域存活指标。若要定义“连续稳定站立 10 秒”的成功率，还需要明确稳定阈值并从姿态、桌面接触、速度及轨迹中独立计算；现有终端摘要未提供该数值。

## 4. 运行中遇到的问题及解决

### 4.1 ONNXRuntime GPU 设备发现警告

终端显示：

```text
GPU device discovery failed
Failed to open file: /sys/class/drm/card0/device/vendor
```

本机为 WSL2。该警告不等于 CUDA 整体不可用：同一轮输出中 `mujoco_warp ... load on device 'cuda:0'` 成功，Actor/Critic 初始化且评估产出了完整文件。处理方式是优先检查是否出现 Python traceback、CUDA 失败，以及结果文件是否成功生成，而非看到 ONNXRuntime 警告就重装驱动。

### 4.2 `survivors=0` 与交接事件看似矛盾

首次统计发现 `survivors=0`，但 `switches.json` 有 52 条记录。进一步去重 `env` 后确认是 52 个不同环境；再对比 `first_end.seconds` 与 `switches.seconds`，证实 **52/52 的首次失败早于交接**。原评估设计允许失败标志出现后继续物理仿真，以记录后续恢复/交接行为；因此这两类指标并不矛盾。

### 4.3 终端 Python heredoc 过长、未执行

在手动复制较长的 `python - <<'PY' ... PY` 代码时，出现 `PY~`（而不是独立一行的 `PY`）使 shell 等待输入。另外该 Ubuntu 环境没有全局 `python` 命令，输出 `Command 'python' not found`。处理方式：改用 `.venv/bin/python`、保证独立 `PY` 行闭合，并将长修改脚本保存成独立 `.py` 文件，减少手动复制出错。

### 4.4 官方评估程序没有直接提供 MP4 导出参数

`training/evaluate_sequence.py` 默认保存 JSON、动作与轨迹 NPZ，不接受 `--video`。随后采用**独立录制脚本**重新运行 1 个环境、保留原有 `evaluate_sequence.py` 和 64 环境结果。用户已确认本地视频成功生成。该视频属于**新的一次单环境仿真**，不是把之前的 64 环境某条轨迹逐帧渲染出来。

> 视频文件名及具体画面仍需以最终纳入仓库的本地 MP4 为准；在未独立检查视频之前，不在报告中声称其实现了稳定登桌或无重置连续站立。

## 5. 视频记录

本地已生成 Task 6 单环境录制视频；建议仓库路径：

`media/Task6_Ladder_Climbing.mp4`

[打开 Task 6 本地回放](../media/Task6_Ladder_Climbing.mp4)

上传 GitHub 前确认实际 MP4 放在上述位置，且其对应 JSON 来自 `logs/desk-video/`。尽量保留整个失败、接触、策略交接过程，不剪成单独“成功”片段。

## 6. 实验局限与后续工作

1. **策略基线性质**：使用官方发布的 PPO 模型，当前实验主要完成复现与评估，没有重新从零学习出攀爬技能。
2. **交接与成功不是同一个指标**：切换需要模拟器特权信息，而完整成功需要没有过早终止、真实落脚支撑、桌面起身和持续稳定。
3. **固定 seed 的有限评估**：64 个环境、单个评估配置，不能据此推断跨地形、跨初始姿态和实体机器人的泛化表现。
4. **改进顺序**：优先检查首次跌倒的真实接触位置、脚端支撑和交接前几秒的姿态；其次重新设计跨 seed 的稳定登桌统计，再考虑针对性 PPO 续训。
5. **sim-to-real**：尚未在实体 MicroDuck 上验证监督器的触发机制、驱动延迟及安全保护。

## 7. 可复现性与证据

64 环境评估输出：

```text
logs/desk-eval/
├── eval.json
├── geometry-audit.json
├── actions.npz
├── switches.json
├── handoff-contract.json
├── manifest.json
├── action-intervention.json
├── compiled-table-check.json
└── trajectories.npz
```

建议 GitHub 只纳入脱敏/精简后的统计摘要与 MP4，避免无必要地上传体积较大的原始 NPZ；如需公开完整轨迹可另行发布带哈希的 release asset。请额外记录本地 `git rev-parse HEAD` 以固定源码版本。

## 8. 总结

本地完成了 MicroDuck 双策略梯面攀爬系统的复现、CUDA/MuJoCo Warp 批量评估和单环境视频生成。64 环境中有 **52/64（81.25%）触发策略交接**，但 **0/64 达到完整评估时域存活条件**；首次失败以 `fallen` 为主（61/64），且所有最终触发交接的环境均在交接前已有失败标志。本实验的关键认识是：**接触触发、攀爬进度、失败时序和最终持续稳定性必须分开评价**，不能只凭演示视频或策略交接率宣称任务成功。
