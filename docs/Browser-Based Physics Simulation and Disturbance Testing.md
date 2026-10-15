# MicroDuck Task 2 — Browser-Based Physics Simulation and Disturbance Testing

**基于 mjswan、MuJoCo WASM 与 ONNX Runtime Web 的浏览器机器人仿真实验**

**Project:** Datawhale Every-Embodied  
**Task:** 02 — mjswan MicroDuck 浏览器物理扰动  
**Date:** 2026-10-15  
**Environment:** Windows 11 + WSL2 Ubuntu 24.04  
**Status:** Core Experiment Completed

---

## 1. Abstract（实验摘要）

本实验基于 Datawhale Every-Embodied 的 MicroDuck 运动策略与任务开发专题，使用开源项目 `mjswan_playground` 构建 MicroDuck 浏览器端机器人仿真应用。

实验主要研究如何将机器人模型与训练好的 ONNX 策略部署到浏览器，并利用 MuJoCo WebAssembly 进行物理仿真，实现运动控制与外力扰动交互。

本次实验成功完成：

- 在 WSL2 Ubuntu 环境安装并配置 mjswan Playground。
- 构建 MicroDuck 浏览器静态应用。
- 在 Windows 浏览器中加载 MicroDuck 机器人模型和 Walk 策略。
- 使用 Forward 速度指令控制机器人前进。
- 通过鼠标拖拽机器人施加物理扰动，并观察机器人姿态变化。
- 发现低速启动与持续运动之间存在值得进一步研究的行为差异。
- 排查并解决 Node.js 环境路径问题。

实验成功验证了 MicroDuck 在浏览器中的策略回放与物理交互流程。

本次任务不涉及重新训练神经网络，也不代表已经实现真实机器人硬件部署。

## 2. Background and Objectives（背景与目标）

### 2.1 从强化学习训练到浏览器部署

在 Task 1 中，实验使用 MuJoCo-Warp 和 PPO 对 MicroDuck 篮球平衡策略进行了小规模继续训练，并成功导出 ONNX 模型。

Task 2 的目标有所不同。

本任务主要关注已训练策略如何进入浏览器运行环境，并通过可视化界面与物理世界交互。

两个任务的技术重点如下：

| Task 1 | Task 2 |
|---|---|
| PPO 强化学习 | 浏览器策略部署 |
| MuJoCo-Warp | MuJoCo WASM |
| PyTorch | ONNX Runtime Web |
| 训练与评测 | 策略回放与物理交互 |
| GPU 训练 | 浏览器端仿真 |

Task 2 使用 mjswan Playground 提供的 MicroDuck 场景与策略，并非直接将 Task 1 中的篮球平衡模型加载到 Duck 行走场景。

### 2.2 实验目标

本实验主要验证三个方面：

1. **Simulation Deployment：** 能否将 MicroDuck 模型、策略和物理场景构建为浏览器应用。
2. **Policy Execution：** 能否通过浏览器输入速度指令，使机器人执行相应动作。
3. **Physics Disturbance：** 能否利用鼠标施加外力，观察机器人与环境之间的物理交互。

## 3. System Architecture（系统架构）

根据官方教程，mjswan 浏览器仿真系统主要由以下模块构成：

| 技术组件 | 作用 |
|---|---|
| mjswan Playground | 示例工程及应用构建入口 |
| MuJoCo WASM | 浏览器内物理动力学计算 |
| ONNX Runtime Web | 浏览器端神经网络策略推理 |
| Three.js / WebGL | 机器人和场景的三维可视化 |
| JavaScript | 用户输入、控制面板及运行调度 |
| MuJoCo MJCF | 机器人模型、关节、执行器和物理属性 |

### 3.1 核心执行流程

```text
Browser User Interface
          |
          v
    Velocity Command
          |
          v
  Robot Observation
          |
          v
   ONNX Policy Model
          |
          v
   Joint Actions
          |
          v
    MuJoCo WASM
          |
          v
 Robot State Update
          |
          v
 Three.js Rendering
```

浏览器通过用户界面接收运动命令，将当前仿真状态转换为策略所需的观测向量，再通过 ONNX 模型推理得到关节动作。

MuJoCo 根据动作、重力、碰撞、摩擦以及外部作用力计算下一时刻的机器人状态。

渲染模块随后更新三维画面。

这是一套动态仿真过程，不是预先录制的视频回放。

## 4. Observation and Action Interface（观测与动作接口）

### 4.1 Observation Space

根据官方教程，MicroDuck 浏览器示例中的策略通常采用 61 维观测接口：

| Observation | Dimension |
|---|---:|
| Body angular velocity | 3 |
| Projected gravity | 3 |
| Joint positions | 14 |
| Joint velocities | 14 |
| Previous actions | 14 |
| Velocity commands | 3 |
| Additional command / padding | 10 |
| **Total** | **61** |

其形式为：

\[
o_t\in\mathbb{R}^{61}
\]

观测向量包含机器人身体运动、关节状态、历史动作及当前控制目标等信息。

需要注意，不同 MicroDuck 策略的具体观测内容和预处理方式可能存在差异，应以对应模型和源码中的接口定义为准。

### 4.2 Action Space

策略输出通常为 14 维关节动作：

\[
a_t\in\mathbb{R}^{14}
\]

动作向量与 MicroDuck 的关节顺序、动作缩放及执行器配置相关。

策略输出并不直接代表机器人的世界坐标位置，也不必然等同于电机的实际扭矩。

这意味着不能仅根据 ONNX 输入输出维度一致，就断定不同机器人场景和不同策略能够直接兼容。

## 5. Experimental Environment（实验环境）

| Component | Configuration |
|---|---|
| Operating System | Windows 11 |
| Linux Environment | WSL2 Ubuntu 24.04 |
| Python | 3.12 |
| Package Manager | uv |
| Application | mjswan Playground |
| Browser | Windows 端现代浏览器 |
| Simulation | MuJoCo WebAssembly |
| Policy Inference | ONNX Runtime Web |

本任务的核心仿真运行在浏览器中，与 Task 1 使用 GPU 执行并行强化学习训练的方式不同。

## 6. Implementation（实施过程）

### 6.1 获取源码

在 Ubuntu 环境中执行：

```bash
cd ~/projects

git clone https://github.com/ttktjmt/mjswan_playground.git

cd mjswan_playground
```

### 6.2 安装依赖

```bash
uv sync --all-extras
```

然后通过以下命令确认示例项目能够被识别：

```bash
uv run msp list
```

### 6.3 构建 MicroDuck 浏览器应用

执行：

```bash
uv run msp build microduck --output-dir dist/microduck
```

首次构建过程中，项目自动获取相关 MicroDuck 和 MicroDuck RL 源码，并准备前端 Node.js 环境。

本次实验在 npm 依赖安装阶段遇到了环境路径问题。

具体调试过程见第 8 节。

修复完成后，终端输出：

```text
Microduck 4/4 Rollers

✓ Saved mjswan application to:
/home/lxh20/projects/mjswan_playground/dist/microduck
```

证明静态应用构建成功。

### 6.4 启动浏览器应用

执行：

```bash
uv run msp run microduck --port 8092
```

然后在 Windows 浏览器访问：

```text
http://127.0.0.1:8092/
```

成功加载后，浏览器显示 MicroDuck 三维模型以及 Scene、Policy、Commands、Camera 和 Reset 等控制功能。

## 7. Experimental Results（实验结果）

### 7.1 Experiment A — Browser Simulation Loading

**Objective**

验证 MicroDuck 浏览器端场景与策略能否正常加载。

**Procedure**

打开本地浏览器应用，选择：

```text
Scene: Duck
Policy: Walk
```

**Observed Result**

浏览器成功显示 MicroDuck 三维机器人模型、地面网格和策略控制面板。

界面包含 Forward、Sideways、Turn 等控制参数。

**Conclusion**

本地构建的 MicroDuck 浏览器应用能够成功加载并显示基础场景与策略控制界面。

### 7.2 Experiment B — Forward Velocity Control

**Objective**

验证 Forward 指令能够控制机器人前进。

**Procedure**

在 Walk 策略下，将 Forward 速度指令从 0 调整为正值。

观察机器人从静止到移动的行为。

**Observed Result**

实验中发现：

- 直接将 Forward 设置为 0.2 m/s 时，机器人没有立即表现出明显前进。
- 将 Forward 提高到 0.3 m/s 以上后，机器人开始前进。
- 随后将 Forward 降低回 0.2 m/s，机器人仍可以继续移动。

这表明机器人从静止启动和已经处于运动状态时，对相同速度指令的响应可能不同。

**Discussion**

这一现象可能与多个因素有关：

1. 策略在低速指令下倾向于保持站立。
2. 机器人需要形成足够明显的步态动作才能开始移动。
3. 机器人接触状态和初始运动状态可能影响后续响应。
4. 策略内部的历史信息或步态状态可能影响动作生成。

从现象上看，这与具有状态依赖的启动行为相似，也可能呈现类似滞回的特征。

但当前实验只进行了初步观察，没有使用固定条件重复测试，也没有检查策略代码中是否存在明确阈值。

因此，**不能据此认定系统存在严格的 0.3 m/s 启动阈值，或已经证明存在滞回机制**。

后续可通过多次 Reset、固定不同速度指令和记录机器人位移来进一步验证。

### 7.3 Experiment C — External Physics Disturbance

**Objective**

验证鼠标拖拽是否能够对机器人施加物理扰动。

**Procedure**

1. 将 Forward 设置为 0。
2. 使用鼠标左键按住机器人身体。
3. 拖动鼠标，向机器人施加外部扰动。
4. 观察机器人姿态及物理响应。

**Observed Result**

实验成功触发了鼠标物理拖拽交互，观察到了机器人在外力作用下的姿态响应。

**Conclusion**

本次实验验证了浏览器中可以对 MicroDuck 施加交互式物理扰动。

需要注意：仅凭拖拽功能成功，不代表策略能够在所有外力条件下恢复平衡，也不代表真实机器人能够安全承受同等冲击。

### 7.4 Results Summary

| Test | Result |
|---|---|
| Static application build | Passed |
| Browser scene loading | Passed |
| MicroDuck 3D rendering | Passed |
| Walk policy selection | Passed |
| Forward velocity control | Passed |
| Physical drag interaction | Passed |
| Quantitative disturbance recovery test | Not conducted |
| Real hardware deployment | Not conducted |

## 8. Debugging Record（故障排查记录）

### 8.1 Problem — Node.js executable not found

初次构建时出现：

```text
Installing npm dependencies (npm install)...

/usr/bin/env: 'node': No such file or directory
```

随后程序抛出：

```text
RuntimeError: Build failed with exit code 127
```

### 8.2 Root Cause

构建流程已经创建了独立的 Node.js 运行环境，但是 npm 启动时未能在当前环境变量 `PATH` 中找到对应的 `node` 可执行文件。

因此，Python 部分的项目安装虽然已经完成，前端构建阶段仍然无法继续。

### 8.3 Solution

在 `mjswan_playground` 根目录中执行：

```bash
export PATH="$PWD/.venv/lib/python3.12/site-packages/mjswan/template/.nodeenv/bin:$PATH"
```

然后检查 Node.js 和 npm：

```bash
which node
node --version
npm --version
```

重新运行：

```bash
uv run msp build microduck --output-dir dist/microduck
```

最终成功生成浏览器应用。

这一问题说明：即使 Python 依赖安装成功，跨语言软件工程仍可能受到 Node.js、npm 和环境变量配置影响。

## 9. Physics Disturbance Mechanism（物理扰动原理）

### 9.1 鼠标拖拽并不是直接移动机器人

按照 mjswan 官方教程的实现思路，浏览器拖拽交互通过射线选中机器人动态刚体，将鼠标位移转换成施加在刚体上的力或力矩。

这些外力会进入 MuJoCo 的动力学计算。

MuJoCo 提供用于表示刚体外力的数组：

```text
mjData.xfrc_applied
```

它可以表示施加在刚体上的空间力，包括力与力矩分量。

拖拽过程中，外力改变机器人的运动状态，而物理引擎负责计算由此产生的加速度、碰撞、接触和姿态变化。

### 9.2 外力与策略动作的区别

机器人运动受到至少两类控制输入的影响：

**Policy Action**

由 ONNX 策略根据机器人观测生成，主要通过机器人执行器影响关节运动。

**External Force**

由鼠标拖拽产生，直接进入物理引擎的外力计算。

两者并不相同。

例如，鼠标向侧面拖动机器人时，机器人出现倾斜，不代表策略主动决定向侧面倒下。

如果希望研究策略的抗扰动能力，还需要测量施加扰动后是否摔倒、恢复所需时间、最大姿态偏移和最终稳定性。

## 10. Limitations（实验局限）

本实验仍存在以下限制：

- 没有重新训练 MicroDuck 控制策略。
- 没有将 Task 1 的篮球平衡模型适配到 mjswan。
- 未使用固定条件对低速启动行为进行重复实验。
- 未记录机器人位移、关节控制量和实际速度曲线。
- 未测量鼠标拖拽的外力大小。
- 未进行定量抗扰动测试。
- 尚未在真实 MicroDuck 机器人上验证策略。

因此，本实验验证的是浏览器仿真、策略执行和交互式物理扰动能力，而不是机器人控制性能的正式基准测试。

## 11. Future Work（后续计划）

### 11.1 低速启动特性分析

设计不同 Forward 速度指令的实验，研究机器人从静止到运动的响应差异。

例如：

```text
Reset → 0.1 m/s
Reset → 0.2 m/s
Reset → 0.25 m/s
Reset → 0.35 m/s

0.35 → 0.2 m/s
0.35 → 0.1 m/s
```

记录每个条件下的启动时间、实际位移和行走状态。

### 11.2 物理扰动鲁棒性测试

在可重复的外力条件下，研究机器人受到不同方向扰动后的平衡能力。

建议测量：

- Maximum Body Tilt
- Time to Recovery
- Fall Rate
- Joint Action Changes
- Stability Under Disturbance

### 11.3 从浏览器仿真到自主机器人

未来可以将本实验获得的部署经验应用到更复杂的机器人项目中，包括树莓派移动机器人及无人机仿真控制。

不过，移动机器人和无人机具有不同的动力学模型与执行器接口，因此需要重新设计状态空间、动作空间和控制策略，不能直接复用 MicroDuck 的关节控制 ONNX 模型。

## 12. Conclusion（实验总结）

本实验成功完成 MicroDuck 浏览器端策略部署与物理扰动的核心学习任务。

通过 mjswan Playground 构建静态应用，在浏览器中成功加载 MicroDuck 机器人模型和 Walk 策略，并利用 Forward 指令控制机器人运动。

在交互实验中，成功验证了鼠标拖拽能够产生机器人姿态变化。

此外，实验发现机器人在低速启动与持续行走之间存在不同的响应行为，这为后续研究策略状态依赖、步态启动和运动控制提供了一个值得验证的问题。

本实验的主要成果是建立了以下工程流程：

**MicroDuck 模型与 ONNX 策略 → mjswan 静态构建 → 浏览器 MuJoCo WASM 仿真 → 策略执行 → 外力扰动交互。**

与 Task 1 的强化学习继续训练相结合，本项目初步覆盖了机器人策略开发中两个重要环节：**模型训练与模型部署**。

---

## References

1. [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied)
2. [mjswan](https://github.com/ttktjmt/mjswan)
3. [mjswan Playground](https://github.com/ttktjmt/mjswan_playground)
4. [Pollen Robotics MicroDuck](https://github.com/pollen-robotics/microduck)
5. [Pollen Robotics MicroDuck RL](https://github.com/pollen-robotics/microduck_rl)
6. [MuJoCo](https://github.com/google-deepmind/mujoco)