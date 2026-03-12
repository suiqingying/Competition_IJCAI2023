# 🏃 竞速赛 (Running) 完整 RL 技术开发路线

> **目标**: 针对无对抗的项目 "Running"，搭建、训练并部署一个高水平的 PPO 模型。这套流程将作为基础范本，后续可横向扩展到复杂的对抗游戏中（再另外加上 Self-Play）。

我们将分五个明确的阶段（Phase）在本项目中逐步实现。

---

## 📍 Phase 1: 观测预处理 (Observation Wrapper)
**原初痛点**: 环境给出的 40x40 矩阵中，6 代表墙，7 代表终点。如果直接丢给神经网络，模型由于卷积特性会认为数值 `7 > 6`，这在离散分类特征上是荒唐的，极度阻碍收敛。
**技术方案**: 将单通道类别矩阵转换为多通道的二进制特征图 (Multi-channel One-hot Encoding)。
1. **构建 `ObservationWrapper`** (`rl_trainer/running_wrapper.py`)。
2. **提取通道特征**:
  - `Channel 0`: 障碍物通道（黑色墙壁、灰色和绿色边界）。
  - `Channel 1`: 目标通道（红色的交叉/终点线）。
  - `Channel 2`: 自身与其他对象通道（包含蓝、红球体）。
3. **输出维数**: `(3, 40, 40)` Float Tensor。

---

## 📍 Phase 2: 奖励塑形 (Dense Reward Shaping)
**原初痛点**: 环境默认只有跑完全程才给一次 `+100` 或 `+1` 奖励。在这类包含多个弯道的几百步迷宫中，这是极度的“奖励稀疏”，PPO 会在前期由于找不到任何正反馈而迷失并撞墙死亡。
**技术方案**: 注入物理反馈，将稀疏奖励变为“密集奖励” (Dense Reward)。
1. **存活惩罚 (Time Penalty)**：每走一步给予 `-0.01`，逼迫 agent 尽快跑完，绝不原地转圈。
2. **动能奖励 (Velocity Reward)**：读取引擎底层数据 `env.env_core.agent_v`，给予 $(Speed / SpeedCap) \times 0.02$ 的奖励，鼓励维持高速运动。
3. **僵直惩罚 (Stuck Penalty)**：如果检测到速度低于极小值且未在终点，追加较大负奖励惩罚其卡在死角的行为 `-0.05`。
4. **终点放大 (Win Bonus)**：一旦跨越终点线，追加巨额奖励 `+100`，使长期目标价值远超短期惩罚。

---

## 📍 Phase 3: 网络架构与算法强化 (Architecture Upgrade)
**原初痛点**: 原始 PPO 实现里的 `CNN_Actor` 随意堆叠，且维度写死为 `in_channel=8`。
**技术方案**: 引入强化版 CNN 结构并适配连续-离散决策映射。
1. **三层重卷结构** (`rl_trainer/algo/running_network.py`):
   - `Conv2d(3 -> 16)` -> `BatchNorm2d` -> `ReLU`
   - `Conv2d(16 -> 32)` -> `BatchNorm2d` -> `ReLU`
   - `Conv2d(32 -> 64)` -> `BatchNorm2d` -> `ReLU`
   - 最终展平并通过全连接层输出。
2. **独立 PPO Core** (`rl_trainer/algo/running_ppo.py`): 重构一个专属于本策略的稳定 PPO 类，解决原有 PPO 多维度 Gather 操作不严谨的问题。

---

## 📍 Phase 4: 编写训练管道 (Training Pipeline)
**原初痛点**: `train_on_subgame.py` 处理的数据流非常乱，不利于加入刚才的 Wrapper 逻辑。
**技术方案**:
1. **新建隔离的训练脚本** (`rl_trainer/train_running.py`)以保持代码整洁。
2. **主训练 Loop**: 
   - 加载环境并包裹 `RunningEnvWrapper`。
   - 配置环境：以 PPO agent 作为 player 1，设定基线 random agent 为 player 0。
   - 收集 `buffer_capacity`（如 2000步）的 Transitions 触发 PPO 的 Actor-Critic 网络更新。
3. **日志监控**: 集成 Tensorboard 输出 Reward 与回合完成度，并定时保存 `running_actor.pth`。

---

## 📍 Phase 5: 模型加载与评测提交 (Deployment)
**目标**: 将训出的高能模型直接塞入最终的执行提交中。
1. **部署**: 在 `agents/smart/submission.py` 开头进行判断，实例化 `RunningCNN_Actor` 并使用 `torch.load` 加载权重文件。
2. **融合**: `strategy_running()` 删去原有的测距逻辑代码，直接读取模型预测动作。
3. **单机验证**: 最终通过 `python run_log.py` 检查对打随机对手的 100% 胜率和稳定竞速。
