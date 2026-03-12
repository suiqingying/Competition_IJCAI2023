# IJCAI 2023 AI Olympics - 智能体小组项目 (Group Smart)

![AI-Olympics_render](imgs/AI-Olympics_render.gif)

## 1. 项目背景与简介
本项目基于 [IJCAI 2023 AI Olympics Competition](http://www.jidiai.cn/compete_detail?compete=34) 的多智能体仿真环境。我们将开发一个能够操控 Agent 在六种不同奥运赛道（Running, Wrestling, Football, Table Hockey, Curling, Billiard）中对战的高性能智能体。

本项目目标是通过规则驱动（Rule-based）与深度强化学习（DRL）相结合，实现最优决策。

---

## 2. 环境要求与运行说明

### 2.1 环境安装 (Conda)
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
conda create -n olympics python=3.8.5 -y
conda run -n olympics python -m pip install -r requirements.txt
```

如需使用 `conda activate olympics`，请先执行一次：
```bash
conda init powershell
```
然后重开 PowerShell。

### 2.2 运行本地测试评测
```bash
conda run -n olympics python run_log.py --my_ai "smart" --opponent "random"
```

### 2.3 UI 可视化展示
```bash
conda run -n olympics python olympics_engine/main.py
```

## 3. RL PPO 训练与评估指南

我们为其中的 **Running (赛跑)** 游戏实现了一个从头搭建的、无死角的工业级 PPO 训练流。它整合了**阶梯难度自动切换 (Auto-Curriculum)**、随机场地、Side Swap 防偏态以及基于优势函数归一化的稳定更新策略。

### 3.1 启动训练 (带 OSD 可视化大屏)
在配置好环境依赖后，运行以下命令可以直观地看到 PPO 正在自我进化，训练参数会在界面的左上角实时跳动展示：
```bash
conda run -n olympics python rl_trainer/train_running.py --render
```
- **基于 Loss 的自动收敛停止机制**：程序移除了写死的回合数上限。加入了独立的泛化验证逻辑。默认情况下，当它完成了全部进阶训练课表，并且能够**稳定保持 C-Loss < 0.015** 超过连续 5 局时，它会宣布自动出师并保存退出。你也可以通过 `--target_c_loss 0.02` 参数来手动控制更宽容的收敛上限。
- 模型文件会自动并定时保存在 `rl_trainer/models/running_ppo/` 目录下。
- 支持断点续传（如果训练意外停止，直接重复运行上列命令，系统自动载入最新的 `.pth` 接着训练）。
- 去掉 `--render` 参数可以转入静默极速后台训练。

### 3.2 本地对抗测试 (评测)
现在 `agents` 文件夹内已经封装了一个完整的强化学习参赛口 `agents/running_ppo/submission.py`。
使用以下命令，让训练好的大脑跟随机对手较量（它会在本地跑满 4 局完整集成环境）：
```bash
conda run -n olympics python run_log.py --my_ai "running_ppo" --opponent "random"
```

---

## 4. 仓库结构
- **`rl_trainer/`**：核心深度强化学习模型训练器。
  - `train_running.py`: PPO 主训练大循环（原生仿真环境交互）。
  - `running_wrapper.py`: 原生环境向 RL MDP 的密集奖励与特征提取包裹器。
  - `algo/running_ppo.py`: PPO (Actor-Critic) 数学核心梯度引擎。
- **`agents/running_ppo/submission.py`**：赛跑游戏强化学习模型直接可执行参赛接口。
- **`agents/smart/submission.py`**：综合规则树参赛接口。
- **`strategy.md`**：详细的 AlphaGo 级强化学习及规则实施方案（技术路线白皮书）。
- **`run_log.py`**：本地比赛跑分与评测主入口。
