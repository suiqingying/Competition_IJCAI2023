# IJCAI 2023 AI Olympics - 智能体小组项目 (Group Smart)

![AI-Olympics_render](imgs/AI-Olympics_render.gif)

## 1. 项目背景与简介
本项目基于 [IJCAI 2023 AI Olympics Competition](http://www.jidiai.cn/compete_detail?compete=34) 的多智能体仿真环境。我们将开发一个能够操控 Agent 在六种不同奥运赛道（Running, Wrestling, Football, Table Hockey, Curling, Billiard）中对战的高性能智能体。

本项目目标是通过规则驱动（Rule-based）与深度强化学习（DRL）相结合，实现对 Random Baseline 的 100% 胜率压制。

---

## 2. 项目规划与团队路线图 (Project Roadmap)

我们按四个核心阶段推进小组作业：

### ✅ 阶段 1：环境搭建与基础基线 (已完成)
- 熟悉 Jidi 平台和 40x40 像素局部观察矩阵（POMDP）的输入输出机制。
- 配置 Conda 隔离环境并跑通 `run_log.py` 评价流水线。

### 🔄 阶段 2：专家系统与物理逻辑优化 (进行中)
- **视觉特征提取**：精准定位球、线、对手的相对坐标质心。
- **全局航向追踪 (Theta Tracking)**：通过积分算法解决旋转后的失位问题，防止乌龙球。
- **参数化物理控制**：针对冰壶实现 `170力矩 / 7帧时长` 的完美停靠方案。
- **防死锁机制**：计算视野像素熵，实时判定卡墙状态并触发逃逸动作。

### ⏳ 阶段 3：AlphaGo 级 DRL 演化与自我博弈
- **轨迹预判 (Look-ahead)**：预测球体在高速碰撞下的反弹落位。
- **自我博弈 (Self-Play)**：建立 ELO 排名池进行异步对抗训练，习得阻挡与干扰战术。

### 📝 阶段 4：答辩筹备与总结
- 代码合规性与性能优化（内存 < 500M）。
- 胜率统计录制与项目结课演示报告编写。

---

## 3. 环境要求与运行说明

### 3.1 环境安装 (Conda)
```bash
conda create -n olympics python=3.8.5 -y
conda activate olympics
pip install -r requirements.txt
```

### 3.2 运行本地测试评测
```bash
conda run -n olympics python run_log.py --my_ai "smart" --opponent "random"
```

### 3.3 UI 可视化展示
```bash
conda run -n olympics python olympics_engine/main.py
```

---

## 4. 仓库结构
- **`agents/smart/submission.py`**：核心代码成品文件。
- **`strategy.md`**：详细的 AlphaGo 级强化学习及规则实施方案（英文文件名，中文内容）。
- **`run_log.py`**：本地比赛跑分与评测主入口。
