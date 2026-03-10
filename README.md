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

---

## 3. 仓库结构
- **`agents/smart/submission.py`**：核心代码成品文件。
- **`strategy.md`**：详细的 AlphaGo 级强化学习及规则实施方案（英文文件名，中文内容）。
- **`run_log.py`**：本地比赛跑分与评测主入口。
