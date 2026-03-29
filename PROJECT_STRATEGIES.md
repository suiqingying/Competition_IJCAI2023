# Olympics Integrated 项目分析

## 1. 范围

本文档只分析正式 integrated 提交路径：

- `run_log.py`
- `env/olympics_integrated.py`
- `olympics_engine/AI_olympics.py`

不使用 `olympics_engine/main.py` 的演示入口做结论。

## 2. 正式提交接口

### 2.1 控制器输入

`my_controller(observation, action_space, is_act_continuous)` 中：

- `observation` 顶层字段：
  - `obs`
  - `controlled_player_index`
- `observation["obs"]` 内部字段：
  - `agent_obs`
  - `id`
  - `energy`
  - `game_mode`

其中：

- `agent_obs` 是局部观测
- `game_mode == "NEW GAME"` 表示刚切换到新的子项目
- `controlled_player_index` 是当前控制方编号

README 对 integrated 观测的说明是：

- 主体为 `40 x 40` 的二维观测
- 附带能量和切换标志等侧信息

### 2.2 控制器输出

动作是连续二维控制：

- 力：`[-100, 200]`
- 转角：`[-30, 30]`

提交输出格式：

```python
[[force], [angle]]
```

### 2.3 integrated 调度与计分

- 子项目共 6 个：
  - `running`
  - `football`
  - `table hockey`
  - `wrestling`
  - `curling`
  - `billiard`
- 子项目顺序随机打乱
- integrated 胜负由 6 个子项目的赢局数决定
- `AI_Olympics` 结束时对外返回：
  - `[100, 0]`
  - `[0, 100]`
  - `[0, 0]`

因此，正式提交的整体结构应为：

```text
子项目识别 + 子项目专用策略
```

## 3. 总表

| 项目 | 正式评测地图数 | 分支情况 | 是否需要对抗 | 打分性质 | 推荐主方法 |
| --- | --- | --- | --- | --- | --- |
| Running | 4 | 地图随机 | 是 | 以终局输赢为主 | Recurrent PPO |
| Football | 1 | 球初始位置随机 | 是，强对抗 | 以终局输赢为主 | Self-play PPO |
| Table Hockey | 1 | 球初始位置随机 | 是，强对抗 | 以终局输赢为主 | Self-play PPO |
| Wrestling | 1 | 碰撞与边界状态分支 | 是，最强对抗 | 以终局输赢为主 | Self-play PPO + 对手池 |
| Curling | 1 | 两局、换先后手、轮次分支 | 是，间接对抗 | 需要过程量 | 状态重建 + Shot Planning |
| Billiard | 1 | 剩余球、白球入袋、总分分支 | 是，间接对抗 | 需要过程量 | 状态重建 + Shot Planning |

## 4. 地图与分支

### 4.1 Running

- 正式 integrated 采样地图：4 张
- 采样逻辑：`random.randint(1,4)`
- 地图资源总数：11 张
- 资源文件：`olympics_engine/scenario/running_competition_maps/maps.json`

分支：

- 地图分支：`map1` 到 `map4`
- 对抗分支：碰撞、卡位、抢线
- 状态分支：能量变化、速度变化、过线顺序

### 4.2 Football

- 固定 1 张球场
- 无地图分支

分支：

- 球初始 `y` 位置随机
- 球权变化
- 反弹路径变化
- 门前攻防变化

### 4.3 Table Hockey

- 固定 1 张球台
- 无地图分支

分支：

- 球初始 `y` 位置随机
- 球反弹路径变化
- 半场站位变化
- 门前防守变化

### 4.4 Wrestling

- 固定 1 张擂台
- 无地图分支

分支：

- 碰撞结果分支
- 朝向变化
- 边界压力变化
- 双方相对站位变化

### 4.5 Curling

- 固定 1 张冰壶图
- 无地图分支

分支：

- 两局制
- 每队 `max_n = 3`
- 第一局结束后换先后手
- 非当前队观测为全 `-1`
- 每一壶落点与碰撞会改变后续局面

### 4.6 Billiard

- 固定 1 张台球桌
- 无地图分支

分支：

- 每边 `max_n_hit = 3`
- 白球可能入袋并重置
- 剩余球数量变化
- `pot_reward`
- `white_penalty`
- `total_score`

## 5. 对抗性与打分性质

### 5.1 对抗性

强直接对抗：

- `football`
- `table hockey`
- `wrestling`

中等直接对抗：

- `running`

间接对抗：

- `curling`
- `billiard`

### 5.2 打分性质

只需以输赢为核心组织策略：

- `running`
- `football`
- `table hockey`
- `wrestling`

必须利用过程量：

- `curling`
- `billiard`

说明：

- integrated 总赛最终都折算为子项目输赢
- 但 `curling` 和 `billiard` 的单项目策略不能只看终局

## 6. 统一提交架构

### 6.1 结构

建议结构：

1. `game_mode == "NEW GAME"` 时清空上一项目状态
2. 用前 1 到 3 帧识别当前子项目
3. 路由到该项目专用策略
4. 该项目结束前保持同一策略实例

### 6.2 子项目识别

优先用规则识别，不先用分类网络。

可用顺序：

1. `agent_obs` 几乎全为 `-1`
   - 优先判为 `curling` 的非当前回合
2. 画面存在明显圆形边界
   - 判为 `wrestling`
3. 画面存在多个球和袋口结构
   - 判为 `billiard`
4. 画面是长通道、赛道、终点线结构
   - 判为 `running`
5. 剩余在 `football` 和 `table hockey` 中区分
   - 用球门结构和边界结构区分

### 6.3 动作层

RL 项目建议训练时使用仓库已有的 36 个离散动作格点，再映射回连续动作输出。

原因：

- 仓库示例训练脚本已采用该离散化
- 训练与调试成本更低
- 便于先得到稳定基线

规划项目直接生成候选 shot，不必走离散 RL。

## 7. 分项目方案

### 7.1 Running

推荐方法：

- `CNN + GRU/LSTM + PPO`
- 或 `Frame Stack + PPO`

输入：

- `observation["obs"]["agent_obs"]`
- `observation["obs"]["energy"]`
- 上一步动作
- 最近若干帧历史

动作：

- 使用 36 个离散动作格点

训练信号：

- 终局胜利奖励
- 小的时间惩罚
- 可选的小幅前进 shaping

要求：

- 4 张图混训
- 需要部分观测记忆
- 需要对抗训练或历史对手回放

最低可交付版本：

- `Frame Stack + PPO`
- 4 图混训
- 随机对手

### 7.2 Football

推荐方法：

- `Self-play PPO`

输入：

- 当前 `agent_obs`
- `energy`
- 最近几帧历史

动作：

- 使用 36 个离散动作格点

训练信号：

- 终局进球输赢
- 可加轻量 shaping：
  - 球朝对方球门移动
  - 球逼近我方球门惩罚
  - 短时控球奖励

训练组织：

- 先弱对手
- 再 self-play
- 保存历史快照

最低可交付版本：

- `Frame Stack + PPO`
- self-play

### 7.3 Table Hockey

推荐方法：

- `Self-play PPO`

输入：

- 当前 `agent_obs`
- `energy`
- 最近几帧历史

动作：

- 使用 36 个离散动作格点

训练信号：

- 终局进球输赢
- 可加轻量 shaping：
  - 球在对方半场
  - 球逼近我方球门惩罚
  - 门线覆盖奖励

训练组织：

- self-play
- 强化守门站位
- 强化贴墙反弹处理

最低可交付版本：

- `Frame Stack + PPO`
- self-play

### 7.4 Wrestling

推荐方法：

- `Self-play PPO + 历史对手池`

输入：

- 当前 `agent_obs`
- `energy`
- 最近几帧历史

动作：

- 使用 36 个离散动作格点

训练信号：

- 终局输赢
- 可加轻量 shaping：
  - 自己接近边界惩罚
  - 对手接近边界奖励
  - 中心区域控制奖励

训练组织：

- 先学不自杀出界
- 再学中心控制
- 再学主动撞击
- 保留历史模型做对手池

最低可交付版本：

- `Frame Stack + PPO`
- self-play
- 周期性冻结历史对手

### 7.5 Curling

推荐方法：

- `状态重建 + Shot Planning`

输入：

- 当前可见壶位置
- 壶心相对位置
- 当前局数
- 当前轮次
- 剩余投掷次数
- 历史局面缓存

动作：

- 直接生成候选 `力度 x 角度`

评分函数：

- 我方最近壶到圆心距离
- 对方最近壶到圆心距离
- 我方有效壶数
- 对方有效壶数
- 是否撞走对手关键壶
- 是否留下防守位

实现方式：

1. 从观测重建局面
2. 枚举候选 shot
3. 用评分函数排序
4. 对前若干个候选做细搜索

最低可交付版本：

- 基于模板或连通域重建壶位置
- 固定候选集合
- 手工评分函数

### 7.6 Billiard

推荐方法：

- `状态重建 + Shot Planning`

输入：

- 我方白球位置
- 对方白球位置
- 我方剩余目标球
- 对方剩余目标球
- 袋口位置
- 当前分差
- 历史局面缓存

动作：

- 直接生成候选击球：
  - 目标球
  - 袋口
  - 方向
  - 力度

评分函数：

- 是否能直接进球
- 白球入袋风险
- 进球后留位
- 总分差变化
- 防守位质量

实现方式：

1. 先做直线可进球检测
2. 再做遮挡和碰撞检查
3. 对可行 shot 调力度
4. 无直接进球时选防守 shot

最低可交付版本：

- 几何直线检测
- 简单风险打分
- 候选击球选择

## 8. 实现顺序

建议顺序：

1. 统一提交外壳
2. `running`
3. `wrestling`
4. `football`
5. `table hockey`
6. `billiard`
7. `curling`
8. integrated 回归测试

原因：

- 前 4 项可复用 RL 框架
- 后 2 项适合独立做规划器
- 最后再做 integrated 级联调试

## 9. 结论

推荐算法：

- `running`：Recurrent PPO
- `football`：Self-play PPO
- `table hockey`：Self-play PPO
- `wrestling`：Self-play PPO + 对手池
- `curling`：状态重建 + Shot Planning
- `billiard`：状态重建 + Shot Planning

地图与分支：

- `running`：正式评测 4 张图，地图随机分支明显
- 其余 5 项：固定 1 张地图，主要是状态分支和流程分支

对抗性：

- 6 项都需要对抗
- 强直接对抗：`football / table hockey / wrestling`
- 中等直接对抗：`running`
- 间接对抗：`curling / billiard`

打分性质：

- 以输赢为核心：`running / football / table hockey / wrestling`
- 需要过程量：`curling / billiard`

## 10. 依据文件

- `README.md`
- `run_log.py`
- `env/olympics_integrated.py`
- `olympics_engine/AI_olympics.py`
- `olympics_engine/scenario/running_competition.py`
- `olympics_engine/scenario/football.py`
- `olympics_engine/scenario/table_hockey.py`
- `olympics_engine/scenario/wrestling.py`
- `olympics_engine/scenario/curling_competition.py`
- `olympics_engine/scenario/billiard_competition.py`
- `olympics_engine/scenario.json`
- `olympics_engine/scenario/running_competition_maps/maps.json`
- `rl_trainer/train_on_subgame.py`
- `rl_trainer/main.py`
- `agents/random/submission.py`
