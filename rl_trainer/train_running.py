import os
import sys
import argparse
import torch
import random
import numpy as np
import pygame

# 添加搜索路径，确保能 import env 和自定义算法
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "olympics_engine"))

from olympics_engine.scenario.running_competition import Running_competition
from olympics_engine.generator import create_scenario
from rl_trainer.algo.running_ppo import RunningPPO
from rl_trainer.running_wrapper import RunningEnvWrapper

def draw_text(surface, text, x, y, color=(255, 255, 255), size=20):
    font = pygame.font.SysFont("SimHei", size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_c_loss', type=float, default=0.015, help='目标 Critic Loss，达到后自动收敛停止')
    parser.add_argument('--save_interval', type=int, default=50, help='Save every N episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Visualize in Pygame + OSD stats')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    action_space = 36 
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'running_ppo')

    print(f"=============================================")
    print(f"🚀 初始化 真实场景 Running PPO 训练器 (带 OSD 可视化)")
    print(f"=============================================")

    # 初始化强化学习模型 PPO
    agent = RunningPPO(run_dir, action_space)
    if os.path.exists(os.path.join(run_dir, 'actor.pth')):
        print("✅ 权重载入成功，断点续演...")
        agent.load(run_dir)
    else:
        print("🌱 新航程开始，从 0 训练...")

    running_meta_map = create_scenario("running-competition")
    
    # 统计数据
    history_scores = []
    last_a_loss, last_c_loss = 0, 0

    if args.render:
        pygame.init()

    try:
        i_ep = 0
        consecutive_pass = 0
        curriculum_level = 1
        
        while True:
            # 阶梯难度 (由 Loss 驱动的层级)
            if curriculum_level == 1: map_id = 1
            elif curriculum_level == 2: map_id = random.choice([1, 2])
            elif curriculum_level == 3: map_id = random.choice([1, 2, 3])
            else: map_id = random.randint(1, 4)
            
            base_env = Running_competition(meta_map=running_meta_map, map_id=map_id, vis=200, vis_clear=5, agent1_color='purple', agent2_color='green')
            base_env.max_step = 3000  # 【延长时间上限】：从默认的 400 步增加到 3000步，防止球还没跑到终点环境就强制中止
            env = RunningEnvWrapper(base_env)
            obs_list = env.reset()

            rl_agent_id = random.choice([0, 1])
            random_agent_id = 1 - rl_agent_id
            
            color_name_zh = "紫球 (Purple)" if rl_agent_id == 0 else "绿球 (Green)"
            color_pygame = (128, 0, 128) if rl_agent_id == 0 else (0, 255, 0)

            ep_reward = 0
            step_in_ep = 0
            done = False

            while not done:
                joint_actions = [None, None]
                act_random = [random.uniform(-100, 200), random.uniform(-30, 30)]

                # RL Decision
                obs_rl = obs_list[rl_agent_id] 
                action_idx, action_log_prob = agent.select_action(obs_rl, train=True)
                
                computed_force = -100.0 + (action_idx // 6) * 60.0
                computed_angle = -30.0 + (action_idx % 6) * 12.0
                act_rl = [computed_force, computed_angle]

                joint_actions[random_agent_id] = act_random
                joint_actions[rl_agent_id] = act_rl

                obs_list_new, reward, done, _ = env.step(joint_actions)
                
                if args.render:
                    base_env.render()
                    # 获取屏幕进行 OSD 绘制
                    screen = pygame.display.get_surface()
                    if screen:
                        # 绘制半透明挡板背景 (可选)
                        # overlay = pygame.Surface((250, 180), pygame.SRCALPHA)
                        # overlay.fill((0, 0, 0, 150))
                        # screen.blit(overlay, (10, 10))
                        
                        draw_text(screen, f"Episode: {i_ep+1}", 20, 20, (255, 255, 0))
                        draw_text(screen, f"Map ID: {map_id}", 20, 50)
                        draw_text(screen, f"Agent: {color_name_zh}", 20, 80, color_pygame)
                        draw_text(screen, f"Step: {step_in_ep}", 20, 110)
                        draw_text(screen, f"Reward: {ep_reward:.1f}", 20, 140, (0, 255, 255))
                        draw_text(screen, f"A-Loss: {last_a_loss:.4f}", 20, 170, (255, 100, 100))
                        draw_text(screen, f"C-Loss: {last_c_loss:.4f}", 20, 200, (100, 255, 100))
                        pygame.display.flip()

                reward_rl = reward[rl_agent_id]
                
                # 记忆存储 (加入了 done，防止跨局导致的回报污染)
                agent.store_transition(type('T', (object,), {'state': obs_rl, 'action': action_idx, 'a_log_prob': action_log_prob, 'reward': reward_rl, 'done': done})())

                obs_list = obs_list_new
                ep_reward += reward_rl
                step_in_ep += 1

            # 【防卡顿】: 将反向传播更新移动到本局结束之后！
            # 这样就不会在跑车的途中突然卡住计算梯度了。
            if agent.counter >= 2000:
                last_a_loss, last_c_loss = agent.update()

            history_scores.append(ep_reward)
            avg_score = np.mean(history_scores[-20:]) if history_scores else 0
            print(f"🎬 Ep: {i_ep+1} | Map: {map_id} | Side: {rl_agent_id} | Reward: {ep_reward:.1f} | Avg20: {avg_score:.1f}")

            if (i_ep + 1) % args.save_interval == 0:
                agent.save()
                print(f"💾 第 {i_ep+1} 局结束，存档完成。")

            # === 阶梯难度动态晋级 (基于 Loss 和 分数) ===
            # C-Loss 下降代表看懂赛道了，Reward 较高且平滑代表动作做对了
            if curriculum_level == 1 and avg_score > 70 and 0 < last_c_loss < 0.05 and i_ep > 50:
                curriculum_level = 2
                print(f"\n🚀 【模型突破】看懂了直道 (Map 1)！难度升级，加入 U型弯 (Map 2)！\n")
            elif curriculum_level == 2 and avg_score > 75 and 0 < last_c_loss < 0.04 and i_ep > 100:
                curriculum_level = 3
                print(f"\n🚀 【模型突破】搞定了 U型弯！难度再升级，加入 蛇形弯 (Map 3)！\n")
            elif curriculum_level == 3 and avg_score > 75 and 0 < last_c_loss < 0.03 and i_ep > 150:
                curriculum_level = 4
                print(f"\n🔥 【终极试炼开启】所有难度地图全开混战 (Map 1~4)！\n")

            # === AI 自动毕业检测 (基于 Loss 的终结机制) ===
            # 只有在最高难度(4级) 才有资格触发毕业，并且 C-Loss 极小
            if curriculum_level == 4:
                if 0 < last_c_loss < args.target_c_loss and abs(last_a_loss) < 0.05 and ep_reward > 80:
                    consecutive_pass += 1
                else:
                    consecutive_pass = 0

                if consecutive_pass >= 5:
                    print(f"\n🎉 达成极致收敛条件！网络 Loss 低于限定阈值且完美泛化。")
                    print(f"📊 最终指标 -> C-Loss: {last_c_loss:.4f} | A-Loss: {last_a_loss:.4f} | Reward: {ep_reward:.1f}")
                    print(f"🥇 【模型已出师】已主动结束无效的死循环，安全封存最强权重。")
                    agent.save()
                    break

            i_ep += 1

    except KeyboardInterrupt:
        print("\n\n⚠️ 中断中... 存档退出...")
        agent.save()
        sys.exit(0)

if __name__ == '__main__':
    main()
