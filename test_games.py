"""测试当前策略在每个子游戏中的胜负表现"""
import sys, os
sys.path.append("./olympics_engine")
from env.chooseenv import make

# 导入两个 agent
sys.path.append(os.path.join(os.path.dirname(__file__), "agents", "smart"))
from importlib import import_module
import agents.smart.submission as smart_mod
import agents.random.submission as random_mod
smart_agent = smart_mod.my_controller
random_agent = random_mod.my_controller

def run_test(n_episodes=5):
    # 统计每个子游戏的胜负
    per_game_stats = {}  # game_name -> {"win":0, "lose":0, "draw":0}
    overall = {"win": 0, "lose": 0, "draw": 0}

    for ep in range(n_episodes):
        env = make("olympics-integrated")  # 每轮重建以 shuffle 地图
        action_space = env.joint_action_space[0]
        obs = env.reset()
        done = False

        # 追踪当前游戏
        core = env.env_core
        prev_score = list(core.game_score)  # [0, 0]
        current_game_name = core.game_pool[core.selected_game_idx_pool[core.current_game_count]]["name"]
        game_order = [core.game_pool[core.selected_game_idx_pool[i]]["name"] for i in range(6)]
        
        print(f"\n{'='*50}")
        print(f"Episode {ep+1} | Game order: {game_order}")
        print(f"{'='*50}")

        game_results_this_ep = []

        while not done:
            # random=player0, smart=player1
            action0 = random_agent(obs[0], action_space, True)
            action1 = smart_agent(obs[1], action_space, True)
            obs, reward, done, _, _ = env.step([action0, action1])
            
            new_score = list(core.game_score)
            
            # 检测是否刚切换了游戏（得分变化了或已结束）
            if new_score != prev_score or done:
                # 判断上一个游戏谁赢了
                diff0 = new_score[0] - prev_score[0]
                diff1 = new_score[1] - prev_score[1]
                
                if diff1 > 0:
                    result = "WIN ✅"
                    result_key = "win"
                elif diff0 > 0:
                    result = "LOSE ❌"
                    result_key = "lose"
                else:
                    result = "DRAW ➖"
                    result_key = "draw"
                
                if current_game_name not in per_game_stats:
                    per_game_stats[current_game_name] = {"win": 0, "lose": 0, "draw": 0}
                per_game_stats[current_game_name][result_key] += 1
                game_results_this_ep.append(f"  {current_game_name}: {result}")
                
                prev_score = list(new_score)
                
                # 更新当前游戏名
                if not done and hasattr(core, 'current_game_count'):
                    idx = core.selected_game_idx_pool[core.current_game_count]
                    current_game_name = core.game_pool[idx]["name"]

        # 打印本轮结果
        for r in game_results_this_ep:
            print(r)
        
        final_score = core.game_score
        if final_score[1] > final_score[0]:
            ep_result = "SMART WINS! ✅"
            overall["win"] += 1
        elif final_score[0] > final_score[1]:
            ep_result = "RANDOM WINS! ❌"
            overall["lose"] += 1
        else:
            ep_result = "DRAW ➖"
            overall["draw"] += 1
        print(f"  >> Final: {final_score[0]}:{final_score[1]} → {ep_result}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"\n  Overall: {overall['win']}W / {overall['lose']}L / {overall['draw']}D")
    print(f"\n  Per-game breakdown:")
    for game_name in sorted(per_game_stats.keys()):
        s = per_game_stats[game_name]
        total = s['win'] + s['lose'] + s['draw']
        wr = s['win'] / total * 100 if total > 0 else 0
        print(f"    {game_name:30s}  {s['win']}W / {s['lose']}L / {s['draw']}D  (win rate: {wr:.0f}%)")

if __name__ == "__main__":
    run_test(5)
