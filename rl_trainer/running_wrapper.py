import numpy as np
import math

class RunningEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.n_player = env.agent_num
        self.step_cnt = 0
        self.agent_finished = [False, False]

    def reset(self):
        obs = self.env.reset()
        self.step_cnt = 0
        self.agent_finished = [False, False]
        return self._process_obs(obs)

    def _process_obs(self, obs_list):
        """
        Convert the 40x40 discrete colored grid into a 3-channel (3, 40, 40) float tensor.
        Channel 0: Walls & Tracks (Index 6=Black, 4=Grey)
        Channel 1: Finish Line (Index 7=Red)
        Channel 2: Other Agents & Objects (Index 8, 10, 2, 3, etc)
        """
        processed = []
        for i in range(self.n_player):
            # Compatibility check depending on whether it's dict observation
            if isinstance(obs_list[i], dict):
                matrix = obs_list[i].get('obs', obs_list[i].get('agent_obs'))
                if isinstance(matrix, dict):
                    matrix = matrix.get('agent_obs', matrix)
            else:
                matrix = obs_list[i]
                
            matrix = np.array(matrix)
            
            # Create Multi-channel One-hot features
            ch0 = ((matrix == 6) | (matrix == 4) | (matrix == 1)).astype(np.float32)
            ch1 = (matrix == 7).astype(np.float32)
            ch2 = ((matrix == 8) | (matrix == 10) | (matrix == 2) | (matrix == 3) | (matrix == 5)).astype(np.float32)
            
            stacked = np.stack([ch0, ch1, ch2], axis=0) # Size: (3, 40, 40)
            processed.append(stacked)
        return processed

    def step(self, action_list):
        # Step through the real env
        obs, reward, done, info = self.env.step(action_list)
        
        # Dense Reward Shaping
        shaped_reward = list(reward)
        
        core = self.env 
        
        for i in range(self.n_player):
            agent = core.agent_list[i]
            
            # === 【核心修复: 即时终点奖惩】 ===
            # 如果这名特工在这一步【刚刚】越过终点线
            if not self.agent_finished[i] and agent.finished:
                self.agent_finished[i] = True
                
                # 如果对方还没过线，说明我是第一名！(大奖 +100，对方被立即嘲讽 -50)
                if not self.agent_finished[1-i]:
                    shaped_reward[i] += 100.0
                    shaped_reward[1-i] -= 50.0
                # 如果对方已经过线，说明是我输了，但不在这里重复扣分（因为当时自己已经被扣过了）

            if not agent.finished:
                # 1. Time Penalty (encourage moving fast)
                shaped_reward[i] -= 0.01
                
                # 2. Velocity Reward (encourage maintaining momentum)
                vel = core.agent_v[i]
                speed = math.hypot(vel[0], vel[1])
                shaped_reward[i] += (speed / core.speed_cap) * 0.02
                
                # 3. Stuck Penalty
                if speed < 1.0:
                    shaped_reward[i] -= 0.05

        self.step_cnt += 1
        return self._process_obs(obs), shaped_reward, done, info
