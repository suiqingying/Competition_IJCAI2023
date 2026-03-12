import os
import sys
import numpy as np
import torch

from pathlib import Path
current_path = str(Path(__file__).resolve().parent)
sys.path.append(current_path)

from running_network import RunningCNN_Actor

class RLAgent:
    def __init__(self, model_dir):
        self.device = torch.device("cpu")
        self.action_space = 36
        self.actor_net = RunningCNN_Actor(self.action_space).to(self.device)
        self.actor_net.eval()
        
        # Load weights
        weight_path = os.path.join(model_dir, 'actor.pth')
        if os.path.exists(weight_path):
            self.actor_net.load_state_dict(torch.load(weight_path, map_location=self.device))
            print(f"Loaded trained PPO model from {weight_path}")
        else:
            print(f"Warning: No trained model found at {weight_path}. Using random initialization.")

    def _process_obs(self, obs_dict):    
        matrix = obs_dict.get('obs', obs_dict.get('agent_obs', obs_dict))
        if isinstance(matrix, dict):
            matrix = matrix.get('agent_obs', matrix)
            
        matrix = np.array(matrix)
        
        # Channel 0: Walls & Tracks (Index 6=Black, 4=Grey)
        ch0 = ((matrix == 6) | (matrix == 4) | (matrix == 1)).astype(np.float32)
        # Channel 1: Finish Line (Index 7=Red)
        ch1 = (matrix == 7).astype(np.float32)
        # Channel 2: Other Agents & Objects (Index 8, 10, 2, 3, etc)
        ch2 = ((matrix == 8) | (matrix == 10) | (matrix == 2) | (matrix == 3) | (matrix == 5)).astype(np.float32)
        
        stacked = np.stack([ch0, ch1, ch2], axis=0)
        return stacked

    def select_action(self, obs_dict):
        state = self._process_obs(obs_dict)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_prob = self.actor_net(state_tensor)
        
        # Inference mode: take the most probable action
        action_idx = torch.argmax(action_prob).item()
        
        # Decode action to Force and Angle
        computed_force = -100.0 + (action_idx // 6) * 60.0
        computed_angle = -30.0 + (action_idx % 6) * 12.0
        
        return computed_force, computed_angle

# Initialize global agent instance (persists across step calls in Jidi)
# Ensure we map the model_dir to where the trained models are saved.
model_saved_dir = current_path
global_agent = RLAgent(model_saved_dir)

def my_controller(observation, action_space, is_act_continuous=False):
    """
    Standard interface for Jidi Platform validation & RL evaluation.
    """
    obs_dict = observation
    force, angle = global_agent.select_action(obs_dict)
    
    # Platform wrappers usually expect [[force], [angle]] for 2D continuous actions.
    return [[force], [angle]]
