import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import time
import numpy as np
from .running_network import RunningCNN_Actor, RunningCNN_Critic

class Args:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 2000
    batch_size = 64
    gamma = 0.99
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args()

class RunningPPO:
    def __init__(self, run_dir, action_space):
        self.actor_net = RunningCNN_Actor(action_space).to(args.device)
        self.critic_net = RunningCNN_Critic().to(args.device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=args.lr)

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        self.run_dir = run_dir

    def select_action(self, state, train=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(args.device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        
        dist = Categorical(action_prob)
        if train:
            action = dist.sample()
        else:
            action = torch.argmax(action_prob)
        
        # Return action and its log_prob
        return action.item(), dist.log_prob(action).item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self):
        if len(self.buffer) < args.batch_size:
            return 0, 0
            
        states = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(args.device)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(args.device)
        old_action_log_probs = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(args.device)
        # 【重要修复1：Reward缩放】将巨大的100分奖励缩小10倍，稳定Critic均方误差爆炸。
        rewards = [t.reward * 0.1 for t in self.buffer]
        dones = [t.done for t in self.buffer]

        # 【重要修复2：采用 GAE (Generalized Advantage Estimation)】打通3000步的奖励长梯！
        with torch.no_grad():
            values = self.critic_net(states).view(-1).cpu().numpy()

        gae = 0
        advantages = []
        lmbda = 0.95
        for step in reversed(range(len(self.buffer))):
            if step == len(self.buffer) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + args.gamma * next_value * next_non_terminal - values[step]
            gae = delta + args.gamma * lmbda * next_non_terminal * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float).to(args.device).view(-1, 1)
        returns = advantages + torch.tensor(values, dtype=torch.float).to(args.device).view(-1, 1)

        # 【重要修复3：全局势能归一化】在全样本上归一化，而不是在 MiniBatch 上，消除抽样偏差
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        a_loss_sum, c_loss_sum = 0, 0
        update_cnt = 0

        for _ in range(args.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), args.batch_size, False):
                
                # Critic Update
                V = self.critic_net(states[index])
                value_loss = F.mse_loss(returns[index], V)
                
                # Actor Update
                action_prob = self.actor_net(states[index])
                dist = Categorical(action_prob)
                action_log_prob = dist.log_prob(actions[index].squeeze()).view(-1, 1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(action_log_prob - old_action_log_probs[index])
                surr1 = ratio * advantages[index]
                surr2 = torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param) * advantages[index]
                
                # Add entropy bonus to encourage exploration
                action_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), args.max_grad_norm)
                self.critic_optimizer.step()
                
                a_loss_sum += action_loss.item()
                c_loss_sum += value_loss.item()
                update_cnt += 1
                self.training_step += 1

        del self.buffer[:]
        self.counter = 0
        return a_loss_sum/update_cnt, c_loss_sum/update_cnt

    def save(self):
        torch.save(self.actor_net.state_dict(), os.path.join(self.run_dir, 'actor.pth'))
        torch.save(self.critic_net.state_dict(), os.path.join(self.run_dir, 'critic.pth'))

    def load(self, run_dir):
        self.actor_net.load_state_dict(torch.load(os.path.join(run_dir, 'actor.pth'), map_location=args.device))
        self.critic_net.load_state_dict(torch.load(os.path.join(run_dir, 'critic.pth'), map_location=args.device))
