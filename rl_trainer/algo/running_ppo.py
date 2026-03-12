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
            
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(args.device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(args.device)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(args.device)
        rewards = [t.reward for t in self.buffer]
        dones = [t.done for t in self.buffer]

        # 【核心修复 1: 打破局数污染】计算 Return (Gt) 时如果遇到本局结束(d=True)，则收益清零阻断。
        R = 0
        Gt = []
        for r, d in zip(rewards[::-1], dones[::-1]):
            if d:
                R = 0
            R = r + args.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(args.device).view(-1, 1)

        a_loss_sum, c_loss_sum = 0, 0
        update_cnt = 0

        for _ in range(args.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), args.batch_size, False):
                # Critic Update
                V = self.critic_net(state[index])
                advantage = (Gt[index] - V).detach()
                
                # 【核心修复 2: 梯度归一化】极大稳定 PPO 的剧烈震荡，防止“水平突然下降”
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
                # Actor Update
                action_prob = self.actor_net(state[index])
                dist = Categorical(action_prob)
                action_log_prob = dist.log_prob(action[index].squeeze()).view(-1, 1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - args.clip_param, 1 + args.clip_param) * advantage
                
                # Add entropy bonus to encourage exploration
                action_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), args.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt[index], V)
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
