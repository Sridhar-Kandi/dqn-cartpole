import random
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.deque = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.deque.append((state, action , reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.deque, batch_size)
        states, actions, rewards, next_states,dones = zip(*batch)

        states = torch.FloatTensor(states)

        actions = torch.LongTensor(actions).unsqueeze(1)

        rewards = torch.FloatTensor(rewards).unsqueeze(1)

        next_states = torch.FloatTensor(next_states)

        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    

    def __len__(self):
        return len(self.deque)