import torch
import numpy as np
from buffers import ReplayBuffer
from models import MLP

class RewardModel(object):
    
    def __init__(self, args):
        
        self.replay_buffer = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device=args.device)
        
        self.policy = 
        self.Q1 = 
        self.Q2 = 
        self.Q1_target = 
        self.Q2_target = 
        
    def update(self):
        
    def evaluate_action(self, action)