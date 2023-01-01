import torch
import numpy as np
from .buffers import ReplayBuffer
from models.models import MLP
from torch.distributions import Normal
from torch import distributions as pyd
import torch.nn as nn 
from torch.nn import functional as F
from typing import Iterable
from itertools import zip_longest
import math

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return 

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds=[-5,2]):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = MLP(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

class Q_function(nn.Module):
    
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layer):
        super().__init__()
        
        self.layer = MLP(obs_dim + action_dim, hidden_dim, 1, num_layer)


    def forward(self, obs, action):
        
        obs_action = torch.cat([obs, action], dim=1)
        
        return self.layer(obs_action)
    
class SAC(object):
    
    def __init__(self, args, action_dim, obs_dim):
        
        self.device = args.device
        
        self.action_dim = action_dim,
        self.obs_dim = obs_dim
        
        self.replay_buffer = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device=args.device)
        
        self.actor = Actor(obs_dim, action_dim, args.hidden_dim, args.num_layer)
        
        self.Q1 = Q_function(obs_dim, action_dim, args.hidden_dim, args.num_layer)
        self.Q2 = Q_function(obs_dim, action_dim, args.hidden_dim, args.num_layer)
        
        self.Q1_target = Q_function(obs_dim, action_dim, args.hidden_dim, args.num_layer)
        self.Q2_target = Q_function(obs_dim, action_dim, args.hidden_dim, args.num_layer)
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=args.learning_rate)

        self.log_alpha = torch.tensor(np.log(args.initial_temperature))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_learning_rate)
        self.target_entropy = -action_dim
        
        self.tau = args.critic_tau
        self.gamma = args.discount_factor
        self.batch_size = args.batch_size
        self.learnable_temperature = args.learnable_temperature
        self.critic_target_update_freq = args.critic_target_update_freq
        
        self.preference_based = args.preference_based
        
        self.num_updates = 0
        
    def update(self):
        
        replay_data = self.replay_buffer.sample(self.batch_size)
        
        obs = replay_data.observations
        next_obs = replay_data.next_observations
        actions = replay_data.actions 
        
        # update critic
        with torch.no_grad():
            next_action_dist = self.actor(next_obs)
            
            next_actions = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_actions).sum(-1, keepdim=True)
            
            next_Q_target_1 = self.Q1_target(next_obs, next_actions)
            next_Q_target_2 = self.Q2_target(next_obs, next_actions)
            
            next_Q_target = torch.min(next_Q_target_1, next_Q_target_2)
            
            if self.preference_based:
                y = replay_data.reward_estimates + self.gamma * (next_Q_target - self.log_alpha.exp().detach() * next_log_prob)
            else:
                y = replay_data.rewards + self.gamma * (next_Q_target - self.log_alpha.exp().detach() * next_log_prob)
            
        Q1 = self.Q1(obs, actions)
        Q2 = self.Q2(obs, actions)
        
        self.critic_optimizer.zero_grad()
        Q_loss = F.mse_loss(Q1, y) + F.mse_loss(Q2, y)
        Q_loss_item = Q_loss.item()
        Q_loss.backward()
        self.critic_optimizer.step()
        
        # update actor
        
        current_action_dist = self.actor(obs)
        current_action = current_action_dist.rsample()
        current_log_prob = current_action_dist.log_prob(current_action).sum(-1, keepdim=True)
        
        Q1 = self.Q1(obs, current_action)
        Q2 = self.Q2(obs, current_action)
        Q = torch.min(Q1, Q2)
        
        self.actor_optimizer.zero_grad()
        actor_loss = (self.log_alpha.exp().detach() * current_log_prob - Q).mean()
        actor_loss_item = actor_loss.item()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update ent coeff
        
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.log_alpha.exp() * (- current_log_prob - self.target_entropy).detach()).mean()
            alpha_loss_item = alpha_loss.item()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()


        # update critic target
        if self.num_updates % self.critic_target_update_freq == 0:
            
            for target_param, param in zip_strict(self.Q1_target.parameters(), self.Q1.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
                
            for target_param, param in zip_strict(self.Q2_target.parameters(), self.Q2.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
        
        self.num_updates += 1
        
        return Q_loss_item, actor_loss_item, alpha_loss_item
    
    def to_device(self):
        self.actor.to(self.device)
        self.Q1.to(self.device)
        self.Q2.to(self.device)
        self.Q1_target.to(self.device)
        self.Q2_target.to(self.device)
        self.log_alpha.to(self.device)
    
    def train_mode(self):
        self.actor.train()
        self.Q1.train()
        self.Q2.train()
        self.Q1_target.train()
        self.Q2_target.train()
        self.log_alpha.requires_grad = True
        
    def eval_mode(self):
        self.actor.eval()
        self.Q1.eval()
        self.Q2.eval()
        self.Q1_target.eval()
        self.Q2_target.eval()
        self.log_alpha.requires_grad = False
        
        
    