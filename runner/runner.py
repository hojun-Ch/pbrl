# General
import copy
import os
import itertools
import torch

# from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# for dmcontrol suite
from dm_control import suite

# for buffer
from algo.sac import SAC
# from algo.reward import RewardModel
import wandb
class Runner():
    def __init__(self, args, env, action_dim, obs_dim):
       
        self.device = args.device
       
        self.env = env
        self.env_name = args.env_name
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        
        self.env_step = 0
        self.max_step = args.max_step
        self.episode_length = args.episode_length
        self.learning_starts = args.learning_starts
        self.eval_frequency = args.eval_frequency
        self.eval_num_env = args.eval_num_env
        self.update_frequency = args.update_frequency
        
        self.env_name = args.env_name
        self.domain_name = args.domain_name
        self.task_name = args.task_name
        
        self.unsup_pre_training = args.unsupervised_pre_training
        self.preference_based = args.preference_based
        
        if args.rl_algo == 'sac':
            self.algo = SAC(args, self.action_dim, self.obs_dim)
        
        self.algo.to_device()
        
        self.pbrl = False
        
        if self.preference_based:
            self.pbrl = True
            self.reward_function = RewardModel(args)

        self.run_name = args.domain_name + "_" + args.task_name + "_" + args.exp_name
    
    def flatten_obs(self, obs):
        flattened_obs = []
        
        if self.env_name == 'dmcontrol':
        
            for key in obs.keys():
                if type(obs[key]) == np.ndarray:
                    flattened_obs += list(obs[key])
                else:
                    flattened_obs.append(obs[key])
        
        return flattened_obs
        
    def unsup_pre_train(self):
        return
    
    
    def train(self):
        
        wandb.init(project="pbrl", reinit=True, entity="hojun-chung")
        wandb.run.name = self.run_name
        
        self.algo.train_mode()
        
        # Unsupervised Pre-Training(PEBBLE)
        if self.unsup_pre_training:
            self.unsup_pre_train()
        
        if self.env_name == "dmcontrol":
            
            episode_return = 0
            n_episode = 0
            
            for i in range(self.max_step):
                
                # for evaluation
                if i % self.eval_frequency == 0 and i > 0:
                    eval_return = self.evaluation()
                    wandb.log({
                                'eval_return': sum(eval_return) / len(eval_return)
                            })
                    print(f"eval return:", sum(eval_return) / len(eval_return))
                    
                # start new episode
                if i % self.episode_length == 0:
                    if i > self.learning_starts:
                        print(f"episode[{n_episode}] return {episode_return}")
                        print()
                        wandb.log({
                            'episode_return': episode_return,
                            'temperature':self.algo.log_alpha.item(),
                        })
                        
                    time_step = self.env.reset()
                    obs = np.array(self.flatten_obs(time_step.observation))
                    episode_return = 0
                    Q_losses = []
                    actor_losses = []
                    alpha_losses = []
                    episode_return = 0
                    n_episode += 1
                
                # reward learning
                if self.preference_based:
                    # sample query and give preference
                    
                    # learn reward
                    self.reward_function.learn_reward()
                
                    # bi-level optimization (Meta-Reward-Net)
                    if args.bi_level_optimization:
                        ...
                        # do bi_level optimization
                
                # episode step 
                
                self.algo.eval_mode()
                action_dist = self.algo.actor(torch.from_numpy(obs.astype(np.float32)).to(self.device))
                action = action_dist.sample()
                action = torch.clamp(action, -1, 1)
                action = action.clone().cpu().numpy()
                self.algo.train_mode()
                
                time_step = self.env.step(action)
                
                next_obs = np.array(self.flatten_obs(time_step.observation))
                reward = np.array([time_step.reward])
                episode_return += reward
                reward_estimates = reward
                if self.preference_based:
                    reward_estimates = reward_function(new_obs)
                done = np.array([0])
                
                self.algo.replay_buffer.add(obs, next_obs, action, reward, reward_estimates, done)
                
                obs = next_obs
                
                # update
                if i % self.update_frequency == 0 and i >= self.learning_starts:
                    Q_loss, actor_loss, alpha_loss = self.algo.update()
                    wandb.log({
                            'Q_loss': Q_loss,
                            'actor_loss': actor_loss,
                            'alpha_loss': alpha_loss
                        })
                    Q_losses.append(Q_loss)
                    actor_losses.append(actor_loss)
                    alpha_losses.append(alpha_loss)

        return 

    def evaluation(self):
        
        self.algo.eval_mode()
        
        seed_set = [1312, 1984, 2022, 3071, 4567, 5293, 7777, 6225, 8172, 9977]
        episode_return = [0] * self.eval_num_env
        for i in range(self.eval_num_env):
            seed = seed_set[i]
            random_state = np.random.RandomState(seed)
            eval_env = suite.load(domain_name=self.domain_name, task_name=self.task_name, task_kwargs={'random': random_state})
            
            time_step = eval_env.reset()
            obs = np.array(self.flatten_obs(time_step.observation))
            
            for t in range(self.episode_length):
                action_dist = self.algo.actor(torch.from_numpy(obs.astype(np.float32)).to(self.device))
                action = action_dist.sample()
                action = torch.clamp(action, -1, 1)
                action = action.clone().cpu().numpy()
                
                time_step = eval_env.step(action)
                obs = np.array(self.flatten_obs(time_step.observation))
                
                episode_return[i] += time_step.reward
        
            eval_env.close()
        
        return episode_return
    
    def save_ckpt(self):
        return
    