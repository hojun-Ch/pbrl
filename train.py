# General
import copy
import os
import itertools
# from IPython.display import clear_output
import numpy as np
import torch
# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# for dmcontrol suite
from dm_control import suite
from utils.dmcontrol import dmc_info
import dmc2gym
# for args
from arguments import parser
# for PBRL agent
from runner.runner import Runner

#@title Loading and simulating a `suite` task{vertical-output: true}

def build_env(args):
    if args.env_name == "dmcontrol":
        random_state = np.random.RandomState(args.seed)
        env = suite.load(domain_name=args.domain_name, task_name=args.task_name, task_kwargs={'random': random_state})
    elif args.env_name == "metaworld":
        return
    
    return env

def build_runner(args, env):
    
    if args.env_name == "dmcontrol":
        action_dim = dmc_info[args.domain_name]['action']
        obs_dim = dmc_info[args.domain_name]['observation']
        print("action_dim:", action_dim)
        print("obs_dim:", obs_dim)
        
        runner = Runner(args, env, action_dim, obs_dim)
    
    elif args.env_name == 'metaworld':
        return
    
    return runner

if __name__ == '__main__':
    
    args = parser.parse_args()

    env = build_env(args)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    runner = build_runner(args, env)
        
    # train the agent
    runner.train()
    


