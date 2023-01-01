import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='pedsim')

parser.add_argument(
    '--device',
    type=str,
    default="cuda:9",
    help='device (cuda or cpu)'
)

## Environment setting
parser.add_argument(
    '--env_name',
    type=str,
    default='dmcontrol',
    help='env name (dmcontrol or metaworld'
)

parser.add_argument(
    '--domain_name',
    type=str,
    default='cheetah',
    help='domain name for each env'
)

parser.add_argument(
    '--task_name',
    type=str,
    default='run',
    help='task name for each domain'
)

parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='random seed'
)

parser.add_argument(
    '--exp_name',
    type=str,
    default='SAC-demo',
    help='exp name for logging'
)
## parameters for training
parser.add_argument(
    '--max_step',
    type=int,
    default=1000000,
    help='task name for each domain'
)

parser.add_argument(
    '--unsupervised_pre_training',
    type=str2bool,
    default=False,
    help='task name for each domain'
)

parser.add_argument(
    '--episode_length',
    type=int,
    default=1000,
    help='single episode length'
)

parser.add_argument(
    '--eval_frequency',
    type=int,
    default=10000,
    help='evalutation frequency'
)

parser.add_argument(
    '--eval_num_env',
    type=int,
    default=10,
    help='num of env for evaluation (<10)'
)

## parameters for RL algorithm
parser.add_argument(
    '--rl_algo',
    type=str,
    default='sac',
    help='reinforcement learning algorithm'
)
parser.add_argument(
    '--buffer_size',
    type=int,
    default=1000000,
    help='buffer size'
)
parser.add_argument(
    '--hidden_dim',
    type=int,
    default=1024,
    help='hidden dimension for MLP'
)
parser.add_argument(
    '--num_layer',
    type=int,
    default=2,
    help='num of layer for MLP'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.001
)
parser.add_argument(
    '--alpha_learning_rate',
    type=float,
    default=0.001
)
parser.add_argument(
    '--discount_factor',
    type=float,
    default=0.99
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='batch size for rl training'
)
## parameters for preference and reward learning
parser.add_argument(
    '--preference_based',
    type=str2bool,
    default=False,
    help='do preference based reinforcement learning'
)
parser.add_argument(
    '--bi_level_optimization',
    type=str2bool,
    default=False,
    help='do bi-level optimization by pseudo Q updating'
)

## parameters for SAC
parser.add_argument(
    '--update_frequency',
    type=int,
    default=1,
    help='update policy per (this value) env step'
)
parser.add_argument(
    '--learning_starts',
    type=int,
    default=5000,
    help='start learning after this value env step'
)
parser.add_argument(
    '--critic_target_update_freq',
    type=int,
    default=2,
    help='target critic update frequency'
)
parser.add_argument(
    '--initial_temperature',
    type=float,
    default=0.1,
    help='initial temperature'
)
parser.add_argument(
    '--learnable_temperature',
    type=str2bool,
    default=True,
    help='learn entropy coeff'
)
parser.add_argument(
    '--critic_tau',
    type=float,
    default=0.005,
    help='critic ema'
)