# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import os
from datetime import datetime
import atari_py
import numpy as np
import torch
import wandb

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
from tqdm import tqdm, trange
import pickle


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')

# Custom arguments I added
DEFUALT_WANDB_ENTITY = 'augmented-frostbite'
parser.add_argument('--wandb-entity', default=DEFUALT_WANDB_ENTITY)
DEFAULT_WANDB_PROJECT = 'initial-experiments'
parser.add_argument('--wandb-project', default=DEFAULT_WANDB_PROJECT)
parser.add_argument('--wandb-omit-watch', action='store_true')
parser.add_argument('--wandb-resume', action='store_true')
DEFAULT_MEMORY_SAVE_FOLDER = r'/misc/vlgscratch4/LakeGroup/guy/'
parser.add_argument('--memory-save-folder', default=DEFAULT_MEMORY_SAVE_FOLDER)

# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

# Handle slurm array ids
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
if array_id is not None:
  args.seed = args.seed + int(array_id)

np.random.seed(args.seed)
# TODO: why not just fix the torch seed to the same one as np?
# torch.manual_seed(np.random.randint(1, 10000))
torch.manual_seed(args.seed)

if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  # torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

memory_save_folder = os.path.join(args.memory_save_folder, args.id)
os.makedirs(memory_save_folder, exist_ok=True)

replay_memory_pickle = f'{args.seed}-replay-memory.pickle'
replay_memory_T_reached = f'{args.seed}-T-reached.txt'


def get_memory_file_path(name, folder=memory_save_folder):
  return os.path.join(folder, name)


# Set up wandb
wandb_name = f'{args.id}-{args.seed}'

if args.wandb_resume:
  api = wandb.Api()

  original_run_name = None
  T_resume = None
  resume_checkpoint = None
  loaded_replay_memory = None

  for existing_run in api.runs(f'{args.wandb_entity}/{args.wandb/project}'):
    if existing_run.config['seed'] == args.seed:
      original_run_name = existing_run.name

      history = existing_run.history(pandas=True, samples=1000)
      T_resume = history['steps'][-1]

      try:
        resume_checkpoint = existing_run.file(f'{wandb_name}-{resume_T}.pth')
        resume_checkpoint.download(replace=True)

      except (AttributeError, wandb.CommError) as e:
        print('Failed to download most recent checkpoint, will not resume')

      if not os.path.exists(get_memory_file_path(replay_memory_T_reached)):
        print('Couldn\'t find replay memory T reached file...')

      with open(get_memory_file_path(replay_memory_T_reached), 'r') as T_file:
        mem_T_reached = int(T_file.read())

      if mem_T_reached != T_resume:
        print(f'Timestep mismatch: wandb has {T_resume}, while memory file has {mem_T_reached}...')

      with open(get_memory_file_path(replay_memory_pickle), 'rb') as pickle_file:
        loaded_replay_memory = pickle.load(pickle_file)

      break

  if original_run_name is None:
    print(f'Failed to find run to resume for seed {args.seed}, running from scratch')

  elif resume_checkpoint is None:
    print(f'Failed to find checkpoint to resume for seed {args.seed}, running from scratch')

  elif loaded_replay_memory is None:
    print('Failed to load replay memory, running from scratch')

  else:
    os.environ['WANDB_RESUME'] = 'must'
    os.environ['WANDB_RUN_ID'] = original_run_name

    args.model = os.path.join('.', resume_checkpoint.name)


wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=wandb_name,
           config=vars(args))




# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Environment
env = Env(args)
env.train()
action_space = env.action_space()


# Agent
dqn = Agent(args, env)
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

if not args.wandb_omit_watch:
  wandb.watch(dqn.online_net)

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)

T_start = 0
if args.wandb_resume and T_resume is not None:
  T_start = T_resume
  mem = loaded_replay_memory

T, done = T_start, True
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, None, None, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

else:
  # Training loop
  dqn.train()
  done = True
  for T in trange(T_start, args.T_max):
    if done:
      state, done = env.reset(), False

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action, reward, done)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        dqn.learn(mem)  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

        dqn.save(wandb.run.dir, f'{wandb_name}-{T}.pth')

        memory_save_folder = os.path.join(args.memory_save_folder, args.id)
        os.makedirs(memory_save_folder, exist_ok=True)

        with open(get_memory_file_path(replay_memory_pickle), 'wb') as pickle_file:
          pickle.dump(mem, pickle_file)

        with open(get_memory_file_path(replay_memory_T_reached), 'w') as T_file:
          T_file.write(str(T))

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

    state = next_state

env.close()
