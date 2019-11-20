# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import psutil
import pickle
import shutil
import time

import atari_py
import numpy as np
import torch
from tqdm import tqdm, trange
import wandb
from guppy import hpy

from agent import Agent
from env import make_env
from memory import ReplayMemory
from test import test


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
parser.add_argument('--save-evaluation-gifs', action='store_true', help='Save GIFs of evaluation episodes')
parser.add_argument('--evaluation-gif-folder', default=None, help='Folder to save evaluation GIFs in')
parser.add_argument('--save-evaluation-states', action='store_true', help='Save the states of evaluation episodes')
parser.add_argument('--evaluation-state-folder', default=None, help='Folder to save evaluation state in')

# Custom arguments I added

SCRATCH_FOLDER = r'/misc/vlgscratch4/LakeGroup/guy/'

DEFUALT_WANDB_ENTITY = 'augmented-frostbite'
parser.add_argument('--wandb-entity', default=DEFUALT_WANDB_ENTITY)
DEFAULT_WANDB_PROJECT = 'initial-experiments'
parser.add_argument('--wandb-project', default=DEFAULT_WANDB_PROJECT)
DEFAULT_WANDB_DIR = SCRATCH_FOLDER  # wandb creates its own folder inside
parser.add_argument('--wandb-dir', default=DEFAULT_WANDB_DIR)
parser.add_argument('--wandb-omit-watch', action='store_true')
parser.add_argument('--wandb-resume', action='store_true')
DEFAULT_MEMORY_SAVE_FOLDER = os.path.join(SCRATCH_FOLDER, 'rainbow_memory')
parser.add_argument('--memory-save-folder', default=DEFAULT_MEMORY_SAVE_FOLDER)

# Arguments for the augmented representations
parser.add_argument('--add-masks', action='store_true')
parser.add_argument('--maskers', default=None)
parser.add_argument('--use-numpy-masker', action='store_true')
parser.add_argument('--debug-heap', action='store_true')
parser.add_argument('--heap-interval', default=1e4)
parser.add_argument('--heap-debug-file', default=None)

# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', f'{args.id}-{args.seed}')
os.makedirs(results_dir, exist_ok=True)

if args.evaluation_gif_folder is None:
  args.evaluation_gif_folder = os.path.join(results_dir, 'evaluation', 'gifs')

if args.save_evaluation_gifs:
  os.makedirs(args.evaluation_gif_folder, exist_ok=True)

if args.evaluation_state_folder is None:
  args.evaluation_state_folder = os.path.join(results_dir, 'evaluation', 'states')

if args.save_evaluation_states:
  os.makedirs(args.evaluation_state_folder, exist_ok=True)

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
replay_memory_pickle_bz2 = f'{args.seed}-replay-memory.pickle.bz2'
replay_memory_pickle_bz2_temp = f'{args.seed}-replay-memory.pickle.bz2.temp'
replay_memory_T_reached = f'{args.seed}-T-reached.txt'


def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        print(f'{method.__name__} took {end_time - start_time:.2f} s')

        return result
    return timed


def get_memory_file_path(name, folder=memory_save_folder):
  return os.path.join(folder, name)


@timeit
def load_memory(use_bz2=True):
  global replay_memory_pickle_bz2, replay_memory_pickle

  if use_bz2:
    with bz2.open(get_memory_file_path(replay_memory_pickle_bz2), 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)

  else:
    with open(get_memory_file_path(replay_memory_pickle), 'rb') as regular_pickle_file:
      return pickle.load(regular_pickle_file)


@timeit
def save_memory(memory, T_reached):
  global replay_memory_pickle_bz2, replay_memory_pickle_bz2_temp, replay_memory_T_reached

  with bz2.open(get_memory_file_path(replay_memory_pickle_bz2_temp), 'wb') as zipped_pickle_file:
    pickle.dump(memory, zipped_pickle_file)

  # Switch to copying and moving separately to mitigate the effect of instane shutdown while writing
  shutil.move(get_memory_file_path(replay_memory_pickle_bz2_temp), get_memory_file_path(replay_memory_pickle_bz2))

  with open(get_memory_file_path(replay_memory_T_reached), 'w') as memory_T_file:
    memory_T_file.write(str(T_reached))


# Set up wandb
wandb_name = f'{args.id}-{args.seed}'

for wandb_key in ('WANDB_RESUME', 'WANDB_RUN_ID'):
  if wandb_key in os.environ:
    del os.environ[wandb_key]


if args.wandb_resume:
  api = wandb.Api()

  original_run_id = None
  T_resume = None
  resume_checkpoint = None
  loaded_replay_memory = None

  existing_runs = api.runs(f'{args.wandb_entity}/{args.wandb_project}', {'$and': [{'config.id': str(args.id)},
                                                                                  {'config.seed': int(args.seed)}]})

  if len(existing_runs) > 1:
    raise ValueError(f'Found more than one matching run to id {args.id} and seed {args.seed}: {[r.id for r in existing_runs]}. Aborting... ')

  elif len(existing_runs) == 1:
    existing_run = existing_runs[0]
    original_run_id = existing_run.id

    history = existing_run.history(pandas=True, samples=1000)

    # Verify there's actually a run to resume
    if len(history) > 0:
      T_checkpoint = int(history['steps'].iat[-1])

      if not os.path.exists(get_memory_file_path(replay_memory_T_reached)):
        print('Couldn\'t find replay memory T reached file...')

      with open(get_memory_file_path(replay_memory_T_reached), 'r') as T_file:
        T_memory = int(T_file.read())

      # Take the min to resume from the earlier of the two potential points
      T_resume = min(T_checkpoint, T_memory)

      if T_memory != T_checkpoint:
        print(f'Timestep mismatch: wandb has {T_checkpoint}, while memory file has {T_memory}. Going with {T_resume}...')

      # Now that we now that T_resume is, we can load from there.
      try:
        resume_checkpoint = existing_run.file(f'{wandb_name}-{T_resume}.pth')
        resume_checkpoint.download(replace=True)
      except (AttributeError, wandb.CommError) as e:
        print('Failed to download most recent checkpoint, will not resume')

      # temporary condition to handle the non-zipped, old pickle files
      if os.path.exists(get_memory_file_path(replay_memory_pickle)):
        loaded_replay_memory = load_memory(use_bz2=False)
        save_memory(loaded_replay_memory, T_memory)
        os.remove(get_memory_file_path(replay_memory_pickle))

      else:
        loaded_replay_memory = load_memory()

  if original_run_id is None:
    print(f'Failed to find run to resume for seed {args.seed}, running from scratch')

  elif T_resume is None:
    print(f'Failed to find the correct resume timestamp for seed {args.seed}, running from scratch')

  elif resume_checkpoint is None:
    print(f'Failed to find checkpoint to resume for seed {args.seed}, running from scratch')

  elif loaded_replay_memory is None:
    print('Failed to load replay memory for seed {args.seed}, running from scratch')

  else:
    os.environ['WANDB_RESUME'] = 'must'
    os.environ['WANDB_RUN_ID'] = original_run_id

    args.model = resume_checkpoint.name

for key in os.environ:
  if 'WANDB' in key:
    print(key, os.environ[key])

wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=wandb_name,
           dir=args.wandb_dir, config=vars(args))
wandb.save(os.path.join(wandb.run.dir, '*.pth'))

memory_save_folder = os.path.join(args.memory_save_folder, args.id)
os.makedirs(memory_save_folder, exist_ok=True)


# Simple ISO 8601 timestamped logger
def log_message(s):
  return f'[{str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))}]: {s}'


def log(s):
  print(log_message(s))


def log_to_file(path, s):
  with open(path, 'a') as log_file:
    log_file.write(f'{log_message(s)}\n')


# Augmented representations and Environments
env = make_env(args)
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
done = True
for _ in range(args.evaluation_size):
  if done:
    state, done = env.reset(), False

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, None, None, done)
  state = next_state

T_start = 0
if args.wandb_resume and T_resume is not None:
  T_start = T_resume
  mem = loaded_replay_memory

if args.debug_heap:
  process = psutil.Process()
  heap = hpy()
  heap.setref()

  heap_debug_path = args.heap_debug_file
  if heap_debug_path is None:
    heap_debug_path = os.path.join(results_dir, 'heap_debug.log')

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, T_start, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

else:
  # Training loop
  dqn.train()
  done = True
  for T in trange(T_start + 1, args.T_max + 1):
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

      if args.debug_heap and T % args.heap_interval == 0:
        replay_size = mem.capacity if mem.transitions.full else mem.transitions.index
        log_to_file(heap_debug_path, f'After {T} steps, replay buffer size is {replay_size}:')
        # allocated = torch.cuda.memory_allocated(args.device)
        # log_to_file(heap_debug_path,
        #             f'CUDA memory allocated after training: {allocated} bytes = {allocated / 1024.0 / 1024:.3f} MB.')
        # cached = torch.cuda.memory_cached(args.device)
        # log_to_file(heap_debug_path,
        #             f'CUDA memory cached after training: {cached} bytes = {cached / 1024.0 / 1024:.3f} MB.')
        process_mem = process.memory_info().rss
        log_to_file(heap_debug_path, f'OS-level memory usage after training: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')
        log_to_file(heap_debug_path, 'Heap after training:')
        log_to_file(heap_debug_path, heap.heap())
        heap.setref()

      if T % args.evaluation_interval == 0:
        log(f'Starting to test at T = {T}')
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

        if args.debug_heap and T % args.heap_interval == 0:
          log_to_file(heap_debug_path, f'After {T} steps:')
          # allocated = torch.cuda.memory_allocated(args.device)
          # log_to_file(heap_debug_path,
          #             f'CUDA memory allocated after testing: {allocated} bytes = {allocated / 1024.0 / 1024:.3f} MB.')
          # cached = torch.cuda.memory_cached(args.device)
          # log_to_file(heap_debug_path,
          #             f'CUDA memory cached after testing: {cached} bytes = {cached / 1024.0 / 1024:.3f} MB.')
          process_mem = process.memory_info().rss
          log_to_file(heap_debug_path,
                      f'OS-level memory usage after testing: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')
          log_to_file(heap_debug_path, 'Heap after testing:')
          log_to_file(heap_debug_path, heap.heap())
          heap.setref()

        log('Before model save')

        dqn.save(wandb.run.dir, f'{wandb_name}-{T}.pth')
        if args.debug_heap and T % args.heap_interval == 0:
          process_mem = process.memory_info().rss
          log_to_file(heap_debug_path,
                      f'OS-level memory usage after saving model: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

        log('Before memory save')
        save_memory(mem, T)
        if args.debug_heap and T % args.heap_interval == 0:
          process_mem = process.memory_info().rss
          log_to_file(heap_debug_path,
                      f'OS-level memory usage after saving memory: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

        log('After both saves')

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

    state = next_state

env.close()
