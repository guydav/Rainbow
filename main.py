# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime, timedelta
import os
import psutil
import pickle
import shutil
import time
import subprocess
import sys
import string

import atari_py
import numpy as np
import torch
from tqdm import tqdm, trange
import wandb
# from guppy import hpy

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
parser.add_argument('--memory-save-interval', type=int, default=None, help='How often to save the memory, defaults to the evaluation interval')
parser.add_argument('--use-native-pickle-serialization', action='store_true', help='Use native pickle saving rather than torch.save()')

# Arguments for the augmented representations
parser.add_argument('--add-masks', action='store_true', help='Add masks for each semantic object types')
parser.add_argument('--maskers', default=None, help='Select specific maskers to use')
parser.add_argument('--use-numpy-masker', action='store_true', help='Use the previous, much slower numpy-based masker')
parser.add_argument('--omit-pixels', action='store_true', help='Omit the raw pixels from the environment')
parser.add_argument('--zero-out-masks-test', action='store_true', help='Test zeroing out particular indices')
parser.add_argument('--zero-out-mask-indices', default=None, help='Which indices to zero out each time')

# Arguments to give it a soft time cap that will help it not fail
parser.add_argument('--soft-time-cap', help='Format: <DD>:HH:MM, stop after some soft cap such that the saving the memory does not fail')

# Debugging-related arguments
parser.add_argument('--debug-heap', action='store_true')
parser.add_argument('--heap-interval', default=1e4)
parser.add_argument('--heap-debug-file', default=None)

# Setup
args = parser.parse_args()

# Handle slurm array ids
array_id = os.getenv('SLURM_ARRAY_TASK_ID')
if array_id is not None:
  args.seed = args.seed + int(array_id)

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
if args.memory_save_interval is None or args.memory_save_interval == 0:
  args.memory_save_interval = args.evaluation_interval

REPLAY_MEMORY_PICKLE = 'replay_memory_pickle'
REPLAY_MEMORY_BZIP = 'replay_memory_bzip'
REPLAY_MEMORY_FINAL = 'replay_memory_final'
REPLAY_MEMORY_T_REACHED = 'replay_memory_T_reached'

REPLAY_MEMORY_PICKLE_TEMPLATES = {
  REPLAY_MEMORY_PICKLE: f'{args.seed}-replay-memory{{rand}}.pickle',
  REPLAY_MEMORY_BZIP: f'{args.seed}-replay-memory{{rand}}.pickle.bz2',
  REPLAY_MEMORY_FINAL: f'{args.seed}-replay-memory.pickle.bz2.final',
  REPLAY_MEMORY_T_REACHED: f'{args.seed}-T-reached.txt'
}


def get_memory_file_path(key, format_args=None, folder=memory_save_folder, template_dict=REPLAY_MEMORY_PICKLE_TEMPLATES):
  if key not in template_dict:
    raise ValueError(f'Received key {key}, expected one of {list(template_dict.keys())}')

  if format_args is None:
    format_args = {}

  template = template_dict[key]
  template_keys = [tup[1] for tup in string.Formatter().parse(template) if tup[1] is not None]
  args = {key: '' for key in template_keys}
  args.update(format_args)

  return os.path.join(folder, template.format(**args))


if args.debug_heap:
  process = psutil.Process()
  # heap = hpy()
  # heap.setref()

  heap_debug_log_path = args.heap_debug_file
  if heap_debug_log_path is None:
    heap_debug_log_path = os.path.join(results_dir, 'heap_debug.log')

general_debug_log_path = os.path.join(results_dir, 'debug.log')


def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        print(f'{method.__name__} took {end_time - start_time:.2f} s')

        return result
    return timed

# Simple ISO 8601 timestamped logger
def format_log_message(s):
  return f'[{str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))}]: {s}'


def log_to_file(path, s):
  with open(path, 'a') as log_file:
    log_file.write(f'{format_log_message(s)}\n')


def log(s, write_to_file=True, path=general_debug_log_path):
  msg = format_log_message(s)
  print(msg)

  if write_to_file:
    log_to_file(path, msg)


if args.soft_time_cap is not None:
  start_time = datetime.now()
  split_time_cap = [int(x.strip()) for x in args.soft_time_cap.split(':')]
  if len(split_time_cap) < 2 or len(split_time_cap) > 3:
    raise ValueError(f'Expected time cap to have the format <DD>:HH:MM, got {args.soft_time_cap}')

  if len(split_time_cap) == 2:
    split_time_cap.insert(0, 0)

  end_time = start_time + timedelta(days=split_time_cap[0], hours=split_time_cap[1], minutes=split_time_cap[2])
  log(f'Received a soft cap of {args.soft_time_cap}, will terminate at {end_time}')


@timeit
def load_memory(use_bz2=True, use_native_pickle_serialization=False):
  global heap_debug_log_path, process

  pickle_full_path = get_memory_file_path(REPLAY_MEMORY_PICKLE)
  zipped_full_path = get_memory_file_path(REPLAY_MEMORY_BZIP)
  final_full_path = get_memory_file_path(REPLAY_MEMORY_FINAL)

  if use_native_pickle_serialization:
    if use_bz2:
      with bz2.open(final_full_path, 'rb') as zipped_pickle_file:
        return pickle.load(zipped_pickle_file)

    else:
      with open(pickle_full_path, 'rb') as regular_pickle_file:
        return pickle.load(regular_pickle_file)

  else:
    # Copy "final" file to zipped name
    shutil.copy(final_full_path, zipped_full_path)

    # Unzip
    subprocess_args = ['bzip2', '-f', '-d', zipped_full_path]
    unzip_process = subprocess.Popen(' '.join(subprocess_args), shell=True,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = unzip_process.communicate()
    log(f'Memory load unzip popen return code: {unzip_process.returncode}')
    if out is not None and len(out) > 0:
      log(f'Popen stdout: {out}')
    if err is not None and len(err) > 0:
      log(f'Popen stderr: {err}')

    # Load memory
    with open(pickle_full_path, 'rb') as regular_pickle_file:
      memory = torch.load(regular_pickle_file)

    # Remove the unzipped file
    os.remove(pickle_full_path)

    return memory

@timeit
def save_memory(memory, T_reached, use_native_pickle_serialization=False):
  global heap_debug_log_path, process

  save_process = None

  save_rand = np.random.randint(0, 1000)

  replay_memory_T_reached_path = get_memory_file_path(REPLAY_MEMORY_T_REACHED)
  pickle_full_path = get_memory_file_path(REPLAY_MEMORY_PICKLE, dict(rand=save_rand))
  zipped_full_path = get_memory_file_path(REPLAY_MEMORY_BZIP, dict(rand=save_rand))
  final_full_path = get_memory_file_path(REPLAY_MEMORY_BZIP)

  if use_native_pickle_serialization:
    with bz2.open(zipped_full_path, 'wb') as zipped_pickle_file:
      process_mem = process.memory_info().rss
      log_to_file(heap_debug_log_path,
                  f'OS-level memory usage after file open: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

      pickle.dump(memory, zipped_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    process_mem = process.memory_info().rss
    log_to_file(heap_debug_log_path,
                f'OS-level memory usage after save, before move: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

    # Switch to copying and moving separately to mitigate the effect of instant shutdown while writing
    shutil.move(zipped_full_path, final_full_path)

    process_mem = process.memory_info().rss
    log_to_file(heap_debug_log_path,
                f'OS-level memory usage after move: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

  else:
    with open(pickle_full_path, 'wb') as pickle_file:
      torch.save(memory, pickle_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    process_mem = process.memory_info().rss
    log_to_file(heap_debug_log_path,
                f'OS-level memory usage after save, before bzip: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

    subprocess_args = ['bzip2', '-f', '-z', '--fast', pickle_full_path, '&&', 'mv', zipped_full_path, final_full_path]
    save_process = subprocess.Popen(' '.join(subprocess_args), shell=True,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    process_mem = process.memory_info().rss
    log_to_file(heap_debug_log_path,
                f'OS-level memory usage after starting bzip: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

  with open(replay_memory_T_reached_path, 'w') as memory_T_file:
    memory_T_file.write(str(T_reached))

  process_mem = process.memory_info().rss
  log_to_file(heap_debug_log_path,
              f'OS-level memory usage after T-reached file: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

  return save_process


def evaluate_and_log_model(t, dqn):
  log(f'Starting to test at T = {t}')
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, t, dqn, val_mem, metrics, results_dir)  # Test
  log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
  dqn.train()  # Set DQN (online network) back to training mode

  log('Before model save')
  dqn.save(wandb.run.dir, f'{wandb_name}-{t}.pth')
  if args.debug_heap and t % args.heap_interval == 0:
    process_mem = process.memory_info().rss
    log_to_file(heap_debug_log_path,
                f'OS-level memory usage after saving model: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')
  log('After model save')


def wait_and_log_popen(popen_handle):
  log('Waiting for previous popen to end')
  log('About to call popen.commumicate')
  out, err = popen_handle.communicate()
  result = popen_handle.returncode
  log(f'Popen return code: {result}')

  if out is not None and len(out) > 0:
    log(f'Popen stdout: {out}')
  if err is not None and len(err) > 0:
    log(f'Popen stderr: {err}')

  popen_handle.terminate()
  return None


def save_and_log_memory(t, memory, popen_handle):
  if popen_handle is not None:
    log('WARNING: previous save has not finished. Waiting for it to finish...')
    wait_and_log_popen(popen_handle)

  log('Before memory save')
  save_process = save_memory(memory, t, use_native_pickle_serialization=args.use_native_pickle_serialization)
  if args.debug_heap and t % args.heap_interval == 0:
    process_mem = process.memory_info().rss
    log_to_file(heap_debug_log_path,
                f'OS-level memory usage after saving memory: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

  log('After memory save')
  return save_process


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

      replay_memory_T_reached_path = get_memory_file_path(REPLAY_MEMORY_T_REACHED)

      if not os.path.exists(replay_memory_T_reached_path):
        print('Couldn\'t find replay memory T reached file...')

      with open(replay_memory_T_reached_path, 'r') as T_file:
        T_memory = int(T_file.read())

      # Take the min to resume from the earlier of the two potential points
      T_resume = min(T_checkpoint, T_memory)

      if T_memory != T_checkpoint:
        print(f'Timestep mismatch: wandb has {T_checkpoint}, while memory file has {T_memory}. Going with {T_resume}...')

      if T_resume >= args.T_max:
        print(f'T_resume ({T_resume}) is greater than or equal to T_max ({args.T_max}), nothing to do here...')
        sys.exit(0)

      # Now that we now that T_resume is, we can load from there.
      try:
        resume_checkpoint = existing_run.file(f'{wandb_name}-{T_resume}.pth')
        resume_checkpoint.download(replace=True)
      except (AttributeError, wandb.CommError) as e:
        print('Failed to download most recent checkpoint, will not resume')

      if not args.evaluate:
        loaded_replay_memory = load_memory(use_native_pickle_serialization=args.use_native_pickle_serialization)

  if original_run_id is None:
    print(f'Failed to find run to resume for seed {args.seed}, running from scratch')

  elif T_resume is None:
    print(f'Failed to find the correct resume timestamp for seed {args.seed}, running from scratch')

  elif resume_checkpoint is None:
    print(f'Failed to find checkpoint to resume for seed {args.seed}, running from scratch')

  elif loaded_replay_memory is None and not args.evaluate:
    print(f'Failed to load replay memory for seed {args.seed}, running from scratch')

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


if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode

  if args.zero_out_masks_test:
    if args.add_masks is False:
      raise ValueError('Cannot test zeroing out masks without adding masks (args.zero_out_mask_test is True and args.add_masks is False)')

    if args.zero_out_mask_indices is not None:
      indices = [x.strip() for x in args.zero_out_mask_indices.split(',')]
      indices = [[i for i in x] for x in indices]

    else:
      indices = [[x] for x in range(len(env.masker.masker_definitions))]

    columns = ['Zero Indices'] + [f'Reward #{i + 1}' for i in range(args.evaluation_episodes)] +\
              ['Reward Mean', 'Reward Std']
    table = wandb.Table(columns=columns)

    print(f'Evaluating the following index sets: {indices}')
    for index_set in indices:
      print(f'Evaluating masking out the indices {index_set}')
      args.zero_mask_indices = index_set
      rewards, _ = test(args, T_start, dqn, val_mem, metrics, results_dir,
                               evaluate=True, return_individual_values=True)  # Test

      table.add_data(index_set, *rewards, np.mean(rewards), np.std(rewards))

    wandb.log({'Zero Mask Evaluation Results': table})

  else:
    avg_reward, avg_Q = test(args, T_start, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

else:
  popen = None

  # Training loop
  dqn.train()
  done = True

  for T in trange(T_start + 1, args.T_max + 1):

    if args.soft_time_cap is not None and end_time < datetime.now():
      log(f'Hit soft time cap, evaluating, saving, and exiting')

      if popen is not None:
        popen = wait_and_log_popen(popen)

      evaluate_and_log_model(T, dqn)
      popen = save_and_log_memory(T, mem, popen)
      popen = wait_and_log_popen(popen)
      break

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
        process_mem = process.memory_info().rss
        log_to_file(heap_debug_log_path, f'After {T} steps, replay buffer size is {replay_size}, {process_mem / 1024.0 / replay_size:.3f} KB/transition')
        log_to_file(heap_debug_log_path,
                    f'OS-level memory usage after training: {process_mem} bytes = {process_mem / 1024.0 / 1024:.3f} MB.')

      if T % args.evaluation_interval == 0:
        evaluate_and_log_model(T, dqn)

      if T % args.memory_save_interval == 0:
        popen = save_and_log_memory(T, mem, popen)

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

      # Check the potential bzip process
      if popen is not None:
        result = popen.poll()
        if result is not None:
          log_to_file(heap_debug_log_path, f'Popen return code: {result}')

          try:
            log_to_file(heap_debug_log_path, 'About to call popen.commumicate')
            out, err = popen.communicate(timeout=10)
            if out is not None and len(out) > 0:
              log_to_file(heap_debug_log_path, f'Popen stdout: {out}')
            if err is not None and len(err) > 0:
              log_to_file(heap_debug_log_path, f'Popen stderr: {err}')
          except subprocess.TimeoutExpired:
            pass

          popen.terminate()
          popen = None

      state = next_state

env.close()
