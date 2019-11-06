# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
import numpy as np


FULL_SHAPE = (210, 160)
SMALL_SHAPE = (84, 84)


class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode
    self.state_depth = args.state_depth

  def _resize(self, frame):
    return cv2.resize(frame, SMALL_SHAPE, interpolation=cv2.INTER_LINEAR)

  def _to_tensor(self, frame, dtype=torch.float32):
    return torch.tensor(frame, dtype=dtype, device=self.device)

  def _reset_buffer(self):
    for _ in range(self.window * self.state_depth):
      self.state_buffer.append(torch.zeros(1, *SMALL_SHAPE, device=self.device))

  def _prepare_state(self, observation, full_color_state):
    observation = self._to_tensor(self._resize(observation)).div_(255)
    augmentation = self._augment_state(full_color_state)
    return torch.stack((observation, *augmentation))

  def _augment_state(self, full_color_state):
    return list()

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    state = self._prepare_state(self.ale.getScreenGrayscale(), self.ale.getScreenRGB())

    self.state_buffer.append(state)
    self.lives = self.ale.lives()
    return torch.cat(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = np.zeros((2, *FULL_SHAPE))
    full_color_frame_buffer = np.zeros((2, *FULL_SHAPE, 3))
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t >= 2:
        frame_buffer[t - 2] = self.ale.getScreenGrayscale().squeeze()
        full_color_frame_buffer[t - 2] = self.ale.getScreenRGB()
      done = self.ale.game_over()
      if done:
        break

    observation = frame_buffer.max(0)
    indices = frame_buffer.argmax(0)
    # resized_indices = indices.unsqueeze(0).unsqueeze(3)
    # resized_shape = list(resized_indices.shape)
    # resized_shape[3] = 3
    # full_color_observation = torch.squeeze(torch.gather(torch.tensor(full_color_frame_buffer, dtype=torch.float32),
    #                                                     0, resized_indices.expand(*resized_shape)))
    full_color_observation = full_color_frame_buffer[indices, np.arange(FULL_SHAPE[0])[:, None], np.arange(FULL_SHAPE[1])]
    # TODO: avoid the call to .numpy() here if I rewrite the masker to be native in torch
    state = self._prepare_state(observation, full_color_observation)
    self.state_buffer.append(state)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    # TODO: to torch and correct device
    return torch.cat(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()


class MaskerEnv(Env):
  def __init__(self, args, maskers):
    super(MaskerEnv, self).__init__(args, state_size=1 + len(maskers))
    self.maskers = maskers

  def _augment_state(self, full_color_state):
    return [self._to_tensor(self._resize(masker(full_color_state)), dtype=torch.float32)
            for masker in self.maskers]






