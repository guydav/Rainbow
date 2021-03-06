# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch


FULL_FRAME_SHAPE = (210, 160)


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
    self.full_observation_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _to_tensor(self, frame, dtype=torch.float32):
    return torch.tensor(frame, dtype=dtype, device=self.device)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))
      self.full_observation_buffer.append(torch.zeros((*FULL_FRAME_SHAPE, 3), device=self.device))

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
    observation = self._get_state()
    self.state_buffer.append(observation)
    full_color_state = self._to_tensor(self.ale.getScreenRGB())
    self.full_observation_buffer.append(full_color_state)

    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    grayscale_frame_buffer = torch.zeros((2, *FULL_FRAME_SHAPE), device=self.device)
    full_color_frame_buffer = torch.zeros((2, *FULL_FRAME_SHAPE, 3), device=self.device)

    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
        grayscale_frame_buffer[0] = self._to_tensor(self.ale.getScreenGrayscale().squeeze())
        full_color_frame_buffer[0] = self._to_tensor(self.ale.getScreenRGB())
      elif t == 3:
        frame_buffer[1] = self._get_state()
        grayscale_frame_buffer[1] = self._to_tensor(self.ale.getScreenGrayscale().squeeze())
        full_color_frame_buffer[1] = self._to_tensor(self.ale.getScreenRGB())

      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)

    indices = grayscale_frame_buffer.max(0)[1]
    resized_indices = indices.unsqueeze(0).unsqueeze(3)
    resized_shape = list(resized_indices.shape)
    resized_shape[3] = 3
    full_color_observation = torch.squeeze(torch.gather(full_color_frame_buffer,
                                                        0, resized_indices.expand(*resized_shape)))
    self.full_observation_buffer.append(full_color_observation)

    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

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
