# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
from masker import ALL_MASKERS, FULL_FRAME_SHAPE, SMALL_FRAME_SHAPE, ColorFilterMasker, TorchMasker


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
    self.state_depth = args.state_depth
    self.omit_pixels = args.omit_pixels

  def _resize(self, frame):
    if isinstance(frame, torch.Tensor):
      # TODO: validate which mode should this be
      return torch.nn.functional.interpolate(frame, SMALL_FRAME_SHAPE, mode='bilinear')
    else:
      return cv2.resize(frame, SMALL_FRAME_SHAPE, interpolation=cv2.INTER_LINEAR)

  def _to_tensor(self, frame, dtype=torch.float32):
    return torch.tensor(frame, dtype=dtype, device=self.device)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros((self.state_depth, *SMALL_FRAME_SHAPE), device=self.device))
      self.full_observation_buffer.append(torch.zeros((*FULL_FRAME_SHAPE, 3), device=self.device))

  def _prepare_state(self, observation, full_color_state, augmentation=None):
    observation = self._resize(observation.view(1, 1, *observation.shape)).div_(255)
    if augmentation is None:
      augmentation = self._augment_state(full_color_state)

    if isinstance(augmentation, torch.Tensor):
      if self.omit_pixels:
        return augmentation.squeeze(0)
      else:
        return torch.cat((observation.squeeze(0), augmentation.squeeze(0)))
    else:
      if self.omit_pixels:
        return torch.stack(augmentation)
      else:
        return torch.stack((observation.squeeze(), *augmentation))

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
    full_color_state = self._to_tensor(self.ale.getScreenRGB())
    state = self._prepare_state(self._to_tensor(self.ale.getScreenGrayscale().squeeze()),
                                full_color_state)

    self.state_buffer.append(state)
    self.full_observation_buffer.append(full_color_state)
    self.lives = self.ale.lives()
    return torch.cat(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    # frame_buffer = np.zeros((2, *FULL_FRAME_SHAPE))
    # full_color_frame_buffer = np.zeros((2, *FULL_FRAME_SHAPE, 3))
    frame_buffer = torch.zeros((2, *FULL_FRAME_SHAPE), device=self.device)
    full_color_frame_buffer = torch.zeros((2, *FULL_FRAME_SHAPE, 3), device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t >= 2:
        frame_buffer[t - 2] = self._to_tensor(self.ale.getScreenGrayscale().squeeze())
        full_color_frame_buffer[t - 2] = self._to_tensor(self.ale.getScreenRGB())
      done = self.ale.game_over()
      if done:
        break

    # observation = frame_buffer.max(0)
    # indices = frame_buffer.argmax(0)
    observation, indices = frame_buffer.max(0)
    resized_indices = indices.unsqueeze(0).unsqueeze(3)
    resized_shape = list(resized_indices.shape)
    resized_shape[3] = 3
    full_color_observation = torch.squeeze(torch.gather(full_color_frame_buffer,
                                                        0, resized_indices.expand(*resized_shape)))
    # full_color_observation = full_color_frame_buffer[indices, np.arange(FULL_FRAME_SHAPE[0])[:, None], np.arange(FULL_FRAME_SHAPE[1])]

    state = self._prepare_state(observation, full_color_observation)
    self.state_buffer.append(state)
    self.full_observation_buffer.append(full_color_observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
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
    super(MaskerEnv, self).__init__(args)
    self.maskers = maskers

  def _augment_state(self, full_color_state):
    return [self._to_tensor(self._resize(masker(full_color_state)))
            for masker in self.maskers]


class TorchMaskerEnv(Env):
  def __init__(self, args, masker):
    super(TorchMaskerEnv, self).__init__(args)
    self.masker = masker

  def _augment_state(self, full_color_state):
    return self._resize(self.masker(full_color_state).unsqueeze(0))


def make_env(args):
  args.state_depth = int(not args.omit_pixels)

  if args.add_masks:
    if args.maskers is None:
      masker_defs = list(ALL_MASKERS.values())
    else:
      masker_defs = [ALL_MASKERS[name.strip().lower()] for name in args.maskers.split(',')]

    if args.custom_mask_grouping is None or len(args.custom_mask_grouping) == 0:
      args.state_depth += len(masker_defs)
    else:
      args.state_depth += len(args.custom_mask_grouping)

    if 'zero_mask_indices' not in args:
      args.zero_mask_indices = None

    return TorchMaskerEnv(args, TorchMasker(masker_defs, args.device, args.zero_mask_indices,
                                            args.custom_mask_grouping))
  else:
    return Env(args)





