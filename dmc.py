from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
import torch
from PIL import Image
from numpy.linalg import norm

import cmc_model
import context_changers
import datasets
import utils

from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_env._environment import TimeStep
from hydra.utils import to_absolute_path



class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]

        if self.to_lab:
            self._obs_spec = specs.Array(shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                dtype=np.float,
                name='observation')
        else:
            self._obs_spec = specs.BoundedArray(shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                                dtype=np.uint8,
                                                minimum=0,
                                                maximum=255,
                                                name='observation')


    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]

        if self.to_lab:
            pixels = utils.rgb_to_lab(pixels)
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RewardComputeStackWrapper(dm_env.Environment):

    def __init__(self, env, encoder, im_w, im_h, to_lab):

        self._env = env
        self.encoder: cmc_model.CMCModel = encoder
        self.encoder.eval()

        self.agent_obs = None

        self._im_w = im_w
        self._im_h = im_h
        self._to_lab = to_lab

        self.frame_stack = self._env.observation_spec().shape[0] // 3

    def compute_obs_and_rewards(self, expert_video_dir=None, video_frame=None):
        if expert_video_dir is None and video_frame is None:
            raise ValueError

        self.encoder.eval()
        agent_obs = np.array(self.agent_obs, dtype=np.uint8)

        if self._to_lab:
            agent_episode = datasets.VideoDataset.rgb_to_lab(agent_obs)
            agent_episode = agent_episode.transpose(0, 3, 1, 2)
        else:
            agent_episode = agent_obs.transpose(0, 3, 1, 2)

        if expert_video_dir is not None:
            expert_episode = datasets.VideoDataset.sample_from_dir(expert_video_dir, self.episode_len)
            expert_episode = datasets.VideoDataset.transform_frames(expert_episode, self._im_w, self._im_h, self._to_lab)
        else:
            expert_episode = video_frame[:self.episode_len + 1]
            expert_episode = datasets.VideoDataset.transform_frames(expert_episode, self._im_w, self._im_h, self._to_lab)
        expert_episode = expert_episode.transpose(0, 3, 1, 2)

        with torch.no_grad():
            expert_seq_states = self.encoder.encode(torch.tensor(expert_episode, dtype=torch.float, device=utils.device())).cpu().numpy()
            agent_seq_states = self.encoder.encode(torch.tensor(agent_episode, dtype=torch.float, device=utils.device())).cpu().numpy()
        rewards = - np.linalg.norm(expert_seq_states - agent_seq_states, axis=1)
        return rewards, agent_obs

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        self.agent_obs = [self.physics.render(height=self._im_h, width=self._im_w, camera_id=0)]
        return time_step

    def step(self, action) -> TimeStep:
        time_step = self._env.step(action)
        self.agent_obs.append(self.physics.render(height=self._im_h, width=self._im_w, camera_id=0))
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class EncoderStackWrapper(dm_env.Environment):
    def __init__(self, env, encoder, state_dim):

        self._env = env
        self.encoder: cmc_model.CMCModel = encoder
        self.encoder.eval()

        self.state_dim = state_dim

        self.expert_seq_states = None
        self.agent_seq_states = None
        self.agent_states = None

        self.frame_stack = self._env.observation_spec().shape[0] // 3

    def encode(self, observation):
        frames = []
        for i in range(self.frame_stack):
            frames.append(observation[3 * i: 3 * (i+1)])
        frames = np.array(frames, dtype=np.float)
        with torch.no_grad():
            frames = torch.tensor(frames, device=utils.device(), dtype=torch.float)
            states = self.encoder.encode_frame(frames)
            states = torch.flatten(states)
            states = states.cpu().numpy()
        return states

    def compute_episode_reward(self, expert_video_dir=None, video_frame=None):
        if expert_video_dir is None and video_frame is None:
            raise ValueError
        if expert_video_dir is not None:
            episode = datasets.VideoDataset.sample_from_dir(expert_video_dir, self.episode_len)
            episode = datasets.VideoDataset.transform_frames(episode, self.im_w, self.im_h, self.to_lab)
        else:
            episode = video_frame[:self.episode_len + 1]
            episode = datasets.VideoDataset.transform_frames(episode, self.im_w, self.im_h, self.to_lab)

        with torch.no_grad():
            T = len(episode)
            batches = []
            for i in range(0, T, 64):
                batch = episode[i: i + 64]
                batch = np.array(batch)
                batch = torch.tensor(batch.transpose((0, 3, 1, 2)), device=utils.device(), dtype=torch.float)
                e_seq = self.encoder.encode_frame(batch)
                del batch
                batches.append(e_seq)
            e_seq = torch.concat(batches)
            z_seq = self.encoder.encode_state_seq(e_seq)

        self.expert_seq_states = z_seq.cpu().numpy()

        s_seq = torch.tensor(np.array(self.agent_states), dtype=torch.float, device=utils.device())
        with torch.no_grad():
            self.agent_seq_states = self.encoder.encode_state_seq(s_seq).cpu().numpy()
            rewards = - np.linalg.norm(self.agent_seq_states - self.expert_seq_states, axis=1)
        return rewards

    def reset(self) -> TimeStep:
        self.agent_states = []
        time_step = self._env.reset()
        with torch.no_grad():
            s = self.encode(time_step.observation)
            self.agent_states.append(s[-self.state_dim:])
        return time_step._replace(observation=s)

    def step(self, action) -> TimeStep:
        time_step = self._env.step(action)
        with torch.no_grad():
            s = self.encode(time_step.observation)
            self.agent_states.append(s[-self.state_dim:])
        return time_step._replace(observation=s)

    def observation_spec(self):
        state_dim = self.state_dim * self.frame_stack
        return specs.Array(shape=(state_dim,), dtype=np.float32, name='observation')

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ChangeContextWrapper(dm_env.Environment):
    def __init__(self, env, context_changer: context_changers.ContextChanger, camera_id, pixels_key):
        self._context_changer = context_changer
        self._env = env
        self._camera_id = camera_id
        self._pixels_key = pixels_key

    def reset(self):
        self._context_changer.reset()
        time_step = self._env.reset()
        self._context_changer.change_env(self._env)
        observation = time_step.observation
        observation[self._pixels_key] = self._env.physics.render(height=self.im_h, width=self.im_w,
                                                                 camera_id=self._camera_id)
        time_step = time_step._replace(observation=observation)
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        self._context_changer.change_env(self._env)
        observation = time_step.observation
        observation[self._pixels_key] = self._env.physics.render(height=self.im_h, width=self.im_w,
                                                                 camera_id=self._camera_id)
        time_step = time_step._replace(observation=observation)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class EpisodeLenWrapper(dm_env.Environment):
    def __init__(self, env, ep_len):
        self._env = env
        self._ep_len = ep_len
        self._counter = 0

    def reset(self) -> TimeStep:
        self._counter = 0
        return self._env.reset()

    def step(self, action) -> TimeStep:
        self._counter += 1
        time_step = self._env.step(action)
        if self._counter >= self._ep_len:
            time_step = time_step._replace(step_type=StepType.LAST)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, xml_path=None, camera_id=None, im_w=84, im_h=84, context_changer: context_changers.ContextChanger = None, episode_len=None, to_lab=False):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'

    if xml_path is not None:
        env.physics.reload_from_xml_path(to_absolute_path(xml_path))

    env.im_w = im_w
    env.im_h = im_h
    env.to_lab = to_lab
    env.episode_len = episode_len

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        if camera_id is None:
            camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=im_h, width=im_w, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        if context_changer is not None:
            env = ChangeContextWrapper(env, context_changer, camera_id, pixels_key)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    if episode_len is not None:
        env = EpisodeLenWrapper(env, episode_len)
    return env


def wrap(env, frame_stack, action_repeat, episode_len=None, to_lab=False):
    env.to_lab = to_lab
    env.episode_len = episode_len
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, frame_stack, 'pixels')
    env = ExtendedTimeStepWrapper(env)
    if episode_len is not None:
        env = EpisodeLenWrapper(env, episode_len)
    return env

