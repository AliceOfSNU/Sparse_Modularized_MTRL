import gym
import numpy as np


class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
        self._wrapped_env = env
        self.training = True

    def train(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.train()
        self.training = True

    def eval(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.eval()
        self.training = False

    def render(self, mode='human', **kwargs):
        return self._wrapped_env.render(mode=mode, **kwargs)

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)


class RewardShift(gym.RewardWrapper, BaseWrapper):
    def __init__(self, env, reward_scale=1):
        super(RewardShift, self).__init__(env)
        self._reward_scale = reward_scale

    def reward(self, reward):
        if self.training:
            return self._reward_scale * reward
        else:
            return reward


def update_mean_var_count_from_moments(
        mean, var, count, 
        batch_mean, batch_var, batch_count):
    """
    Imported From OpenAI Baseline
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormObsSimp(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Simple normalization between -1 and 1
    """
    def __init__(self, env, clipob=10.):
        super(NormObsSimp, self).__init__(env)
        self.dim = self.env.observation_space.shape[0]
        self.high = self._wrapped_env.observation_space.high
        self.low = self._wrapped_env.observation_space.low
        self.mean = (self.high+self.low)/2
        self.diff = (self.high-self.low)/2
        self.clipob = clipob

    def observation(self, observation):
        res = observation[:self.dim]
        res = np.clip((res-self.mean)/(1e-8 + self.diff), -self.clipob, self.clipob)
        observation[:self.dim] = res
        return observation
    

class NormObs(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Optional, Use Momentum
    """
    def __init__(self, env, epsilon=1e-4, clipob=10.):
        super(NormObs, self).__init__(env)
        self.count = epsilon
        self.clipob = clipob
        self._obs_mean = np.zeros(env.observation_space.shape[0])
        self._obs_var = np.ones(env.observation_space.shape[0])

    def _update_obs_estimate(self, obs):
        obs = obs[:self._obs_mean.shape[0]]
        self._obs_mean, self._obs_var, self.count = update_mean_var_count_from_moments(
            self._obs_mean, self._obs_var, self.count, obs, np.zeros_like(obs), 1)

    def _apply_normalize_obs(self, raw_obs):
        if self.training:
            self._update_obs_estimate(raw_obs)
        obs_mean = np.zeros_like(raw_obs)
        obs_mean[:self._obs_mean.shape[0]] = self._obs_mean
        obs_var = np.ones_like(raw_obs)
        obs_var[:self._obs_var.shape[0]] = np.sqrt(self._obs_var) + 1e-8

        return np.clip(
                (raw_obs - obs_mean) / obs_var,
                -self.clipob, self.clipob)

    def observation(self, observation):
        return self._apply_normalize_obs(observation)


class NormRet(BaseWrapper):
    def __init__(self, env, discount=0.99, epsilon=1e-4):
        super(NormRet, self).__init__(env)
        self._ret = 0
        self.count = 1e-4
        self.ret_mean = 0
        self.ret_var = 1
        self.discount = discount
        self.epsilon = 1e-4

    def step(self, act):
        obs, rews, done, infos = self.env.step(act)
        if self.training:
            self.ret = self.ret * self.discount + rews
            # if self.ret_rms:
            self.ret_mean, self.ret_var, self.count = update_mean_var_count_from_moments(
                self.ret_mean, self.ret_var, self.count, self.ret, 0, 1)
            rews = rews / np.sqrt(self.ret_var + self.epsilon)
            self.ret *= (1-done)
            # print(self.count, self.ret_mean, self.ret_var)
        # print(self.training, rews)
        return obs, rews, done, infos

    def reset(self, **kwargs):
        self.ret = 0
        return self.env.reset(**kwargs)
