from collections import deque
import torch
import torch.multiprocessing as mp
import copy
import numpy as np
import gym

class EnvInfo():
    def __init__(self, 
            env,
            device,
            train_render,
            eval_render,
            epoch_frames,
            eval_episodes,
            max_episode_frames,
            continuous,
            env_rank):

        self.current_step = 0

        self.env = env
        self.device = device
        self.train_render = train_render
        self.eval_render = eval_render
        self.epoch_frames = epoch_frames
        self.eval_episodes = eval_episodes
        self.max_episode_frames = max_episode_frames
        self.continuous = continuous
        self.env_rank = env_rank

        # For Parallel Async
        self.env_cls = None
        self.env_args = None

    def start_episode(self):
        self.current_step = 0

    def finish_episode(self):
        pass


class BaseCollector:

    def __init__(
        self,
        env, pf, replay_buffer,
        train_render=False,
        eval_episodes=10,
        eval_render=False,
        eval_env = None,
        epoch_frames=1000,
        device='cpu',
        max_episode_frames = 999):

        self.pf = pf
        self.replay_buffer = replay_buffer

        self.env = env
        self.env.train()
        continuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.train_render = train_render

        #uncomment 
        #if eval_env is None:
        #    self.eval_env = copy.copy(env)
        #else:
        #    self.eval_env = eval_env
        #self.eval_env._reward_scale = 14
        #self.eval_env.eval()
        self.eval_env = None
        self.eval_episodes = eval_episodes
        self.eval_render = eval_render

        self.env_info = EnvInfo(
            env, device, train_render, eval_render,
            epoch_frames, eval_episodes,
            max_episode_frames, continuous, None
        )
        self.c_ob = {
            "ob": self.env.reset()
        }

        self.train_rew = 0
        self.training_episode_rewards = deque(maxlen=20)
        
        # device specification
        self.device = device

        self.to(self.device)

        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        self.worker_nums = 1
        self.active_worker_nums = 1

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        out = pf.explore( torch.Tensor( ob ).to(env_info.device).unsqueeze(0))
        act = out["action"]
        act = act.detach().cpu().numpy()

        if not env_info.continuous:
            act = act[0]

        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = env_info.env.step(act)
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        if env_info.current_step >= env_info.max_episode_frames:
            done = False
            info["time_limit"] = True

        sample_dict = { 
            "obs":ob,
            "next_obs": next_ob,
            "acts": act,
            "rewards": [reward],
            "terminals": [done],
            #"time_limits": [True if "time_limit" in info else False]
        }

            

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.env_rank)

        return next_ob, done, reward, info

    def terminate(self):
        pass

    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0
        self.env.train()
        for _ in range(self.epoch_frames):
            # Sample actions
            next_ob, done, reward, info = self.__class__.take_actions(self.funcs,
                self.env_info, self.c_ob, self.replay_buffer )
            self.c_ob["ob"] = next_ob
            # print(self.c_ob)
            self.train_rew += reward
            train_epoch_reward += reward
            if done or "time_limit" in info:
                self.training_episode_rewards.append(self.train_rew)
                train_rews.append(self.train_rew)
                self.train_rew = 0

        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }

    def eval_one_epoch(self):

        eval_infos = {}
        eval_rews = []

        done = False

        #self.eval_env = copy.copy(self.env)
        #self.eval_env.eval()
        # print(self.eval_env._obs_mean)
        success_epi_cnt = 0.0
        for idx in range(self.eval_episodes):
            eval_ob = self.eval_env.reset()#this will change goal pos
            rew = 0
            success = 0.0
            while not done:
                act = self.pf.eval_act(torch.Tensor(eval_ob).to(
                    self.device).unsqueeze(0))
                eval_ob, r, done, info = self.eval_env.step(act)
                rew += r
                if self.eval_render:
                    self.eval_env.render()
                success = max(success, info["success"])

            eval_rews.append(rew)
            success_epi_cnt += success

            done = False
        
        eval_infos["eval_rewards"] = eval_rews
        eval_infos["success_rates"] = 1.0*success_epi_cnt/self.eval_episodes
        return eval_infos

    def to(self, device):
        for func in self.funcs:
            self.funcs[func].to(device)

    @property
    def funcs(self):
        return {
            "pf": self.pf
        }
