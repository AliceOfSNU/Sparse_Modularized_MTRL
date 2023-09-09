import torch
import numpy as np
import copy
from collections import OrderedDict

from .base import BaseCollector
import torchrl.policies as policies
from metaworld_utils.meta_env import generate_single_mt_env


# I don't think this is used in the project...
class MultiTaskCollectorBase(BaseCollector):

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        # idx = ob_info["task_index"]
        task_idx = env_info.env.active_task
        out = pf.explore( torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
            [task_idx])
        act = out["action"]
        act = act[0]
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

        sample_dict = { 
            "obs":ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idx": task_idx,
            "rewards": [reward],
            "terminals": [done]
        }

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.env_rank)

        return next_ob, done, reward, info
    
class MultiTaskCollector(BaseCollector):
    def __init__(self, env, pf, replay_buffer, # required arguments for base
            env_cls, env_args, # required arguments for multi-task
            progress_alpha=0.1, reset_idx=False, # optional named
            **kwargs):
        
        super().__init__(env, pf, replay_buffer,**kwargs) # collector base
        self.env_cls  = env_cls
        self.env_args = env_args
        self.tasks = list(self.env_cls.keys())
        self.task_nums = len(self.tasks)
        self.env_info.task_nums = self.task_nums
        self.active_worker_nums = self.task_nums

        self.env_info.device = self.device
        self.replay_buffer.rebuild_from_tag()
        self.reset_idx = reset_idx
        self.tasks_progress = [0] * self.task_nums
        self.progress_alpha = progress_alpha

        # per-env variables persisting between epochs, reset on env done.
        # (train only)
        self.c_obs = [None] * self.task_nums
        self.cached_train_rews = [0] * self.task_nums
        self.obs_info_dict = {}
        
        # build all train envs
        single_mt_env_args = {
                "task_cls": None,
                "task_args": None,
                "env_rank": None,
                "num_tasks": self.task_nums,
                "max_obs_dim": np.prod(self.env.observation_space.shape),
                "env_params": self.env_args[0],
                "meta_env_params": self.env_args[2]
            }
        self.norm_obs_flag = self.env_args[0]["obs_norm"]

        self.env = None 
        self.envs = []
        self.eval_envs = []

        for task_idx, task_name in enumerate(self.tasks):
            # set build arguments
            single_mt_env_args["task_cls"] = self.env_cls[task_name]
            single_mt_env_args["task_args"] = copy.deepcopy(self.env_args[1][task_name])
            if "start_epoch" in single_mt_env_args["task_args"]:
                del single_mt_env_args["task_args"]["start_epoch"]
            single_mt_env_args["env_rank"] = task_idx

            # build train envs
            env = generate_single_mt_env(**single_mt_env_args)
            if self.norm_obs_flag:
                self.obs_info_dict[task_name] = {
                    "obs_mean": env._obs_mean,
                    "obs_var": env._obs_var
                }
            self.c_obs[task_idx] = {
                "ob": env.reset()
            }
            env.train()
            self.envs.append(env)
            print("built " + task_name)


            # build eval envs
            eval_env = generate_single_mt_env(**single_mt_env_args)
            eval_env.eval()
            eval_env._reward_scale = 1
            self.eval_envs.append(eval_env)
            print("built eval " + task_name)


    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):
        # funcs are in eval mode at this stage.(batch_norms are off)
        pf = funcs["pf"]
        ob = ob_info["ob"]
        task_idx = env_info.env_rank

        assert isinstance(pf, policies.EmbeddingGuassianContPolicyBase)

        # sample action without grad
        # no grad because we are only collecting experience off-policy, not training anything..
        with torch.no_grad():
            embedding_input = torch.zeros(env_info.task_nums)
            embedding_input[task_idx] = 1
            embedding_input = embedding_input.unsqueeze(0)
            out = pf.explore(torch.Tensor(ob).to(env_info.device).unsqueeze(0), embedding_input.to(env_info.device))
            act = out["action"]

        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]

        # step env
        next_ob, reward, done, info = env_info.env.step(act)
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        # unset done flag on termination by step length.
        if env_info.current_step >= env_info.max_episode_frames:
            done = False
            info["time_limit"] = True

        # add to replay
        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [task_idx],
            "rewards": [reward],
            "terminals": [done]
        }
        sample_dict["embedding_inputs"] = embedding_input.cpu().numpy()

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample(sample_dict, task_idx)

        return next_ob, done, reward, info

    
    def eval_one_epoch(self):
        eval_rews = [] # collect across all envs
        mean_success_rate = 0 # collect across all envs
        tasks_result = []

        self.pf.eval() # batch norm off..
        # iterate over all envs
        for task_idx, env in enumerate(self.eval_envs):
            env.eval()
            env._reward_scale = 1

            task_name = self.tasks[task_idx]

            if self.norm_obs_flag:
                env._obs_mean = self.obs_info_dict[task_name]["obs_mean"]
                env._obs_var = self.obs_info_dict[task_name]["obs_var"]

            done = False
            env_success = 0
            env_eval_rews = []

            # evaluate for certain num of episodes
            for idx in range(self.eval_episodes):
                if self.reset_idx:
                    #eval_ob = env.reset_with_index(idx)
                    eval_ob = env.reset()
                else:
                    eval_ob = env.reset()
                rew = 0
                current_success = 0 # 0 or 1
                while not done: # one episode
                    # sample action
                    embedding_input = torch.zeros(self.task_nums)
                    embedding_input[task_idx] = 1
                    embedding_input = embedding_input.unsqueeze(0).to(self.device)
                    act = self.pf.eval_act( torch.Tensor( eval_ob ).to(self.device).unsqueeze(0), embedding_input)
                    
                    # step env
                    eval_ob, r, done, info = env.step( act )
                    rew += r
                    
                    # measures success
                    if self.eval_render:
                        env.render()
                    current_success = max(current_success, info["success"])

                # summarize one episode--_+_+
                env_eval_rews.append(rew)
                done = False
                env_success += current_success
            
            # summarize evaluation for this task
            env_success_rate = env_success / self.eval_episodes
            tasks_result.append((task_name, env_success_rate, np.mean(env_eval_rews)))
            self.tasks_progress[task_idx] *= (1 - self.progress_alpha)
            self.tasks_progress[task_idx] += self.progress_alpha * env_success_rate
            mean_success_rate += env_success_rate
            eval_rews += env_eval_rews


        # in dictionary order of task names:
        tasks_result.sort()
        dic = OrderedDict()

        for task_name, env_success_rate, env_eval_rews in tasks_result:
            dic[task_name+"_success_rate"] = env_success_rate
            dic[task_name+"_eval_rewards"] = env_eval_rews
    
        dic['eval_rewards']      = eval_rews
        dic['mean_success_rate'] = mean_success_rate / self.task_nums
        self.pf.train() # back to training mode.
        return dic

    def train_one_epoch(self):

        # cumulative over all envs.
        train_epoch_reward = 0 
        train_rews = []

        # iterate over all envs
        for i, env in enumerate(self.envs):
            env.train()
            self.env_info.env = env # to pass to classmethod take_actions.
            self.env_info.env_rank = i
            assert id(self.env_info.env) == id(env) # assure no copy
            # load env related variables
            task_name = self.tasks[i]
            train_rew = self.cached_train_rews[i]
            c_ob = self.c_obs[i]

            # collect *epoch_frames frames. under eval_mode and no grad.
            self.pf.eval()
            for _ in range(self.env_info.epoch_frames):
                # sample actions
                next_ob, done, reward, info = self.__class__.take_actions(self.funcs, self.env_info, c_ob, self.replay_buffer )
                c_ob["ob"] = next_ob #is set to reset'ed obs in case take_action has reached done or max steps.
                train_rew += reward
                train_epoch_reward += reward
                
                if done or ("time_limit" in info and info["time_limit"]):
                    #train_rews - sum of rewards for each episode, regardless of task.
                    #can be multiple entries per task on each exploration epoch. 
                    train_rews.append(train_rew)
                    train_rew = 0
            
            if self.norm_obs_flag:
                self.obs_info_dict[task_name] = {
                    "obs_mean": self.env_info.env._obs_mean,
                    "obs_var": self.env_info.env._obs_var
                }

            # save env-related variables
            self.cached_train_rews[i] = train_rew
            self.c_obs[i] = c_ob

        self.pf.train()

        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }
    
