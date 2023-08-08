# evaluation on single tasks using double q sac

import sys
sys.path.append(".")

import torch

import os
import time
import os.path as osp
import copy
import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.collector import BaseCollector
from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
from metaworld_utils.meta_env import generate_single_mt_env
import gym

from metaworld_utils.meta_env import get_meta_env

import random

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    print("using device: ", device.type)
    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']
    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    tasks = list(cls_dicts.keys())
    task_name = tasks[args.task_id]

    logger = Logger( experiment_name , task_name, args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase

    from torchrl.networks.init import normal_init

    example_ob = env.reset()

    if "q_hidden_shapes" in params['net']:
        qnet_params = copy.deepcopy(params['net'])
        qnet_params['hidden_shapes'] = params['net']['q_hidden_shapes']
        del qnet_params['q_hidden_shapes']
        del params['net']['q_hidden_shapes']
    else:
        qnet_params = params['net']
        
    pf = policies.GuassianContPolicy(
        input_shape=env.observation_space.shape[0],
        output_shape=2 * env.action_space.shape[0],
        **params['net'])

    if args.pf_snap is not None:
        pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

        
    qf1 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **qnet_params )
    qf2 = networks.FlattenNet( 
        input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
        output_shape = 1,
        **qnet_params )
    
    example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
    }
    replay_buffer = SharedBaseReplayBuffer( int(buffer_param['size']), 1)
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + params['general_setting']['num_epochs']
    

    single_env_args = {
                "task_cls": cls_dicts[task_name],
                "task_args": copy.deepcopy(cls_args[task_name]),
                "env_rank": args.task_id,
                "num_tasks": len(tasks),
                "max_obs_dim": np.prod(env.observation_space.shape),
                "env_params": params["env"],
                "meta_env_params": params["meta_env"],
            }
    if "start_epoch" in single_env_args["task_args"]:
        del single_env_args["task_args"]["start_epoch"]

    
    env = generate_single_mt_env(**single_env_args)
    eval_env = generate_single_mt_env(**single_env_args)

    print("learning on task: ", task_name)
    #print("_env._max_epi is", env._max_episode_steps)
    
    params['general_setting']['collector'] = BaseCollector(
        env=env, pf=pf, replay_buffer=replay_buffer,
        device=device,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        eval_env = eval_env
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = TwinSACQ(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

tasks = [
    'reach-v1', 
    'push-v1', 
    'pick-place-v1', 
    'door-v1', 
    'drawer-open-v1', 
    'drawer-close-v1', 
    'button-press-topdown-v1', 
    'ped-insert-side-v1', 
    'window-open-v1', 
    'window-close-v1'
]

cfg_idxs = [2, 3, 4, 5, 6]
#if __name__ == "__main__":
#    for cfg in cfg_idxs:
#        args.config = f"meta_config/mt1/{tasks[args.task_id]}_sac copy {cfg}.json"
#        args.seed = cfg+10
#        params = get_params(args.config)
#        experiment(args)

if __name__ == "__main__":
    experiment(args)
'''
 CODE NAMES
1 'reach-v1', 
2 'push-v1', 
3 'pick-place-v1', 
4 'door-v1', 
5 'drawer-open-v1', 
6 'drawer-close-v1', 
7 'button-press-topdown-v1', 
8 'ped-insert-side-v1', 
9 'window-open-v1', 
10 'window-close-v1'

'''