import sys
sys.path.append(".")

import torch

import os
import time
import os.path as osp

import numpy as np

from torchrl.utils import get_args
from torchrl.utils import get_params

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.algo import MTSACHARD
from torchrl.algo import RTSAC
from torchrl.algo import MTSAC
from torchrl.algo import SACSAC
from torchrl.collector.mt import MultiTaskCollector
from torchrl.collector.mt import MultiTaskCollectorWithRouting

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer

from metaworld_utils.meta_env import get_meta_env

import random


def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    ############## REGION ENV ################

    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)
    example_ob = env.reset()
    example_embedding = env.active_task_one_hot
    print("example_ob:", example_ob)
    print("example_embded:", example_embedding)
    
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    print("using device: ", device.type)
    
    buffer_param = params['replay_buffer']
    
    
    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    
    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    # CNN? FC? real basis type
    params['net']['base_type']=networks.MLPBase
    params['rnet']['base_type']=networks.MLPBase
    params['rnet']['num_modules'] = params['net']['num_modules']
    params['rnet']['num_layers'] = params['net']['num_layers']

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    from torchrl.networks.init import normal_init
    
    total_opt_times = params["general_setting"]["num_epochs"]*params["general_setting"]["opt_times"]
    
    ############## REGION PF ################
    
    pf = policies.ModularRoutedGuassianContPolicy(
        input_shape=env.observation_space.shape[0],
        output_shape=2 * env.action_space.shape[0],
        **params['net'])

    pfr = policies.CategoricalPolicyBase(
        input_shape=env.observation_space.shape[0], #input except the task label
        em_input_shape=np.prod(example_embedding.shape), #10 one hot representation
        **params['rnet']
    )
    
    print("pf ok")
    if args.pf_snap is not None:
        pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

    ############## REGION QF ################
    
    #qf1 = networks.FlattenModularSparseCondNet(
    #    input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
    #    em_input_shape=np.prod(example_embedding.shape),
    #    output_shape=1, cond_ob=True,
    #    **params['net'])
    num_layers = params['rnet']['num_layers']
    num_modules = params['rnet']['num_modules']
    select_size = (num_layers-1)*num_modules*num_modules + num_modules
    qf1 = networks.FlattenModularRoutedCascadeNet(
        input_shape=env.observation_space.shape[0],
        output_shape=2 * env.action_space.shape[0],
        **params['net'])
    qf2 = networks.FlattenModularRoutedCascadeNet(
        input_shape=env.observation_space.shape[0],
        output_shape=2 * env.action_space.shape[0],
        **params['net'])
    
    print("qf ok")
    
    # specify network architecture as FC layers in params
    qfr1 = networks.FlattenNet(
        input_shape=env.observation_space.shape[0]+select_size,
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1, 
        **params['rnet'])
    qfr2 = networks.FlattenNet(        
        input_shape=env.observation_space.shape[0]+select_size,
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1, 
        **params['rnet'])
    print("qfr ok")

    if args.qf1_snap is not None:
        qf1.load_state_dict(torch.load(args.qf1_snap, map_location='cpu'))
    if args.qf2_snap is not None:
        qf2.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    #if args.qfr1_snap is not None:
    #    qfr1.load_state_dict(torch.load(args.qfr1_snap, map_location='cpu'))
    #if args.qfr2_snap is not None:
    #    qfr2.load_state_dict(torch.load(args.qfr2_snap, map_location='cpu'))
    
    
    ########### REGION REPLAY ############
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "routing_acts": np.array([0] * select_size),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "embedding_inputs": example_embedding
    }

    replay_buffer = AsyncSharedReplayBuffer(int(buffer_param['size']), env.num_tasks)
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    print("replay buf ok.")
    
    
    ######## REGION COLLECTOR #######
    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']
        
    params['general_setting']['collector'] = MultiTaskCollectorWithRouting(
        env=env, pf=pf, pfr=pfr, replay_buffer=replay_buffer,
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes']
    )

    print("collector ok")
    
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    
    ############# REGION Agent #############
    actor = MTSAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )
    router = RTSAC(
        pf = pfr,
        qf1 = qfr1,
        qf2 = qfr2,
        task_nums=env.num_tasks,
        **params['rsac'],
        **params['general_setting']
    )
    agent = SACSAC(actor, router, **params['general_setting'])
    
    print("agent ok")
    
    ########## MAIN TRAIN LOOP ###########
    agent.train()

if __name__ == "__main__":
    experiment(args)
