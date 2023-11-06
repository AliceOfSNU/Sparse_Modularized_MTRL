# system dependencies
import sys
sys.path.append(".")
import os
import os.path as osp
import random

# external dependencies
import numpy as np
import torch
from tabulate import tabulate

# custom dependencies
from torchrl.utils import get_args
from torchrl.utils import get_params
args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.algo import MTSAC
from torchrl.collector.mt import MultiTaskCollector
from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer

from metaworld_utils.meta_env import get_meta_env


# run
'''
        
    python analysis/eval_mt10_soft.py \
        --pf_snap log/MT10_Conditioned_Modular_Deep/mt10/1/model/model_pf_best.pth \
        --qf1_snap log/MT10_Conditioned_Modular_Deep/mt10/1/model/model_qf1_best.pth \
        --qf2_snap log/MT10_Conditioned_Modular_Deep/mt10/1/model/model_qf2_best.pth \
        --seed 1 \
        --id MT10_Conditioned_Modular_Deep \
        --config meta_config/mt10/modular_4_4_2_128_reweight_rand.json
'''

def eval_grads(agent):
    agent.train()
    

# main
def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']
    print("using device: ", device.type)
    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = None
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase


    from torchrl.networks.init import normal_init

    example_ob = env.reset()
    example_embedding = env.active_task_one_hot

    pf = policies.ModularGuassianGatedCascadeCondContPolicy(
        input_shape=env.observation_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=2 * env.action_space.shape[0],
        **params['net'])

    if args.pf_snap is not None:
        pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

    qf1 = networks.FlattenModularGatedCascadeCondNet(
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1,
        **params['net'])
    qf2 = networks.FlattenModularGatedCascadeCondNet( 
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1,
        **params['net'])

    if args.qf1_snap is not None:
        qf1.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    if args.qf2_snap is not None:
        qf2.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "embedding_inputs": example_embedding
    }

    print("#tasks: ", env.num_tasks)
    replay_buffer = AsyncSharedReplayBuffer(int(buffer_param['size']),
            env.num_tasks
            #args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    params['general_setting']['collector'] = MultiTaskCollector(
        env=env, pf=pf, replay_buffer=replay_buffer,
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes']
    )

    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = None
    agent = MTSAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )

    #eval_grads(agent)
    eval_infos = agent.evaluate()

    ## tabulate
    tabulate_list = [["Name", "Value"]]
    #for info in eval_infos:
    #    if "_success_rate" not in info:continue
    #    tabulate_list.append([ info, "{:.5f}".format( eval_infos[info] ) ])

    #tabulate_list.append([])
    #print( tabulate(tabulate_list) )

if __name__ == "__main__":
    experiment(args)
# run
'''
        
    python analysis/eval_mt10_soft.py \
        --pf_snap log/MT10_Conditioned_Modular_Deep/mt10/1/model/model_pf_best.pth \
        --qf1_snap log/MT10_Conditioned_Modular_Deep/mt10/1/model/model_qf1_best.pth \
        --qf2_snap log/MT10_Conditioned_Modular_Deep/mt10/1/model/model_qf2_best.pth \
        --seed 1 \
        --id MT10_Conditioned_Modular_Deep \
        --config meta_config/mt10/modular_4_4_2_128_reweight_rand.json
'''