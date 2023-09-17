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
from torchrl.algo import MTSACHARD
from torchrl.collector.mt import MultiTaskCollector
from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer

from metaworld_utils.meta_env import get_meta_env


# run
'''
    python analysis/eval_mt10_hard.py \
        --pf_snap log/MT10_Hard/mt10/3/model/model_pf_best.pth \
        --qf1_snap log/MT10_Hard/mt10/3/model/model_qf1_best.pth \
        --qf2_snap log/MT10_Hard/mt10/3/model/model_qf2_best.pth \
        --seed 3 \
        --id MT10_Hard \
        --config meta_config/mt10/modular_4_4_2_300_hard.json
        
'''

def do_eval(agent):
    eval_infos = agent.evaluate()

    # tabulate
    #tabulate_list = [["Name", "Value"]]
    #for info in eval_infos:
    #    if "_success_rate" not in info:continue
    #    tabulate_list.append([ info, "{:.5f}".format( eval_infos[info] ) ])
    #
    #tabulate_list.append([])
    #print( tabulate(tabulate_list) )
    
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(30,10))

    ax1 = fig.add_subplot(111)
    data = eval_infos["wts_2"]#layer wts{l}
    ax1.pcolor(data.detach().cpu().numpy(), cmap='RdBu')
    plt.title("Routing Differences 2", fontsize=40)
    task_names=['reach-v1', 
        'push-v1', 
        'pick-place-v1', 
        'door-v1', 
        'drawer-open-v1', 
        'drawer-close-v1', 
        'button-press-topdown-v1', 
        'ped-insert-side-v1', 
        'window-open-v1', 
        'window-close-v1']
    ax1.set_xticks(np.arange(0.5, 40 ,4),task_names, rotation=-45)
    ax1.set_yticks(np.arange(0.5, 4, 1), ["module{}".format(i) for i in range(4)])
    ax1.set_xlabel("Count of Each Module In (n-1)th Layer Selected, per Task")
    ax1.set_ylabel("Module in Nth Layer")
    
    if not os.path.exists( "./fig" ):
        os.mkdir( "./fig" )
    plt.savefig( os.path.join( "./fig", 'routing_layer3.png') ) 
    plt.close()

class GradientBox:
    def __init__(self, optim):
        self.optim = optim
        self.objectives = []

    def per_task_grads(self, data):
        # divide gradients into tasks
        # input: batch of raw data loss
        task_grads_info = {}
        task_losses = data.mean(dim=0).squeeze(0)
        self.objectives = [*task_losses.chunk(task_losses.shape[0])]

        task_grads_info = self.backward_proc()
        return task_grads_info
    

    def backward_proc(self):
        # run the backwards
        task_grads, task_shapes = [], []
        for t, obj in enumerate(self.objectives):
            self.optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            task_grads.append({})
            for n in grad:
                task_grads[t][n] = (self._flatten_grad(grad[n], shape[n])) 
            task_shapes.append(shape)

        # cosine similarity
        cosines = {n:torch.zeros(len(task_grads), len(task_grads)) for n in task_grads[0]}
        for ti in range(len(task_grads)):
            for tj in range(len(task_grads)):
                for n in task_grads[ti]:
                    cosines[n][ti, tj] = torch.dot(task_grads[ti][n], task_grads[tj][n])
        for n in cosines:
            cosines[n].cpu().detach()

        return {"cosines":cosines, "mags":[]}

    # toolbox from pcgrad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        
        grad, shape = {}, {}
        i,j = 0,0
        for group in self.optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario

                if p.grad is not None and p.grad.ndim == 2:
                    if p.grad.shape[0] == 300 and p.grad.shape[1] ==300:
                        n = "module{}_{}".format(i, j)
                        shape[n] = p.grad.shape
                        grad[n] = p.grad.clone()
                        j = (j+1)%4
                        if j == 0: i+=1
                
        return grad, shape

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad
    
def do_extract_grads(agent):
    agent.middlebox = GradientBox(agent.pf_optimizer)
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

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)


    from torchrl.networks.init import normal_init

    example_ob = env.reset()
    example_embedding = env.active_task_one_hot
    total_opt_times = params["general_setting"]["num_epochs"]*params["general_setting"]["opt_times"]
    pf = policies.ModularGuassianSelectCascadeContPolicy(
        input_shape=env.observation_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=2 * env.action_space.shape[0],
        **params['net'])

    if args.pf_snap is not None:
        pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

    qf1 = networks.FlattenModularSelectCascadeCondNet(
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1,
        **params['net'])
    qf2 = networks.FlattenModularSelectCascadeCondNet( 
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
    agent = MTSACHARD(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,

        **params['sac'],
        **params['general_setting']
    )
    
    #do_eval(agent)
    do_extract_grads(agent)

if __name__ == "__main__":
    experiment(args)
