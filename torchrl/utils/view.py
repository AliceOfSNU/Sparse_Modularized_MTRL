# this creates an evaluation 
import sys
import os

sys.path.append(os.getcwd())
import torch
import numpy as np
import copy
import torchrl.policies as policies
import torchrl.networks as networks
from metaworld_utils.meta_env import generate_single_mt_env
from metaworld_utils.meta_env import get_meta_env
from torchrl.utils import get_params

import random
import imageio


def evaluate_once(
    task_name:str,
    save_path: str,
):
    # create eval env
    single_env_args = {
                "task_cls": cls_dicts[task_name],
                "task_args": copy.deepcopy(cls_args[task_name]),
                "env_rank": args.task_id,
                "num_tasks": len(tasks),
                "max_obs_dim": np.prod(env.observation_space.shape),
                "env_params": env_params,
                "meta_env_params": mt_params
            }
    if "start_epoch" in single_env_args["task_args"]:
        del single_env_args["task_args"]["start_epoch"]

    eval_env = generate_single_mt_env(**single_env_args)
    eval_env.eval()

    # run model, creating images
    pf.eval()
    eval_ob = eval_env.reset()
    rew = 0
    success = 0

    done = False
    frames = []
    rews = []
    while not done:
        if isinstance(pf, policies.EmbeddingGuassianContPolicyBase):
            embedding_input = torch.zeros(env.num_tasks)
            embedding_input[args.task_id] = 1
            embedding_input = embedding_input.unsqueeze(0).to(device)
            act = pf.eval_act( torch.Tensor( eval_ob ).to(device).unsqueeze(0), embedding_input)
        else: 
            act = pf.eval_act(torch.Tensor(eval_ob).to(device).unsqueeze(0))
        eval_ob, r, done, info = eval_env.step(act)
        rew += r
        rews.append(r)
        frame = eval_env.render('rgb_array')
        frames.append(frame)
        success = max(success, info["success"])

    if success:
        print("environment solved!")

    # convert to gif and save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ptx = len(os.listdir(save_path)) + 1
    filename = task_name + "_" + str(ptx) +".gif"
    filename = os.path.join(save_path, filename)

    print(filename)
    eval_env.close()
    del eval_env
    
    imageio.mimsave(filename, [np.array(fr) for i, fr in enumerate(frames) if i%2 == 0])



# draws network connectivity
def draw_graph(
        
):
    pass

import argparse
parser = argparse.ArgumentParser(description='RL')
    
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--env_name', type=str, default='mt10',
                    help='environment trained on (default: mt10)')
parser.add_argument('--config', type=str, default='mt10',
                    help='config file')
parser.add_argument('--task_id', type=int, default=0,
                    help='environment to test on')
parser.add_argument('--id', type=str, default=None,
                    help='experiment id')

args = parser.parse_args()
params = get_params(args.config)

use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if use_cuda:
    torch.backends.cudnn.deterministic=True
device = torch.device("cuda:0" if use_cuda else "cpu")
print("using device: ", device.type)

# make mt env
env_params = params['env']
mt_params = params['meta_env']

env, cls_dicts, cls_args = get_meta_env( args.env_name, env_params, mt_params)
env.seed(args.seed)
tasks = list(cls_dicts.keys())
task_name = tasks[args.task_id]
print("evaluating on task: ", task_name)

# load model
params['net']['base_type']=networks.MLPBase
example_embedding = env.active_task_one_hot
pf = policies.ModularGuassianGatedCascadeCondContPolicy(
        input_shape=env.observation_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=2 * env.action_space.shape[0],
        **params['net'])
model_path = os.path.join('log', args.id, args.env_name, str(args.seed), 'model' )
out_path = os.path.join('fig/renders', args.id, args.env_name, str(args.seed))
model_file = os.path.join(model_path, "model_pf_best.pth")
pf.load_state_dict(torch.load(model_file))
pf.to(device)
# run
evaluate_once(task_name, out_path)