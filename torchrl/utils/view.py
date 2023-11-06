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
import matplotlib.pyplot as plt
import random
import imageio
import graphviz
from pprint import pprint
from PIL import Image

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

def evaluate_once(
    task_name:str,
    save_path: str,
    draw_g_weights= True,
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
    gwts = []
    framecnt = 0
    while not done:
        if isinstance(pf, policies.EmbeddingGuassianContPolicyBase):
            embedding_input = torch.zeros(env.num_tasks)
            embedding_input[args.task_id] = 1
            embedding_input = embedding_input.unsqueeze(0).to(device) 
            if draw_g_weights:
                act, weights = pf.eval_act(torch.Tensor( eval_ob ).to(device).unsqueeze(0), embedding_input, return_weights = True)
                gwts += weights
            else:
                act = pf.eval_act( torch.Tensor( eval_ob ).to(device).unsqueeze(0), embedding_input)
        else: 
            if draw_g_weights:
                act, weights = pf.eval_act(torch.Tensor(eval_ob).to(device).unsqueeze(0), return_weights = True)
                gwts += weights
            else:
                act = pf.eval_act(torch.Tensor(eval_ob).to(device).unsqueeze(0))
        
        eval_ob, r, done, info = eval_env.step(act)
        rew += r
        rews.append(r)
        frame = eval_env.render('rgb_array')
        framecnt += 1
        if framecnt%2 == 0: frames.append(np.array(frame))
        success = max(success, info["success"])
    print(f"total {framecnt} frames")
    fig = plt.figure(figsize=(5, 1))
    fig.add_subplot(111)

    plt.plot(rews, label='rewards')
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    rew_plt = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rew_plt = rew_plt.reshape(fig.canvas.get_width_height()[::-1] + (3,))   
    print(rew_plt.shape)
    rnd = np.concatenate((frames[0], rew_plt))
    print(rnd.shape)

    for i, rnd in enumerate(frames):
        frames[i] = np.concatenate((rnd, rew_plt))
    
    # convert to gif and save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if draw_g_weights:
        weights = torch.cat(gwts, 0)
        weights = weights.detach().cpu().numpy()
        gframes = _draw_graph(weights, save_path, title=f'weights_{task_name}')
        gframes = gframes[:len(frames)]
        w=frames[0].shape[1]
        h=frames[0].shape[0]
        gw=gframes[0].shape[1]
        gh=gframes[0].shape[0]
        for rnd, gvz in zip(frames, gframes):
            rnd[:gh, w-gw:]=gvz[:,:]

    if success:
        print("environment solved!")


    ptx = len(os.listdir(save_path)) + 1
    filename = task_name + "_" + str(ptx) +".gif"
    filename = os.path.join(save_path, filename)

    eval_env.close()
    del eval_env
    
    imageio.mimsave(filename, [fr for i, fr in enumerate(frames) if i%2 == 0])



# draws network connectivity
# this function has dependency on 
def _draw_graph(
    weights,
    out_path:str,
    title:str = "Module_Graph_Weights"
):
    frames = []
    print(os.path.join(out_path, f'{title}.png'))
    for fr in range(len(weights)):
        if fr%2: continue
        G = graphviz.Graph('Module Graph Weights', directory=out_path, filename=f'gw_{title}_generated.gv')
        G.attr('node', shape='box')
        for i in range(pf.num_layers-1, 0, -1):
            for j, module in enumerate(pf.layer_modules[i]):
                G.node(f'{i}_{j}')
        
        for i in range(pf.num_layers-1):
            for j in range(pf.num_modules):
                for k in range(pf.num_modules):
                    w = weights[fr][j][k]
                    G.edge(f'{i}_{j}', f'{i+1}_{k}', label=f'{w:.2f}',\
                        color =f"#FF{(int)(250-w*200):02X}{(int)(250-w*200):02X}", fontsize="7", labelfloat = "false", contraint="false")
        
        G.attr(fontsize='16')
        
        # save output from graphviz
        png_path = os.path.join(out_path, f'{title}.png')
        G.render(
            engine = 'dot',
            format = 'png',
            outfile= f'{title}.png'
        )
        
        # make numpy image
        im_frame = Image.open(png_path).convert('RGB')
        frames.append(np.array(im_frame))

    # visualize
    imageio.mimsave(os.path.join(out_path, f'{title}.gif'), frames)

    return frames
    
# 
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
evaluate_once(task_name, out_path, draw_g_weights=True)