import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import argparse
import csv

# names
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

# run
'''
python torchrl/utils/plot_curves.py \
    --path log/MT10_Hard/mt10/3/log.csv log/MT10_Conditioned_Modular_Deep/mt10/1/log.csv\
    --tags Hard Soft\
    --item drawer-close-v1_success_rate\
    --max_m 10\
    --title 'Drawer-Close Success Rate'
'''

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--path', type=str, nargs='+', default=(0,),
                        help='which csv files to read')
    parser.add_argument('--tags', type=str, nargs='+', default=[],
                        help='legend items')
    parser.add_argument('--item', type=str, nargs='+', default=('mean_success_rate'),
                        help='which column to read')
    parser.add_argument('--max_m', type=int, default=None,
                        help='maximum million')
    parser.add_argument('--smooth_coeff', type=int, default=25,
                        help='smooth coeff')
    parser.add_argument('--title', type=str, default='experiment',
                        help='title')
    parser.add_argument('--output_dir', type=str, default='./fig',
                        help='directory for plot output (default: ./fig)')
    args = parser.parse_args()

    if args.item[0] == "ALL_TASKS":
        args.path = [args.path[0]]*len(task_names)
        args.item = ["{}_success_rate".format(s) for s in task_names]
        args.tags = [s for s in task_names]
    
    if len(args.tags) == 0:
        args.tags = [s for s in args.item]
    return args


args = get_args()

def post_process(array, smooth_para):
    new_array = []
    for i in range(len(array)):
        if i < len(array) - smooth_para:
            new_array.append(np.mean(array[i:i+smooth_para]))
        else:
            new_array.append(np.mean(array[i:None]))
    return new_array    


fig = plt.figure(figsize=(10,7))
plt.subplots_adjust(left=0.07, bottom=0.15, right=1, top=0.90,
                wspace=0, hspace=0)
ax1 = fig.add_subplot(111)
colors = ['b', 'g', 'r', 'c', 'y' ]
styles = ['solid', 'dotted', 'dashed']

# draw!
for linecnt, (path, tag) in enumerate(zip(args.path, args.tags)):
    lc = colors[linecnt % len(colors)]
    step_number = []

    #assume no repetition of experiments(N=1 always, no average across seeds)
    temp_ys = []
    temp_xs = []
    with open(path,'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            temp_ys.append(float(row[args.item[0]])) #read item
            temp_xs.append(int(row["Total Frames"]))

    step_number = np.array(temp_xs) / 1e6 # time axis in millions
    final_step = []
    last_step = len(step_number)
    for i in range(len(step_number)):
        if args.max_m is not None and step_number[i] >= args.max_m:
            last_step = i
            break
        final_step.append(step_number[i])

    temp_ys= temp_ys[:last_step]
    # main plot
    ys = post_process(temp_ys, args.smooth_coeff)
    ax1.plot(final_step, ys, label=tag, color=lc, linewidth=2, alpha = 1.0)
    # no smoothing
    ax1.plot(final_step, temp_ys, color=lc, linewidth=2, alpha=0.3)


ax1.set_xlabel('Million Samples', fontsize=30)
ax1.tick_params(labelsize=25)

box = ax1.get_position()

leg = ax1.legend(
           loc='upper center', bbox_to_anchor=(0.5, -0.05),
           ncol=3,
           fontsize=20)

for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)

plt.title(args.title, fontsize=40)
if not os.path.exists( args.output_dir ):
    os.mkdir( args.output_dir )
plt.savefig( os.path.join( args.output_dir, '{}.png'.format(args.title) ) )
plt.close()