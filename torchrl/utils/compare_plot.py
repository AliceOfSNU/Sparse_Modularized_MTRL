# draws comparison graphs
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import argparse
import seaborn as sns
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+')
    parser.add_argument('--field_names', type=str, nargs='+')
    parser.add_argument('--fields', type=str, nargs='+')
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--smooth_coeff', type=int, default=25,
                    help='smooth coeff')
    args = parser.parse_args()
    return args


args = get_args()
env_name = args.task_name


def post_process(array):
    smoth_para = args.smooth_coeff
    new_array = []
    for i in range(len(array)):
        if i < len(array) - smoth_para:
            new_array.append(np.mean(array[i:i+smoth_para]))
        else:
            new_array.append(np.mean(array[i:None]))
    return new_array  

# draw a single plot with data from different experiments
def compare_experiments(ax, paths, fields, names):
    values = []
    timesteps = []
    for p, field, name in zip(paths, fields, names):
        file_path = os.path.join(p, 'log.csv')
        with open(file_path,'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                values.append(float(row[args.entry]))
                timesteps.append(int(row["Total Frames"]))
    #plot
    ax.plot(timesteps, values, label=field, color=next(color_it), linestyle='solid', linewidth=2)


sns.set("paper")

current_palette = sns.color_palette()
sns.palplot(current_palette)

fig = plt.figure(figsize=(10,7))
plt.subplots_adjust(left=0.07, bottom=0.15, right=1, top=0.90,
                wspace=0, hspace=0)

ax = fig.add_subplot(111)
colors = current_palette
color_it = iter(colors)
for p,f,n in zip(args.paths, args.field_names, args.names):
    print(p, f, n)
    
compare_experiments(ax, args.paths, args.fields, args.names)