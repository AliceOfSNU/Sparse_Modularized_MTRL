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

import os
import os.path as osp
import random
import copy
import matplotlib.colors as colors
import seaborn as sns

# external dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt

###
# draws each connection(module selection)'s activations for a given layer
# pcolors
def draw_layer_weights(data, layer, title):
    agg = { task_name:[None, None] for task_name in task_names}
    for key in data:
        if "selects_flattened" in key:
            for task_name in task_names:
                prefix = task_name + "_l{}".format(layer)
                if prefix in key:
                    agg[task_name][0] = data[key].flatten().detach().cpu().numpy()
        #if "final_selects" in key:
        #    for task_name in task_names:
        #        if task_name in key:
        #            agg[task_name][1] = data[key].flatten().detach().cpu().numpy()

    #data = eval_infos["selects_flattened"]#layer wts{l}
    i = 0
    task_labels = []
    plot_data = []
    for task_name, dt in agg.items():
        task_labels.append(task_name)
        plot_data.append(dt[0])
    
    fig, ax1 = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    plot_data = np.stack(plot_data, axis=0)
    ax1 = fig.add_subplot(111)
    ax1.pcolor(plot_data)
    ax1.set_yticks(np.arange(0.5, 10, 1), [t for t in task_labels], fontsize = 12)
    ax1.set_xticks(np.arange(0.5, 16, 4), ["module{}".format(i) for i in range(4)], fontsize = 12)
    #ax1.set_xticks(np.arange(0.5, 4, 4), [i for i in range(4)], fontsize = 12)

    plt.title(title, fontsize=20)

    if not os.path.exists( "./fig" ):
        os.mkdir( "./fig" )
    plt.savefig( os.path.join( "./fig", title)) 
    plt.close()


def tsne(x:np.ndarray, cols, targets, title):
    import sklearn
    from sklearn.manifold import TSNE
    
    embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x)
    
    fig, ax1 = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    points_per_task = 5

    ax1.scatter(embedding[:,0], embedding[:,1], c=cols)
    
    plt.show()
    plt.title(title, fontsize=20)

    if not os.path.exists( "./fig" ):
        os.mkdir( "./fig" )
    plt.savefig( os.path.join( "./fig", title)) 
    plt.close()