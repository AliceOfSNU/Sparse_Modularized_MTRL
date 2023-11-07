
from .twin_sac_q import TwinSACQ
from .mt_sac import MTSAC
from .routing_sac import RTSAC
import numpy as np
import time
from collections import deque

import os
import os.path as osp

import torchrl.policies as policies
import torchrl.utils as utils
import torch.nn.functional as F

class SACSAC():
    def __init__(self, main_agent: MTSAC, 
                router_agent:TwinSACQ, 
                replay_buffer,
                collector,        
                logger = None,
                num_epochs = 3000,
                epoch_frames = 1000,
                eval_episodes = 5,
                max_episode_frames = 999,
                batch_size = 128,
                device = 'cpu',
                pretrain_epochs = 0,
                target_hard_update_period = 1000,
                use_soft_update = True,
                opt_times = 1,
                **kwargs):
        
        # components
        self.main_agent = main_agent
        self.router_agent = router_agent
        self.replay_buffer = replay_buffer
        self.collector = collector     
        self.pretrain_epochs = pretrain_epochs
        self.opt_times = opt_times
        # device specification
        self.device = device
        # training information
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.eval_episodes = eval_episodes
        self.max_episode_frames = max_episode_frames
        self.batch_size = batch_size
        #for soft Q updates
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        # Logger & relevant setting
        self.logger = logger
        self.episode_rewards = deque(maxlen=30)
        self.training_episode_rewards = deque(maxlen=30)
        
    def start_epoch(self):
        pass

    def finish_epoch(self):
        return {}


    def pretrain(self):
        total_frames = 0
        #self.pretrain_epochs * self.collector.worker_nums * self.epoch_frames
        
        for pretrain_epoch in range( self.pretrain_epochs ):

            start = time.time()
            self.start_epoch()
            
            training_epoch_info =  self.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)

            finish_epoch_info = self.finish_epoch()
            total_frames += self.collector.active_worker_nums * self.epoch_frames
            infos = {}
            
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(self.training_episode_rewards)
            infos.update(finish_epoch_info)
            print("epoch: ", pretrain_epoch, " frames:", total_frames, "infos:", infos)
            if self.logger:
                self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos, csv_write=False )
        
        self.pretrain_frames = total_frames
        print("Finished Pretrain.")
            
    def snapshot(self, prefix, epoch):
        pass
        #for name, network in self.snapshot_networks:
        #    model_file_name="model_{}_{}.pth".format(name, epoch)
        #    model_path=osp.join(prefix, model_file_name)
        #    torch.save(network.state_dict(), model_path)

    def train(self):
        self.pretrain()
        total_frames = 0
        if hasattr(self, "pretrain_frames"):
            total_frames = self.pretrain_frames

        self.start_epoch()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            start = time.time()

            self.start_epoch() # just passes.
            
            ############ EXPLORE #############
            explore_start_time = time.time()
            # collector loop
            training_epoch_info =  self.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)
            explore_time = time.time() - explore_start_time

            ############ UPDATE ############
            train_start_time = time.time()
            # main agent and router agent samples independently.
            self.main_agent.update_per_epoch()
            self.router_agent.update_per_epoch()
            train_time = time.time() - train_start_time

            finish_epoch_info = self.finish_epoch() # just passes..

            ########### EVALUATE ############
            eval_start_time = time.time()
            eval_infos = self.evaluate()
            eval_time = time.time() - eval_start_time

            total_frames += self.collector.active_worker_nums * self.epoch_frames

        