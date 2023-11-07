from .twin_sac_q import TwinSACQ
import copy
import torch
import numpy as np

import torchrl.policies as policies
import torchrl.utils as utils
import torch.nn.functional as F

class RTSAC(TwinSACQ):
    def __init__(self, task_nums,
                 temp_reweight=False,
                 grad_clip=True, record_weights=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.task_nums = task_nums
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.Tensor(0.0).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = self.optimizer_class(
                [self.log_alpha],
                lr=self.plr,
            )
        self.sample_key = ["obs", "next_obs", "routing_acts", "rewards",
                           "terminals",  "task_idxs"]

        self.pf_flag = True
        self.idx_flag = False
        self.temp_reweight = temp_reweight
        if self.pf_flag:
            self.sample_key.append("embedding_inputs")
            
        self.grad_clip = grad_clip
        self.record_weights = False
        self.record_weights = record_weights
        
        del self.collector
        del self.train_render
        del self.eval_render
            
    def update(self, batch):
        self.training_update_num += 1
        obs = batch['obs']
        actions = batch['routing_acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']

        if self.pf_flag:
            embedding_inputs = batch["embedding_inputs"]

        rewards = torch.Tensor(rewards).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)

        if self.pf_flag:
            embedding_inputs = torch.Tensor(embedding_inputs).to(self.device)

        if self.idx_flag:
            task_idx    = torch.Tensor(task_idx).to( self.device ).long()

        self.pf.train()
        self.qf1.train()
        self.qf2.train()

        """
        Policy operations.
        """
        if self.idx_flag:
            sample_info = self.pf.explore(obs, task_idx,
                                        return_log_probs=True)
        else:
            if self.pf_flag:
                sample_info = self.pf.explore(obs, embedding_inputs,
                                            return_log_probs=True, return_weights=self.record_weights)
            else:
                sample_info = self.pf.explore(obs, return_log_probs=True, return_weights=self.record_weights)

        new_actions = sample_info["action"]
        log_probs = sample_info["log_prob"]

        if self.idx_flag:
            q1_pred = self.qf1([obs, actions], task_idx)
            q2_pred = self.qf2([obs, actions], task_idx)
        else:
            if self.pf_flag:
                q1_pred = self.qf1([obs, actions], embedding_inputs)
                q2_pred = self.qf2([obs, actions], embedding_inputs)
            else:
                q1_pred = self.qf1([obs, actions])
                q2_pred = self.qf2([obs, actions])

        reweight_coeff = 1
        if self.automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha = 1
            alpha_loss = 0

        with torch.no_grad():
            if self.idx_flag:
                target_sample_info = self.pf.explore(next_obs,
                                                    task_idx,
                                                    return_log_probs=True)
            else:
                if self.pf_flag:
                    target_sample_info = self.pf.explore(next_obs,
                                                         embedding_inputs,
                                                         return_log_probs=True,
                                                         return_weights = self.record_weights)
                    
                else:
                    target_sample_info = self.pf.explore(next_obs,
                                                        return_log_probs=True,
                                                        return_weights = self.record_weights)

            target_actions = target_sample_info["action"]
            target_log_probs = target_sample_info["log_prob"]

            if self.idx_flag:
                target_q1_pred = self.target_qf1([next_obs, target_actions],
                                                 task_idx)
                target_q2_pred = self.target_qf2([next_obs, target_actions],
                                                 task_idx)
            else:
                if self.pf_flag:
                    target_q1_pred = self.target_qf1([next_obs, target_actions],
                                                    embedding_inputs)
                    target_q2_pred = self.target_qf2([next_obs, target_actions],
                                                    embedding_inputs)
                else:
                    target_q1_pred = self.target_qf1([next_obs, target_actions])
                    target_q2_pred = self.target_qf2([next_obs, target_actions])

            min_target_q = torch.min(target_q1_pred, target_q2_pred)
            target_v_values = min_target_q - alpha * target_log_probs
        """
        QF Loss
        """
        # q_target = rewards + (1. - terminals) * self.discount * target_v_values
        # There is no actual terminate in meta-world -> just filter all time_limit terminal
        q_target = rewards + self.discount * target_v_values

        qf1_loss = (reweight_coeff *
                    ((q1_pred - q_target.detach()) ** 2)).mean()
        qf2_loss = (reweight_coeff *
                    ((q2_pred - q_target.detach()) ** 2)).mean()

        assert q1_pred.shape == q_target.shape, print(q1_pred.shape, q_target.shape)
        assert q2_pred.shape == q_target.shape, print(q1_pred.shape, q_target.shape)

        if self.idx_flag:
            q_new_actions = torch.min(
                self.qf1([obs, new_actions], task_idx),
                self.qf2([obs, new_actions], task_idx))
        else:
            if self.pf_flag:
                q_new_actions = torch.min(
                    self.qf1([obs, new_actions], embedding_inputs),
                    self.qf2([obs, new_actions], embedding_inputs))
            else:
                q_new_actions = torch.min(
                    self.qf1([obs, new_actions]),
                    self.qf2([obs, new_actions]))
        """
        Policy Loss
        """
        if not self.reparameterization:
            raise NotImplementedError
        else:
            assert log_probs.shape == q_new_actions.shape
            policy_loss = (reweight_coeff *
                           (alpha * log_probs - q_new_actions)).mean()

        #std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
        #mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()
        #policy_loss += std_reg_loss + mean_reg_loss

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip:
            pf_norm = torch.nn.utils.clip_grad_norm_(self.pf.parameters(), 1)
        self.pf_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        if self.grad_clip:
            qf1_norm = torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 1)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        if self.grad_clip:
            qf2_norm = torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 1)
        self.qf2_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Routing_Reward_Mean'] = rewards.mean().item()

        if self.automatic_entropy_tuning:
            info["alpha_r"] = self.log_alpha[i].exp().item()
            info["Alpha_r_loss"] = alpha_loss.item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/qfr1_loss'] = qf1_loss.item()
        info['Training/qfr2_loss'] = qf2_loss.item()

        if self.grad_clip:
            info['Training/pfr_norm'] = pf_norm.item()
            info['Training/qfr1_norm'] = qf1_norm.item()
            info['Training/qfr2_norm'] = qf2_norm.item()


        log_probs_display = log_probs.detach()
        log_probs_display = (log_probs_display.mean(0)).squeeze(1)
        for i in range(self.task_nums):
            info["log_prob_{}".format(i)] = log_probs_display[i].item()

        info['log_probs/mean'] = log_probs.mean().item()
        info['log_probs/std'] = log_probs.std().item()
        info['log_probs/max'] = log_probs.max().item()
        info['log_probs/min'] = log_probs.min().item()

        return info

    def update_per_epoch(self):
        for _ in range(self.opt_times):
            batch = self.replay_buffer.random_batch(self.batch_size,
                                                    self.sample_key,
                                                    reshape=False)
            infos = self.update(batch)
            self.logger.add_update_info(infos)
            
    def train(self):
        pass
