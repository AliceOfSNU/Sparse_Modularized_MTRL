import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torchrl.networks as networks
from .distribution import TanhNormal
import torch.nn.functional as F
import torchrl.networks.init as init
from torch.distributions import Categorical

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class CategoricalPolicyBase(networks.Net):
    def __init__(self, num_layers, num_modules,
                 base_type, 
                 em_input_shape,
                 input_shape,
                 **kwargs):
        
        # this information is needed to correctly normalize the distribution
        self.num_layers = num_layers
        self.num_modules = num_modules
        output_shape = self.num_modules + (self.num_layers-1)*self.num_modules*self.num_modules
        super().__init__(output_shape, base_type, em_input_shape, input_shape, **kwargs)
        
        
    def eval_act(self, x, embedding_input ):
        with torch.no_grad():
            logits, final_logits = self.forward(x, embedding_input)
            selects = torch.argmax(logits, dim=-1, keepdim=True)
            final_selects = torch.argmax(final_logits, dim=-1, keepdim=True)
        return {"selects": selects.detach().cpu().numpy(),
                "final_selects": final_selects().cpu().numpy()}
    
    def forward(self, x, embedding_input):
        x = torch.cat([x, embedding_input], dim=-1)
        out = super().forward(x)
        
        layer_split = [self.num_modules*self.num_modules*(self.num_layers-1), self.num_modules]
        logit, final_logits = out.split(layer_split, dim=-1)
        
        #shape(batch_size*(num_tasks)*L*M*M)
        logit = logit.view([*logit.shape[:-1], self.num_layers-1, self.num_modules, self.num_modules])
        logit = F.softmax(logit, -1)
        final_logits = F.softmax(final_logits, -1)
        return logit, final_logits
    
    # EXPLORE: generate actions to use for env interactions
    # https://github.com/toshikwa/sac-discrete.pytorch/blob/master/sacd/model.py#L98
    def explore( self, x, embedding_input, return_log_probs = False):
        logits, final_logits = self.forward(x, embedding_input)

        dist = Categorical(probs=logits)
        routes = dist.sample()
        log_probs = dist.log_prob(routes)
        routes = F.one_hot(routes, num_classes=self.num_modules)
        log_prob = log_probs.sum()
        
        final_dist = Categorical(probs=final_logits)
        final_route = final_dist.sample()
        log_probs = final_dist.log_prob(final_route)
        final_route = F.one_hot(final_route, num_classes=self.num_modules)
        
        log_prob += log_probs.sum()
        
        # a flattened version to compactly store in replay
        action = torch.cat([torch.flatten(routes, -3), final_route], dim=-1)
        dic = {
            "logits": logits,
            "final_logits": final_logits,
            "routes": routes,
            "final_route": final_route,
            "action": action #flattend version.
        }

        if return_log_probs:
            dic["log_probs"] = log_prob
        
        return dic