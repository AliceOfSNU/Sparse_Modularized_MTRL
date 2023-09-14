import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.networks.init as init
import torchrl.algo.utils as utils

class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)


class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate(append_hidden_shapes):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x):
        out = self.base(x)

        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim = -1)
        return super().forward(out)


def null_activation(x):
    return x

class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, output_shape,
            base_type, em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,

            num_layers, num_modules,

            module_hidden,

            gating_hidden, num_gating_layers,
            
            # gated_hidden
            add_bn = True,
            pre_softmax = False,
            cond_ob = True,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu,
            
             **kwargs ):

        super().__init__()

        self.base = base_type( 
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.activation_func = activation_func
        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * \
                                               num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def forward(self, x, embedding_input, return_weights = False):
        # Return weights for visualization
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        # router

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)
                # weight outputs pre activation
                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last(out)

        if return_weights:
            return out, weights, last_weight
        return out

class ModularSelectCascadeNet(nn.Module):
    def __init__(self, output_shape,
            base_type, em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,

            num_layers, num_modules,

            module_hidden,

            gating_hidden, num_gating_layers,
            # gated_hidden
            add_bn = True,
            pre_softmax = False,
            cond_ob = True,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu,
             **kwargs ):

        super().__init__()

        self.base = base_type( 
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.activation_func = activation_func
        self.select_temp = 1.0
        self.select_desaturation = 1.0
        self.deterministic=False
        module_input_shape = self.base.output_shape
        self.layers = []
        
        self.num_layers = num_layers
        self.num_modules = num_modules

        for i in range(num_layers):
            layer_modules = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_modules.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layers.append(layer_modules)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        # selecting (router)
        self.select_fcs = []
        self.select_cond_fcs = []

        for l in range(num_layers-1):
            select_cond_fc = nn.Linear(self.num_modules**2, gating_input_shape)
            module_hidden_init_func(select_cond_fc)
            self.select_cond_fcs.append(select_cond_fc)
            self.__setattr__(f"select_cond_fc{l+1}", select_cond_fc)

            select_fc = nn.Linear(gating_input_shape, self.num_modules**2)
            module_hidden_init_func(select_fc)
            self.select_fcs.append(select_fc)
            self.__setattr__(f"select_fc_{l+1}", select_fc)

        select_fc = nn.Linear(gating_input_shape, self.num_modules)
        module_hidden_init_func(select_fc)
        self.select_fcs.append(select_fc)
        self.__setattr__(f"select_fc_{num_layers}", select_fc)

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def det(self):
        self.deterministic = True

    def stoc(self):
        self.deterministic = False

    def forward(self, x, embedding_input, return_weights = False):
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        # common embedding network
        if len(self.gating_fcs) > 0:
            for fc in self.gating_fcs:
                embedding = self.activation_func(embedding)
                embedding = fc(embedding)
        # last layer embedding is not passed through activation.

        # select module
        logits = [] #do we need this?
        selects = []
        select_input = self.activation_func(embedding)

        #if self.deterministic:
        if self.deterministic:
            for i in range(self.num_layers-1):
                logit = self.select_fcs[i](select_input) #last dim is num_modules**2
                logit = F.tanh(logit)

                select = logit.view([*logit.shape[:-1], self.num_modules, self.num_modules])
                logits.insert(0, select)

                # v1: select = choose maximum!!
                #max_select_idx = torch.argmax(select, dim=-1, keepdim=True)
                #select = torch.zeros_like(select)
                #select.scatter_(dim = -1, index = max_select_idx, value = 1)

                # v2: select = hard sampling
                select = F.gumbel_softmax(select, tau=0.02, hard=False, dim=-1)

                selects.insert(0, select)
                select_input = self.select_cond_fcs[i](logit)
                select_input = select_input*embedding
                select_input = self.activation_func(select_input)

            final_select = self.select_fcs[self.num_layers-1](select_input)

            # v1: select = choose maximum!!
            #max_select_idx = torch.argmax(select, dim=-1, keepdim=True)
            #select = torch.zeros_like(select)
            #select.scatter_(dim = -1, index = max_select_idx, value = 1)

            # v2: select = hard sampling
            final_select = F.gumbel_softmax(final_select, tau=self.select_temp, hard=False, dim=-1)

        else:
            for i in range(self.num_layers-1):
                logit = self.select_fcs[i](select_input) #last dim is num_modules**2
                logit = F.tanh(logit)
                select = logit.view([*logit.shape[:-1], self.num_modules, self.num_modules])
                #select *= 1/self.select_temp # temperature annealing for hard module
                logits.insert(0, select)
                #logit_select = logit + self.select_daesaturation * torch.max(logit).detach().item()
                
                '''
                Gumbel-Softmax
                **expectation** of gumbel-softmax distribution follows that of categorical dist.with same logits
                up to a reasonable temp(~1.0) then becomes uniform
                **samples** of gumbel-softmax distribution can be either hard or soft(the hard parameter)
                and the soft version still is quite peaked up to reasonable (~1.0)temp, then becomes uniform.
                
                expecting the logits to trained to become peaked -> because even sampling of modules is likely to be bad.
                scheduling from 1.0 to 0.02.
                '''
                selects.insert(0, F.gumbel_softmax(select, tau=self.select_temp, hard=False, dim=-1))
                #selects.insert(0, F.softmax(select , dim=-1))
                select_input = self.select_cond_fcs[i](logit)
                select_input = select_input*embedding
                select_input = self.activation_func(select_input)

            final_select = self.select_fcs[self.num_layers-1](select_input)
            final_select = F.gumbel_softmax(final_select, tau=self.select_temp, hard=False, dim=-1)

        # run forward for each module
        module_outputs = [module(out).unsqueeze(-2) for module in self.layers[0]]
        for l in range(self.num_layers-1):
            module_input = torch.cat(module_outputs, dim=-2)
            module_outputs = []
            for m in range(self.num_modules):
                # broadcasting semantics: starting from last dim(hidden_dim),
                #   either one dim is 1(the selects!) - dim-1
                #   or have equal dims(the number of modules!) - dim-2
                #   or be nonexistant: the batch dim
                out = (selects[l][..., m,:].unsqueeze(-1)*module_input).sum(dim=-2)
                # selection is prior to activation.
                module_outputs.append(self.activation_func(out).unsqueeze(-2))
                
        out = torch.cat(module_outputs, dim=-2)
        out = (final_select.unsqueeze(-1)*out).sum(dim=-2)
        out = self.last(out)
        

        # return the weights?
        if return_weights:
            #logits = [layer_logit.mean(dim = 0).softmax(dim=-1) for layer_logit in logits]
            logits = [layer_logit.mean(dim = 0) for layer_logit in logits]
            selects = [layer_select.sum(dim = 0) for layer_select in selects]
            return (out, logits, selects) #undo stack, without weight plotting
        return out


class FlattenModularGatedCascadeCondNet(ModularGatedCascadeCondNet):
    def forward(self, input, embedding_input, return_weights = False):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, embedding_input, return_weights = return_weights)


class FlattenModularSelectCascadeCondNet(ModularSelectCascadeNet):
    def forward(self, input, embedding_input, return_weights = False):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, embedding_input, return_weights = return_weights)

 
class BootstrappedNet(Net):
    def __init__(self, output_shape, 
                 head_num = 10,
                 **kwargs ):
        self.head_num = head_num
        self.origin_output_shape = output_shape
        output_shape *= self.head_num
        super().__init__(output_shape = output_shape, **kwargs)

    def forward(self, x, idx):
        base_shape = x.shape[:-1]
        out = super().forward(x)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])
        
        out = out.reshape(out_shape)

        idx = idx.view(view_idx_shape)
        idx = idx.expand(expand_idx_shape)

        out = out.gather(-1, idx).squeeze(-1)
        return out


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, idx ):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, idx)
