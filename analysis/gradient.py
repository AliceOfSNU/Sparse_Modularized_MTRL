import numpy as np
class GradientBox:
    def __init__(self, net, optim):
        self.optim = optim
        self.net = net
        self.objectives = []
        self.grad_infos = []
        self.grad_epoch_infos = None
        self.update_cnt = 0 #counts #of backwards call
        self.epoch_cnt = 0 #counts #of epochs for logging

    def per_task_grads(self, data):
        # divide gradients into tasks
        # input: batch of raw data loss
        task_grad_info = {}
        task_losses = data.sum(dim=0).squeeze(0)
        self.objectives = [*task_losses.chunk(task_losses.shape[0])]

        task_grad_info = self.backward_proc()
        self.update_cnt += 1
        if self.grad_epoch_infos == None:
            self.grad_epoch_infos = task_grad_info
        else:
            for k in task_grad_info:
                for name in task_grad_info[k]:
                    self.grad_epoch_infos[k][name] += task_grad_info[k][name]
    
    def end_epoch(self):
        for k in self.grad_epoch_infos:
            for name in self.grad_epoch_infos[k]:
                self.grad_epoch_infos[k][name] /= self.update_cnt #average it out
        self.grad_infos.append(self.grad_epoch_infos)

        for name in self.grad_epoch_infos["cosines"]:
            if self.epoch_cnt % 50 == 0 or self.epoch_cnt < 10:
                fig, (ax1,ax2)= plt.subplots(1, 2, layout="constrained")
                dt = self.grad_epoch_infos["cosines"][name].cpu()
                ax1.pcolor(dt, norm=colors.LogNorm(vmin=dt.min().item(), vmax=dt.max().item()),
                   cmap='PuBu_r', shading='auto')
                ax1.set_xticks(np.arange(0.5, 10 ,1),task_names, fontsize=12)
                ax1.set_yticks(np.arange(0.5, 10, 1),task_names, fontsize=12)

                dt = self.grad_epoch_infos["mags"][name].cpu()
                ax2.pcolor(dt, norm=colors.LogNorm(vmin=dt.min().item(), vmax=dt.max().item()),
                   cmap='PuBu_r', shading='auto')
                ax2.set_xticks(np.arange(0.5, 10 ,1),task_names, fontsize=12)
                ax2.set_yticks(np.arange(0.5, 10, 1),task_names, fontsize=12)

                # good! save
                plt.title('task gradients COV (epoch{}, {})'.format(self.epoch_cnt,name))
                if not os.path.exists( "./fig/grads" ):
                    os.mkdir( "./fig/grads" )
                plt.savefig( os.path.join( "./fig/grads", 'cosines_epoch{}_{}.png'.format(self.epoch_cnt, name)) ) 
                plt.close()

        # reset and advance epoch
        self.grad_epoch_infos = None # clear!
        self.update_cnt = 0
        self.epoch_cnt += 1
        
    def backward_proc(self):
        # run the backwards
        # list containing vec(grad) and its shape, for each task.(length = #tasks)
        task_grads, task_shapes = [], []
        for t, obj in enumerate(self.objectives):
            self.optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            task_grads.append({n:self._flatten_grad(g, shape[n]) for n, g in grad.items()})
            task_shapes.append(shape)

        # cosine similarity
        cosines = {n:torch.zeros(len(task_grads), len(task_grads)) for n in task_grads[0]}
        for ti in range(len(task_grads)):
            for tj in range(len(task_grads)):
                for n in task_grads[ti]:
                    cosines[n][ti, tj] = torch.dot(F.normalize(task_grads[ti][n], dim=0), F.normalize(task_grads[tj][n], dim=0))
        for n in cosines:
            cosines[n].cpu().detach()

        # magnitude similarity - harmonic mean
        
        mags = {n:torch.zeros(len(task_grads), len(task_grads)) for n in task_grads[0]}
        for ti in range(len(task_grads)):
            for tj in range(len(task_grads)):
                for n in task_grads[ti]:
                    n1 = torch.norm(task_grads[ti][n])
                    n2 = torch.norm(task_grads[tj][n])
                    #mags[n][ti, tj] = n1*n2/(1e-8+n1+n2)
                    mags[n][ti, tj] = n1/(1.0e-12 + n2)
        for n in mags:
            mags[n].cpu().detach()
        

        self.optim.zero_grad(set_to_none=True) # we don't need to step on these gradients
        return {"cosines":cosines, "mags":mags}

    def _retrieve_grad(self):
        grad, shape = {}, {}
        #i,j = 0,0
        #for group in self.optim.param_groups:
        #    for p in group['params']:
        #        # if p.grad is None: continue
        #        if p.grad is not None and p.grad.ndim == 2:
        #            if p.grad.mo:
        #                n = "module{}_{}".format(i, j)
        #                shape[n] = p.grad.shape
        #                grad[n] = p.grad.clone()
        #                j = (j+1)%4
        #                if j == 0: i+=1

        for l in range(self.net.num_layers):
            for m in range(self.net.num_modules):
                name = "module_{}_{}".format(l,m)
                g = getattr(self.net,name).weight.grad.clone()
                grad[name] = g
                shape[name] = g.shape
                
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