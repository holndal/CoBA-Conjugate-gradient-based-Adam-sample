# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from torch import Tensor
import torch
import math

def coba(params: List[Tensor],
         grads: List[Tensor],
         cobads: List[Tensor],
         pregrads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         gammatype: str):

    for i, param in enumerate(params):
        grad = grads[i]
        p_grad=pregrads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        cobad=cobads[i]
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        if step==1:
          gamma=0
        else:
          #FR
          if gammatype=="FR":
            a=torch.norm(grad)
            b=torch.norm(p_grad)
            gamma=(a*a)/(b*b)
          #PRP
          elif gammatype=="PRP":
            a=torch.norm(p_grad)
            gamma = grad*(grad-p_grad) / (a*a)
          else:
            raise Exception("UNKNOWN GAMMATYPE: "+str(gammatype))
        #cobad = cobad*(0.0001/step**(1+0.00001))*gamma　+ grad
        cobad.mul_(0.0001/step**(1+0.00001)).mul_(gamma).add_(grad)
        #cobad=grad-(0.0001/step**(1+0.00001))*gamma*cobad
        exp_avg.mul_(beta1).add_(cobad, alpha=1 - beta1)

        # exp_avg_sq<= exp_avg_sq*beta2 + (1-beta2)*grad*grad
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            # root(exp_avg_sq) / root(bias_correction2) + eps
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)

class CoBA(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, gammatype="FR"):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, gammatype=gammatype)
        super(CoBA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CoBA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
       loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            cobads=[]
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            for p in group['params']:
                
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['cobad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        self.pre_grads=[]
                        for grad in grads:
                          self.pre_grads.append(torch.zeros_like(grad, memory_format=torch.preserve_format))
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    cobads.append(state['cobad'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
            coba(params_with_grad,
                   grads,
                   cobads,
                   self.pre_grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   gammatype=group["gammatype"])
            
            self.pre_grads=[]
            for g in grads:
              self.pre_grads.append(g.clone())
        return loss
