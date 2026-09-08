"""Learned sparse superposition and controlled ReLU SAEs.

Toy compression architecture/training adapted from Anthropic's MIT notebook,
commit 562710e079704b84a132b640db134d4cebe22466. Copyright (c) 2022
Anthropic; full license: scripts/licenses/anthropic_toy_models_MIT.txt.
SAE training, generators, and correspondence experiments are CCAD additions.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn


def sample(n, features, probability, regime, generator, device='cuda'):
    """Equal marginal Bernoulli-uniform variables; pair 0,1 may co-occur exactly.

    mixed has 20% independent rows, 80% exact pair copies. Every remaining
    factor stays independent. No evaluation/teacher outputs enter this draw.
    """
    x=torch.rand((n,features),generator=generator,device=device)
    x=x*(torch.rand((n,features),generator=generator,device=device)<probability)
    if regime=='correlated':x[:,1]=x[:,0]
    elif regime=='coactive':
        x[:,1]=torch.rand(n,generator=generator,device=device)*(x[:,0]>0)
    elif regime=='mixed':
        share=torch.rand(n,generator=generator,device=device)<.8
        x[share,1]=x[share,0]
    elif regime!='independent':raise ValueError(regime)
    return x


class ToyModel(nn.Module):
    def __init__(self,features,hidden,seed,device='cuda'):
        super().__init__();g=torch.Generator(device=device).manual_seed(seed)
        self.w=nn.Parameter(torch.randn((features,hidden),generator=g,device=device)*math.sqrt(2/(features+hidden)))
        self.bias=nn.Parameter(torch.zeros(features,device=device))
    def forward(self,x):return torch.relu(x@self.w@self.w.T+self.bias)


class SAEs(nn.Module):
    """Independent seeds, identical shared batches/order and optimization."""
    def __init__(self,hidden,width,seeds,device='cuda'):
        super().__init__();ds=[]
        for seed in seeds:
            g=torch.Generator(device=device).manual_seed(seed)
            d=torch.randn((width,hidden),generator=g,device=device)
            ds.append(d/d.norm(dim=1,keepdim=True))
        d=torch.stack(ds);self.decoder=nn.Parameter(d);self.encoder=nn.Parameter(d.clone().transpose(1,2))
        self.encoder_bias=nn.Parameter(torch.zeros((len(seeds),width),device=device))
        self.decoder_bias=nn.Parameter(torch.zeros((len(seeds),hidden),device=device))
    def encode(self,h):
        a=torch.einsum('bsh,shf->bsf',h[:,None,:]-self.decoder_bias,self.encoder)+self.encoder_bias
        return a,torch.relu(a)
    def forward(self,h):
        a,z=self.encode(h);return torch.einsum('bsf,sfh->bsh',z,self.decoder)+self.decoder_bias,a,z
    @torch.no_grad()
    def normalize(self):self.decoder.div_(self.decoder.norm(dim=2,keepdim=True).clamp_min(1e-12))


def train(cfg,regime,toy_seed,progress,toy_state=None):
    device=cfg['device'];model=ToyModel(cfg['features'],cfg['hidden'],toy_seed,device)
    g=torch.Generator(device=device).manual_seed(cfg['training_data_seed'])
    gv=torch.Generator(device=device).manual_seed(cfg['quality_data_seed'])
    val=sample(cfg['quality_samples'],cfg['features'],cfg['probability'],regime,gv,device)
    if toy_state is not None:model.load_state_dict(toy_state)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['toy_lr'],weight_decay=.01)
    traces=[]
    for step in range(0 if toy_state is not None else cfg['toy_steps']):
        lr=cfg['toy_lr']*math.cos(.5*math.pi*step/max(cfg['toy_steps']-1,1))
        for pg in opt.param_groups:pg['lr']=lr
        x=sample(cfg['batch_size'],cfg['features'],cfg['probability'],regime,g,device)
        loss=(model(x)-x).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (step+1)%cfg['log_every']==0 or step==0 or step==cfg['toy_steps']-1:
            with torch.no_grad():mse=(model(val)-val).square().mean().item()
            row=dict(stage='toy',regime=regime,toy_seed=toy_seed,step=step+1,validation_mse=mse,learning_rate=lr)
            traces.append(row);progress(row)
    model.requires_grad_(False);sae=SAEs(cfg['hidden'],cfg['sae_width'],cfg['sae_seeds'],device)
    opt=torch.optim.Adam(sae.parameters(),lr=cfg['sae_lr'])
    # Reset independent generator: all SAE seeds see the very same stream.
    g=torch.Generator(device=device).manual_seed(cfg['sae_training_data_seed'])
    with torch.no_grad():hv=val@model.w;center_energy=(hv-hv.mean(0)).square().sum(1).mean()
    for step in range(cfg['sae_steps']):
        lr=cfg['sae_lr']*(.1+.9*math.cos(.5*math.pi*step/max(cfg['sae_steps']-1,1)))
        for pg in opt.param_groups:pg['lr']=lr
        x=sample(cfg['batch_size'],cfg['features'],cfg['probability'],regime,g,device);h=x@model.w
        pred,_,z=sae(h);losses=((pred-h[:,None,:]).square().sum(2)+cfg['l1']*z.sum(2)).mean(0)
        # Sum, not mean over instances: gradient/LR does not depend on seed count.
        opt.zero_grad(set_to_none=True);losses.sum().backward();opt.step();sae.normalize()
        if (step+1)%cfg['log_every']==0 or step==0 or step==cfg['sae_steps']-1:
            with torch.no_grad():
                pred,_,z=sae(hv);fve=1-(pred-hv[:,None,:]).square().sum(2).mean(0)/center_energy
                labels=(sae.decoder@model.w.T/model.w.norm(dim=1)).argmax(2)
                truth_energy=(val[:,:,None]*model.w).square().sum((1,2)).mean()
                semantic_errors=torch.zeros(len(cfg['sae_seeds']),device=device)
                for fi in range(cfg['features']):
                    groupz=z*(labels==fi)[None,:,:]
                    group=torch.einsum('bsf,sfh->bsh',groupz,sae.decoder)
                    truth=val[:,fi,None]*model.w[fi]
                    semantic_errors+=(group-truth[:,None,:]).square().sum(2).mean(0)/truth_energy
                for si,seed in enumerate(cfg['sae_seeds']):
                    row=dict(stage='sae',regime=regime,toy_seed=toy_seed,sae_seed=seed,step=step+1,learning_rate=lr,
                             fve=fve[si].item(),l0=(z[:,si]>0).float().sum(1).mean().item(),
                             alive=int((z[:,si]>0).any(0).sum().item()),width=cfg['sae_width'],source_factor_error=semantic_errors[si].item(),
                             decoder_norm_min=sae.decoder[si].norm(dim=1).min().item(),decoder_norm_max=sae.decoder[si].norm(dim=1).max().item())
                    traces.append(row);progress(row)
    return model,sae,traces


def ridge_fit(x,y,mu_x,mu_y,fraction=1e-5):
    xc=x-mu_x;yc=y-mu_y;scale=np.trace(xc.T@xc)/max(x.shape[1],1)
    return np.linalg.solve(xc.T@xc+fraction*max(scale,1e-12)*np.eye(x.shape[1]),xc.T@yc)


def threshold_state(x,indices,coef,mu_x,mu_y):
    """Frozen preactivation estimator followed by the source ReLU."""
    return np.maximum((x[:,indices]-mu_x[indices])@coef+mu_y,0)


def relu_secant(a0,a1):
    """Exact elementwise secant, with zero for equal endpoints.

    Each value lies in [0,1]. For equal endpoints the contribution is zero
    regardless of slope; choosing zero avoids an arbitrary kink convention.
    """
    a0=np.asarray(a0,dtype=np.float64);a1=np.asarray(a1,dtype=np.float64)
    difference=a1-a0
    return np.divide(np.maximum(a1,0)-np.maximum(a0,0),difference,out=np.zeros_like(difference),where=difference!=0)


def fit_rectified_state(x,y,decoder,coef,mu_x,mu_y,maxiter=200):
    """Standard L-BFGS fitting of finite ReLU-state errors on fixed members.

    Decoder-metric loss plus 0.05 diagonal prevents unpenalized cancellation;
    the normalized input penalty is fixed, with no evaluation data used.
    """
    from scipy.optimize import minimize
    # One common input RMS keeps coefficient regularization in the original
    # feature units. Per-feature standardization made near-dead predictors
    # almost unpenalized and produced huge held-out coefficients in pilot v5.
    common=float(np.sqrt(np.mean(np.var(x,axis=0)))) if x.shape[1] else 1.0
    scale=np.full(x.shape[1],max(common,1e-5))
    a=np.column_stack([(x-mu_x)/scale,np.ones(len(x))])
    theta=np.vstack([coef*scale[:,None],mu_y[None,:]])
    gram=decoder@decoder.T;metric=gram+.05*np.diag(np.diag(gram))
    penalty=1e-5
    def objective(flat):
        t=flat.reshape(theta.shape);pre=a@t;res=np.maximum(pre,0)-y
        weighted=res@metric
        loss=float(np.sum(weighted*res)/len(x)+penalty*np.sum(t[:-1]**2))
        grad=2*a.T@(weighted*(pre>0))/len(x);grad[:-1]+=2*penalty*t[:-1]
        return loss,grad.ravel()
    initial=objective(theta.ravel())[0]
    result=minimize(objective,theta.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,ftol=1e-11,gtol=1e-7,maxls=30))
    final=objective(result.x)[0]
    # Retain a finite best iterate even at iteration budget, explicitly report.
    if not np.isfinite(result.x).all() or final>initial+1e-10:raise RuntimeError('Rectified fit invalid or worse than its initialization')
    t=result.x.reshape(theta.shape)
    return t[:-1]/scale[:,None],t[-1],dict(converged=bool(result.success),message=str(result.message),iterations=int(result.nit),initial_loss=initial,final_loss=final,common_input_scale=common,ridge_fraction=penalty)
