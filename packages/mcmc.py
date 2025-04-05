import numpy as np
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from layers_pt import Net
from dynamics_pt import Dynamics
from sampler_pt import propose
from notebook_utils_pt import get_hmc_samples, plot_gaussian_contours

import MHstep
import utils

def rwm_general(dist,hist_0, num_bin, chainlen, Z0, c):
    count = []
    sample0 = []
    timetaken = []
    kl_divergence = []
    ess = []
    burnin = int(chainlen/2)
    for i in range(len(Z0)):
        tic = time.time()
        Zi = [Z0.cpu().detach()[i]]
        counts = 0
        for j in tqdm(range(chainlen)):
            x = Zi[j]
            x_prop = torch.distributions.MultivariateNormal(x,torch.mul(c,torch.eye(len(Z0[i][:])))).sample() # Using multinorm proposal here, should be symmetric
            log_like = dist.Ex(x_prop) # proposal is -U(x)
            log_prop = dist.Ex(x) 
            
            accept = MHstep.torch_logstep(log_like, log_prop, 1,1)
            if accept == 1:
                #sample0.append(x_prop)
                Zi.append(x_prop)
            else:
                #sample0.append(x)
                Zi.append(x)
            if j >= burnin:
                counts += accept
        sample0.append(Zi)
        count.append(counts)
        timing = time.time() - tic
        timetaken.append(timing)
        
        kl_divergenced = utils.kl_div(hist_0, dist.x_edges(num_bin), dist.y_edges(num_bin), np.array(sample0)[i], burnin)
        kl_divergence.append(kl_divergenced)
        
        ess0 = np.array([utils.neff(np.array(sample0)[i,burnin:,k]) for k in range(2)])
        ess.append(ess0)
        
    return timetaken, count, np.array(sample0), kl_divergence, ess


def log_Q(potential, z_prime, z, step):
    z.requires_grad_()
    grad = torch.autograd.grad(potential(z).mean(), z)[0]
    return -(torch.norm(z_prime - z + step * grad, p=2, dim=1) ** 2) / (4 * step)

def mala_general(dist,hist_0, num_bin, chainlen, Z0, step):
    count = []
    sample1 = []
    timetaken = []
    kl_divergence = []
    ess = []
    burnin = int(chainlen/2)
    for i in range(len(Z0)):
        tic = time.time()
        Zi = Z0.cpu().detach()[i]
        sample0 = []
        counts = 0
        for j in tqdm(range(chainlen)):
            Zi.requires_grad_()
            u = dist.Ex(Zi).mean()
            grad = torch.autograd.grad(u, Zi)[0]
            prop_Zi = Zi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(1, 2)
            log_like = dist.Ex(Zi).mean()
            log_prop = dist.Ex(prop_Zi).mean()
            kernel_prop = log_Q(dist.Ex, Zi, prop_Zi, step)
            kernel_x = log_Q(dist.Ex, prop_Zi, Zi, step)
            
            accept = MHstep.torch_logstep(log_like, log_prop, kernel_prop, kernel_x)
            if accept == 1:
                Zi = prop_Zi

            if j >= burnin:
                counts += accept
            sample0.append(Zi.detach().numpy())
        count.append(counts)
        timing = time.time() - tic
        timetaken.append(timing)
        real_sample = np.vstack(sample0)
        samples = np.vstack(sample0)[burnin:]

        kl_divergenced = utils.kl_div(hist_0, dist.x_edges(num_bin), dist.y_edges(num_bin), samples, 0)
        kl_divergence.append(kl_divergenced)
        
        ess0 = np.array([utils.neff(samples[:,k]) for k in range(2)])
        ess.append(ess0)
    
        sample1.append(real_sample)

    return timetaken, count, np.array(sample1), kl_divergence, ess


def neural_train(n_steps, n_samples, dynamics, x_dim, device, use_barker=False):
    scale = torch.tensor(0.1, device=device)
    
    optim = Adam(dynamics.parameters())
    scheduler = StepLR(optim, step_size=1000, gamma=0.96)
    dynamics.alpha
    def criterion(v1, v2):
        return scale * (torch.mean(1.0 / v1) + torch.mean(1.0 / v2)) + (-torch.mean(v1) - torch.mean(v2)) / scale
    
    for t in tqdm(range(n_steps)):    
        if(t==0):
            x = torch.randn(n_samples, x_dim, dtype=torch.float32, device=device)
        else:
            x = output[0]
    
        z = torch.randn_like(x, dtype=torch.float32, device=device)
    
        optim.zero_grad()
        Lx, _, log_px, output, _, _ = propose(x, dynamics, do_mh_step=True, device=device, use_barker=use_barker)
        Lz, _, log_pz, _, _, _ = propose(z, dynamics, do_mh_step=False, device=device, use_barker=use_barker)
        
        if use_barker:
            px = log_px[0].exp()
            pz = log_pz[0].exp()
        else:
            px = log_px.exp()
            pz = log_pz.exp()
        
    
        v1 = (torch.sum((x - Lx)**2, dim=1) * px) + torch.tensor(1e-4, dtype=torch.float32, device=device)
        v2 = (torch.sum((z - Lz)**2, dim=1) * pz) + torch.tensor(1e-4, dtype=torch.float32, device=device)
        scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    
        loss = criterion(v1, v2)
        loss.backward()
    
        optim.step()
    #     pdb.set_trace()
    
        if t % 1000 == 0 or t == n_steps-1:
            current_lr = None
            for param_group in optim.param_groups:
                current_lr = param_group['lr']
            print ('Step: %d / %d, Loss: %.2e, Acceptance sample: %.2f, LR: %.5f' % (t, n_steps, loss.item(), np.mean(px.cpu().detach().numpy()), current_lr))
        scheduler.step()
        optim.zero_grad()
    dynamics.alpha


def l2hmc_general(dist, hist_0, num_bin, chainlen, Z0, T, eps, n_sample, x_dim, network, device, hmc=False):
    half_step = int(chainlen/2)
    kl_divergence = []
    ess = []
    
    #print("T = {}, eps = {}".format(T, eps))
    if not hmc:
        dynamics = Dynamics(x_dim, dist.get_energy_function(), T, eps, hmc=False, net_factory=network, device=device)
        neural_train(int(chainlen/4), 512, dynamics, x_dim, device=device)
    else:
        dynamics = Dynamics(x_dim, dist.get_energy_function(), T, eps, hmc=True, device=device) 
    
    samples = Z0
    final_samples = []

    samples_ = samples
    tic = time.time()
    
    with torch.no_grad():
    #     pdb.set_trace()
        for t in tqdm(range(chainlen)):
            final_samples.append(samples_.cpu().numpy())
            if not hmc:
                _, _, log_px, samples_, _, _ = propose(samples_, dynamics, do_mh_step=True, device=device)
                samples_ = samples_[0].detach()
            else:
                _, _, log_px, samples_ = propose(samples_, dynamics, do_mh_step=True, device=device)
                samples_ = samples_[0][0].detach()
    timetaken = time.time() - tic
    accept = log_px.exp()

    final_samples = np.array(final_samples)
    sample_burn_in = final_samples[half_step:]
    k, l, m = sample_burn_in.shape

    for i in range(n_sample):
        kl_divergenced = utils.kl_div(hist_0, dist.x_edges(num_bin), dist.y_edges(num_bin), sample_burn_in.reshape(l,k,m)[i], 0)
        kl_divergence.append(kl_divergenced)
    
    ess0 = np.array([[utils.neff(sample_burn_in[:,i,j]) for j in range(2)] for i in range(n_sample)])
    ess.append(ess0)

    return timetaken, np.mean(accept.cpu().detach().numpy()), final_samples, kl_divergence, ess



class mcmc_algos(object):
    def __init__(self, mcmc, hist_0, num_bin, chainlen, Z0, para1, para2=0, n_sample=0, x_dim=0, network=0, device='cpu', hmc=False, gravity=False):
        self.hist_0 = hist_0
        self.num_bin = num_bin
        self.chainlen = chainlen
        self.Z0 = Z0
        self.para1 = para1

        if mcmc == "RWM":
            self.result = lambda dist : rwm_general(dist, self.hist_0, self.num_bin, self.chainlen, self.Z0, self.para1)

        elif mcmc == "MALA":
            self.result = lambda dist : mala_general(dist, self.hist_0, self.num_bin, self.chainlen, self.Z0, self.para1)

        elif mcmc == "HMC":
            self.para2 = para2
            self.hmc = hmc
            self.n_sample = n_sample
            self.x_dim = x_dim
            self.network = network
            self.device = device
            self.result = lambda dist : l2hmc_general(dist, self.hist_0, self.num_bin, self.chainlen, self.Z0, self.para1, self.para2, 
                                                      self.n_sample, self.x_dim, self.network, self.device, self.hmc)
        
        else:
            print('no such mcmc algorithm')
        