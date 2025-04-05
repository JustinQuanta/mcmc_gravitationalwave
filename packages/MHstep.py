import torch
import numpy as np
import math
from scipy import stats
from proposals import Proposals

def log_MHstep(new_log_like, new_log_prop, old_log_prop, old_log_like):
 
    #This is the general Metropolis-Hastings step
    r = new_log_like - new_log_prop + old_log_prop - old_log_like 
    u = np.random.uniform(0, 1)
    alpha = min(1, np.exp(r))
    accept = 0
    if u < alpha:
        accept = 1
        return accept
    return accept

def MHstep(new_log_like, new_log_prop, old_log_prop, old_log_like):
 
    #This is the general Metropolis-Hastings step
    r = new_log_like/new_log_prop * old_log_prop/old_log_like 
    u = np.random.uniform(0, 1)
    alpha = min(1, r)
    accept = 0
    if u < alpha:
        accept = 1
        return accept
    return accept

def torch_logstep(new_log_like, new_log_prop, old_log_prop, old_log_like):
 
    #This is the general Metropolis-Hastings step
    r = new_log_like - new_log_prop + old_log_prop - old_log_like 
    u = torch.rand(1)
    alpha = torch.min(torch.tensor([1,torch.exp(r)]))
    accept = 0
    if u < alpha:
        accept = 1
        return accept
    return accept


def torch_step(new_log_like, new_log_prop, old_log_prop, old_log_like):
 
    #This is the general Metropolis-Hastings step
    r = new_log_like/new_log_prop * old_log_prop/old_log_like 
    u = torch.rand(1)
    alpha = torch.min(torch.tensor([1,r]))
    accept = 0
    if u < alpha:
        accept = 1
        return accept
    return accept