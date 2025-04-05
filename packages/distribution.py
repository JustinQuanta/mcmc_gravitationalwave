import numpy as np
from scipy.stats import multivariate_normal, ortho_group
import torch
import torch.nn as nn

def gaussian_torch(x,mu,S,d):
    normalise = torch.sqrt((2*torch.pi)**(d) * torch.det(S))
    return -0.5*torch.sum((x-mu)@torch.inverse(S)@torch.t(x-mu)) + torch.log(1/normalise)

def gmm(x,mu,cov,pis,device):
    d = torch.tensor(x.shape[0],device=device)
    return torch.log(pis) + gaussian_torch(x,mu,cov,d)

def four_gmm(x, mus, covs, pis, device):
    return torch.log(torch.exp(gmm(x,mus[0],covs[0],pis[0],device=device)) + 
                      torch.exp(gmm(x,mus[1],covs[1],pis[1],device=device)) + 
                      torch.exp(gmm(x,mus[2],covs[2],pis[2],device=device)) + 
                      torch.exp(gmm(x,mus[3],covs[3],pis[3],device=device)))

def gmm_sample(pis, mus, covs, n):
    dists = [torch.distributions.MultivariateNormal(loc=mus[i], covariance_matrix=covs[i]) for i in range(len(mus))]
    p1, p2, p3, p4 = pis[0], pis[1], pis[2], pis[3]
    n_first = int(n*p1)
    n_second = int(n*p2)
    n_third = int(n*p3) 
    n_fourth = int(n*p4)
    samples_1 = dists[0].sample((n_first,))
    samples_2 = dists[1].sample((n_second,))
    samples_3 = dists[2].sample((n_third,))
    samples_4 = dists[3].sample((n_fourth,))
    samples = torch.cat([samples_1, samples_2, samples_3, samples_4])
    idx = torch.randperm(samples.shape[0])
    return samples[idx].view(samples.size()).cpu().detach()


def moon_E(x):
    x = x.view(-1, 2)
    x1, x2 = x[:, 0], x[:, 1]
    E_y = x1.pow(2)/2
    E_x = (x2- x1.pow(2) + 4).pow(2)/2
    return -(E_y + E_x)

def moon_E2(x):
    x = x.view(-1, 2)
    x1, x2 = x[:, 0], x[:, 1]
    E_y = x1.pow(2)/2
    E_x = (x2- x1.pow(2) + 4).pow(2)/2
    return (E_y + E_x)

def moon_sample(N_samples):
    y = torch.randn((N_samples,1))
    x = torch.randn((N_samples,1)) + y.pow(2) - 4
    return torch.cat((y,x),dim=1)

def log_likelihood(template,observed):
    # Calculate the difference and norm squared, which is the waveform inner product
    difference = observed - template
    norm_squared = np.linalg.norm(difference, axis=-1)**2
    return -0.5 * torch.tensor(norm_squared) #log likelihood is -0.5<d-h|d-h>


class Moon(nn.Module):
    def __init__(self, device='cpu'):
        super(Moon, self).__init__()
        self.device = device
        self.x_edges = lambda x1: np.linspace(-4, 4, x1)
        self.y_edges = lambda y1: np.linspace(-7, 5, y1)
        
    def moon_E(self, x):
        E_y = x[:, 0].pow(2) / 2
        E_x = (x[:, 1] - x[:, 0].pow(2) + 4).pow(2) / 2
        return E_y + E_x

    def get_energy_function(self):
        def fn(x):
            return self.moon_E(x)
        return fn

    def log_density(self, X):
        return -self.moon_E(X)


class four_GMM(nn.Module):
    def __init__(self, mus, sigmas, pis, device='cpu'):
        super(four_GMM, self).__init__()
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.p1, self.p2, self.p3, self.p4 = pis[0], pis[1], pis[2], pis[3]
        self.log_pis = [torch.tensor(np.log(self.p1), dtype=torch.float32, device=device),
                        torch.tensor(np.log(self.p2), dtype=torch.float32, device=device),
                        torch.tensor(np.log(self.p3), dtype=torch.float32, device=device),
                        torch.tensor(np.log(self.p4), dtype=torch.float32, device=device)]
        self.locs = mus  # list of locations for each of these gaussians
        self.covs = sigmas  # list of covariance matrices for each of these gaussians
        self.dists = [torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i]) for i in range(len(self.locs))]  # list of distributions for each of them

        self.x_edges = lambda x1: np.linspace(-10, 10, x1)
        self.y_edges = lambda y1: np.linspace(-8, 8, y1)

    def get_energy_function(self):
        def fn(x):
            return -self.log_density(x)
        return fn

    def log_density(self, X):
        log_p_1 = (self.log_pis[0] + self.dists[0].log_prob(X)).view(-1, 1)
        log_p_2 = (self.log_pis[1] + self.dists[1].log_prob(X)).view(-1, 1)
        log_p_3 = (self.log_pis[2] + self.dists[2].log_prob(X)).view(-1, 1)
        log_p_4 = (self.log_pis[3] + self.dists[3].log_prob(X)).view(-1, 1)
        log_p_1_2_3_4 = torch.cat([log_p_1, log_p_2, log_p_3, log_p_4], dim=-1)
        log_density = torch.logsumexp(log_p_1_2_3_4, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density

    
class distribution(object):
    def __init__(self,dist_name,device='cpu', mus=0, covs=0, pis=0, template=0):
        #the object that manage all distributions
        #Each distirbution has Ex (energy function), sampling function if available, as well as mean and std for the exact sample for ESS calculation
        self.device = device
        
        if dist_name == 'moon':
            #moon distribution is 2d only...
            #too easy though, not very interesting
            self.x_edges = lambda x1: np.linspace(-4, 4, x1)
            self.y_edges = lambda y1: np.linspace(-7, 5, y1)
            self.Ex = lambda x : moon_E(x)
            self.sample = lambda bs : moon_sample(bs)

        
        elif dist_name == 'moon2':
            #moon distribution is 2d only...
            #too easy though, not very interesting
            self.x_edges = lambda x1: np.linspace(-4, 4, x1)
            self.y_edges = lambda y1: np.linspace(-7, 5, y1)
            self.Ex = lambda x : moon_E2(x)
            self.sample = lambda bs : moon_sample(bs)


        elif dist_name == 'GMM':
            self.mus = mus
            self.covs = covs
            self.pis = pis
            self.x_edges = lambda x1: np.linspace(-10, 10, x1)
            self.y_edges = lambda y1: np.linspace(-8, 8, y1)
            self.Ex = lambda x: four_gmm(x,self.mus,self.covs,self.pis,self.device)
            self.sample = lambda n: gmm_sample(self.pis,self.mus,self.covs,n)


        elif dist_name == 'toy':
            self.template = template
            self.x_edges = lambda x1: np.linspace(-torch.pi,torch.pi,x1+1)
            self.y_edges = lambda y1: np.linspace(0.8,1.8,y1+1)
            self.Ex = lambda x: log_likelihood(self.template, x)

        
        else:
            print('dataset name not recognized')
