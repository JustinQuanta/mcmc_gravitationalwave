import collections
import torch
import numpy as np
import torch.nn as nn
from scipy.stats import multivariate_normal, ortho_group
import pdb
torchType = torch.float32
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def quadratic_gaussian(x, mu, S):
    return torch.diag(0.5* (x-mu) @ S @ (x-mu).T)
    

class Gaussian(nn.Module):
    def __init__(self, mu, sigma, device='cpu'):
        super(Gaussian, self).__init__()
        self.device = device
        self.mu = mu.type(torchType)
        self.sigma = sigma.type(torchType)
        self.target_distr = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu,covariance_matrix=self.sigma)

    def get_energy_function(self):
        # def fn(x, *args, **kwargs):
        #     return quadratic_gaussian(x.type(torchType), self.mu, self.i_sigma)
        def fn(x):
            return -self.target_distr.log_prob(x)
        return fn

    def get_samples(self, n):
        '''
        Sampling is broken in numpy for d > 10
        '''
        return self.target_distr.sample((n, )).cpu().detach().numpy()

    def log_density(self, X):
        # pdb.set_trace()
        return self.target_distr.log_prob(X)


class GMM(nn.Module):
    def __init__(self, mus, sigmas, pis, device='cpu'):
        super(GMM, self).__init__()
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.p = pis[0]  # probability of the first gaussian (1-p for the second)
        self.log_pis = [torch.tensor(np.log(self.p), dtype=torch.float32, device=device),
                        torch.tensor(np.log(1 - self.p), dtype=torch.float32,
                                     device=device)]  # LOGS! probabilities of Gaussians
        self.locs = mus  # list of locations for each of these gaussians
        self.covs = sigmas  # list of covariance matrices for each of these gaussians
        self.dists = [torch.distributions.MultivariateNormal(loc=self.locs[0], covariance_matrix=self.covs[0]),
                      torch.distributions.MultivariateNormal(loc=self.locs[1], covariance_matrix=self.covs[
                          1])]  # list of distributions for each of them

    def get_energy_function(self):
        def fn(x):
            return -self.log_density(x)
        return fn

    def get_samples(self, n):
        n_first = int(n * self.p)
        n_second = n - n_first
        samples_1 = self.dists[0].sample((n_first,))
        samples_2 = self.dists[1].sample((n_second,))
        samples = torch.cat([samples_1, samples_2])
        return samples.cpu().detach().numpy()

    def log_density(self, X):
        log_p_1 = (self.log_pis[0] + self.dists[0].log_prob(X)).view(-1, 1)
        log_p_2 = (self.log_pis[1] + self.dists[1].log_prob(X)).view(-1, 1)
        log_p_1_2 = torch.cat([log_p_1, log_p_2], dim=-1)
        log_density = torch.logsumexp(log_p_1_2, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density


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

    def get_energy_function(self):
        def fn(x):
            return -self.log_density(x)
        return fn

    def get_samples(self, n):
        n_first = int(n * self.p1)
        n_second = int(n * self.p2)
        n_third = int(n * self.p3) 
        n_fourth = int(n * self.p4)
        samples_1 = self.dists[0].sample((n_first,))
        samples_2 = self.dists[1].sample((n_second,))
        samples_3 = self.dists[2].sample((n_third,))
        samples_4 = self.dists[3].sample((n_fourth,))
        samples = torch.cat([samples_1, samples_2, samples_3, samples_4])
        idx = torch.randperm(samples.shape[0])
        
        return samples[idx].view(samples.size()).cpu().detach()

    def log_density(self, X):
        log_p_1 = (self.log_pis[0] + self.dists[0].log_prob(X)).view(-1, 1)
        log_p_2 = (self.log_pis[1] + self.dists[1].log_prob(X)).view(-1, 1)
        log_p_3 = (self.log_pis[2] + self.dists[2].log_prob(X)).view(-1, 1)
        log_p_4 = (self.log_pis[3] + self.dists[3].log_prob(X)).view(-1, 1)
        log_p_1_2_3_4 = torch.cat([log_p_1, log_p_2, log_p_3, log_p_4], dim=-1)
        log_density = torch.logsumexp(log_p_1_2_3_4, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density


class general_GMM(nn.Module):
    def __init__(self, mus, sigmas, pis, device='cpu'):
        super(general_GMM, self).__init__()
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.p = [pis[i] for i in range(len(pis))]
        self.log_pis = [torch.tensor(np.log(self.p[i]), dtype=torch.float32, device=device) for i in range(len(pis))]
        self.locs = mus  # list of locations for each of these gaussians
        self.covs = sigmas  # list of covariance matrices for each of these gaussians
        self.dists = [torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i]) for i in range(len(self.locs))]  # list of distributions for each of them

    def get_energy_function(self):
        def fn(x):
            return -self.log_density(x)
        return fn

    def get_samples(self, n):
        n1 = [int(n * self.p[i]) for i in range(len(self.p))]
        samples = [self.dists[i].sample((n1[i],)) for i in range(len(self.p))]
        samples0 = torch.cat([samples[i] for i in range(len(self.p))])
        return samples0.cpu().detach().numpy()

    def log_density(self, X):
        log_p = [(self.log_pis[i] + self.dists[i].log_prob(X)).view(-1, 1) for i in range(len(self.p))]
        log_ps = torch.cat([log_p[i] for i in range(len(self.p))], dim=-1)
        log_density = torch.logsumexp(log_ps, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density


class Moon(nn.Module):
    def __init__(self, device='cpu'):
        super(Moon, self).__init__()
        self.device = device

    def moon_E(self, x):
        E_y = x[:, 0].pow(2) / 2
        E_x = (x[:, 1] - x[:, 0].pow(2) + 4).pow(2) / 2
        return E_y + E_x

    def moon_sample(self, N_samples):
        y = torch.randn((N_samples, 1), device=self.device)
        x = torch.randn((N_samples, 1), device=self.device) + y.pow(2) - 4
        return torch.cat((y, x), dim=1)

    def get_energy_function(self):
        def fn(x):
            return self.moon_E(x)
        return fn

    def get_samples(self, n):
        return self.moon_sample(n).cpu().detach().numpy()

    def log_density(self, X):
        return -self.moon_E(X)


class toy(nn.Module):
    def __init__(self, template, t, phase, freq, dfreq, ddfreq, device='cpu'):
        super(toy, self).__init__()
        self.device = device
        self.template = template
        self.t = t
        self.phase = phase
        self.freq = freq
        self.dfreq = dfreq
        self.ddfreq = ddfreq

    def h(self,x):
        return torch.sin(x[0][0] + x[0][1]*self.t + self.dfreq*self.t**2 + self.ddfreq*self.t**3)

    def toy_E(self, x):
        difference = self.h(x) - self.template
        norm_squared = torch.linalg.norm(difference, axis=-1)**2
        return -0.5 * norm_squared

    def get_energy_function(self):
        def fn(x):
            return self.toy_E(x)
        return fn
    
    def log_density(self, X):
        return self.toy_E(X)

    def manual_grad_log_likelihood(self):
        def gn(x):
            difference = self.h(x) - self.template
    
            # Partial derivative w.r.t phase
            grad_phase = torch.mean(difference * torch.cos(self.phase + self.freq*self.t + self.dfreq*self.t**2 + self.ddfreq*self.t**3)) 
        
            # Partial derivative w.r.t frequency
            grad_freq = torch.mean(difference * torch.cos(self.phase + self.freq*self.t + self.dfreq*self.t**2 + self.ddfreq*self.t**3) * self.t)
        
            return torch.tensor([grad_phase, grad_freq]) 
        return gn