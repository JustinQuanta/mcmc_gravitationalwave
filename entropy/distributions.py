import numpy as np
import torch
import torch.nn as nn
#References: 



def gaussian_E(x):
    return x.pow(2).flatten(1).sum(1)/2
def gaussian_sample(N_samples,d=100):
    return torch.randn((N_samples,d))
    
#neal's gaussian target with linear scaling of std...
#mass_vect = torch.linspace(0.01,1,100).view(1,-1)
def diag_gaussian_E(x,mass_vect):
    mass_vect = mass_vect.to(x.device)
    x = x/mass_vect
    return x.pow(2).flatten(1).sum(1)/2

def diag_gaussian_sample(N_samples,mass_vect,d):
    return mass_vect*torch.randn((N_samples,d))

#nd gaussian functions used in SCG
def nd_gaussian_E(x,cov_mtx_i):
    #input inverse of covariance
    cov_mtx_i = cov_mtx_i.to(x.device)
    return (x*torch.matmul(x,cov_mtx_i)).sum(1)/2

def nd_gaussian_sample(N_samples,mass_mtx_i,d=2):
    x = torch.randn((N_samples,d))
    x = torch.matmul(x,mass_mtx_i)
    return x

#funnel distribution
def funnel_E(x,d=20,sig=3):
    #sig being the variance of first dimension
    E_y = x[:,0].pow(2)/(2*sig**2)
    if d==2:
        E_x = x[:,1].pow(2)*(x[:,0].clamp(-25,25).exp())/2 - x[:,0]/2
    else:
        E_x = x[:,1:].pow(2).flatten(1).sum(1)*(x[:,0].clamp(-25,25).exp())/2 - ((d-1)/2)*x[:,0]
    return  E_y + E_x

def funnel_sample(N_samples,d=20,sig=3,clip_y=11):
    #sample from Nd funnel distribution
    y = (sig*torch.randn((N_samples,1))).clamp(-clip_y,clip_y)
    x = torch.randn((N_samples,d-1))*(-y/2).exp()
    return torch.cat((y,x),dim=1)

#moon shaped distribution...2D only?
#a little too easy
def moon_E(x):
    E_y = x[:,0].pow(2)/2
    E_x = (x[:,1]- x[:,0].pow(2) + 4).pow(2)/2
    return E_y + E_x

def moon_sample(N_samples):
    y = torch.randn((N_samples,1))
    x = torch.randn((N_samples,1)) + y.pow(2) - 4
    return torch.cat((y,x),dim=1)
    
#UCI classification datasets
def load_regression_data(data_path,features,convert_lable=True):
    #features is number of features, including the label dimension...
    import pandas as pd
    df = pd.read_table(data_path,header=None,sep=r"\s*",engine='python')
    data = torch.from_numpy(df.iloc[0:,[i for i in range(features)]].values).float()
    X = data[:,0:-1]
    Y = data[:,-1:]
    if convert_lable:
        Y[Y==1] = -1
        Y[Y==2] = 1 
    else:
        Y[Y==0] = -1
        
    X = (X - X.mean(dim=0,keepdim=True))/X.std(dim=0,keepdim=True)
    X,Y = X.t(), Y.t()
   
    return X, Y

def LR_E(beta,X,Y,sig):
    #Energy function for logistic regression
    #beta is parameter vector [batch, Nfeatures + 1], first element is treated as offset as in NUTS paper.
    #X is design matrix [N, Nfeatures], Y the label -1 or 1 [N,].
    #sig is prior distribution variance
    X = X.to(beta.device)
    Y = Y.to(beta.device)
    Eprior = beta.pow(2).sum(1)/(2*sig**2)
    beta0, beta1 = beta[:,0:1], beta[:,1:]
    Edata = ((torch.matmul(beta1,X) + beta0)*Y).sigmoid().log().sum(dim=1).mul(-1)
    return Eprior + Edata

class general_GMM(nn.Module):
    def __init__(self, mus, covs, pis):
        super(general_GMM, self).__init__()
        assert len(mus) == len(covs)
        assert sum(pis) == 1.0

        self.p = [pis[i] for i in range(len(pis))]
        self.log_pis = [torch.tensor(np.log(self.p[i]), dtype=torch.float32)
                        for i in range(len(pis))]
        self.locs = mus  # list of locations for each of these gaussians
        self.covs = covs  # list of covariance matrices for each of these gaussians
        self.dists = [torch.distributions.MultivariateNormal(loc=self.locs[i],
                                                             covariance_matrix=self.covs[i])
                      for i in range(len(self.locs))]

    def get_energy_function(self):
        def fn(x):
            return -self.log_density(x)
        return fn

    def get_samples(self, n):
        # Compute the desired number of samples (as floats) for each component.
        n_float = [n * self.p[i] for i in range(len(self.p))]
        # Take the integer part.
        n_int = [int(val) for val in n_float]
        # Compute how many samples have been assigned.
        assigned = sum(n_int)
        # Compute the remainder needed to reach exactly n.
        remainder = n - assigned
        if remainder > 0:
            # Compute the fractional parts.
            fractional = [n_float[i] - n_int[i] for i in range(len(n_float))]
            # Assign the remaining samples to the components with the highest fractional parts.
            for _ in range(remainder):
                idx = np.argmax(fractional)
                n_int[idx] += 1
                fractional[idx] = 0  # Set to zero so that we don't assign extra here.
        # Now, n_int sums exactly to n.
        samples = [self.dists[i].sample((n_int[i],)) for i in range(len(n_int))]
        samples0 = torch.cat(samples)
        return samples0

    def log_density(self, X):
        log_p = [(self.log_pis[i] + self.dists[i].log_prob(X)).view(-1, 1)
                 for i in range(len(self.p))]
        log_ps = torch.cat([log_p[i] for i in range(len(self.p))], dim=-1)
        log_density = torch.logsumexp(log_ps, dim=1)
        return log_density


class distributions(object):
    def __init__(self,dist_name,d=100,funnel_sig=3,RW_ita=1e-2,gpus=[0], mus=0, covs=0, pis=0):
        #the object that manage all distributions
        #Each distirbution has Ex (energy function), sampling function if available, as well as mean and std for the exact sample for ESS calculation
        self.d = d
        
        if dist_name == 'gaussian':
            self.Ex = lambda x : gaussian_E(x)
            self.sample = lambda bs : gaussian_sample(bs,d=self.d)
            self.mean = torch.tensor([0])
            self.std = torch.tensor([1])
            
        elif dist_name == 'Neals gaussian':
            #simple version of Neals gaussian with diagnoal covariance between 0.01 and 1 linearly distributed
            self.mass_vect = torch.linspace(0.01,1,self.d).view(1,-1)
            self.Ex = lambda x : diag_gaussian_E(x,self.mass_vect)
            self.sample = lambda bs : diag_gaussian_sample(bs,self.mass_vect,self.d)
            self.mean = torch.tensor([0])
            self.std = self.mass_vect
            
        elif dist_name == 'ICG':
            #50d ill conditioned gaussian in L2HMC paper, with log-linear spacing of variances between 1e-2 and 1e2
            self.mass_vect = torch.linspace(-np.log(10),np.log(10),self.d).exp().view(1,-1)
            self.Ex = lambda x : diag_gaussian_E(x,self.mass_vect)
            self.sample = lambda bs : diag_gaussian_sample(bs,self.mass_vect,self.d)
            self.mean = torch.tensor([0])
            self.std = self.mass_vect
        
        elif dist_name == 'NDICG':
            #100d non-diagonal ill conditioned gaussian example in IAF neural transport paper
            rng = np.random.RandomState(seed=10)
            eigenvalues = np.sort(rng.gamma(shape=0.5, scale=1.,size=self.d)).astype(np.float32)
            q, _ = np.linalg.qr(rng.randn(self.d, self.d))
            covariance = (q * eigenvalues**-1).dot(q.T)
            self.cov_mtx_i = torch.from_numpy(np.linalg.inv(covariance)).float()
            self.mass_mtx_i = torch.from_numpy(np.linalg.cholesky(covariance).T).float()
            
            self.Ex = lambda x : nd_gaussian_E(x,self.cov_mtx_i)
            self.sample = lambda bs : nd_gaussian_sample(bs,self.mass_mtx_i,self.d)
            self.mean = torch.tensor([0])
            self.std = torch.from_numpy(np.sqrt(np.diag(covariance))).float()
            
        elif dist_name == 'SCG':
            #strongly correlated gaussian example in L2HMC paper, only in 2d
            #in paper it said 1e2 and 1e-2 but in the released code its says 1e2 and 1e-1...
            #performance form released code seem to match paper result so use 1e2 and 1e-1 here...
            self.d = 2
            self.cov_mtx = torch.tensor([[50.05, -49.95], [-49.95, 50.05]])
            self.cov_mtx_i = self.cov_mtx.inverse()
            self.mass_mtx_i = torch.from_numpy(np.linalg.cholesky(self.cov_mtx.numpy())).t()
            
            self.Ex = lambda x : nd_gaussian_E(x,self.cov_mtx_i)
            self.sample = lambda bs : nd_gaussian_sample(bs,self.mass_mtx_i,self.d)
            self.mean = torch.tensor([0])
            self.std = 5*(2*torch.ones((1,2))).sqrt()
            
        elif dist_name == 'funnel':
            self.funnel_sig = funnel_sig
            self.Ex = lambda x : funnel_E(x,d=self.d,sig=self.funnel_sig)
            self.sample = lambda bs : funnel_sample(bs,d=self.d,sig=self.funnel_sig)
            self.mean = torch.tensor([0])
            if funnel_sig == 3:
                self.std = 7.4*torch.ones((self.d))
                self.std[0] = 3
            elif funnel_sig == 1:
                self.std = 1.622*torch.ones((self.d))
                self.std[0] = 1
            
        elif dist_name == 'moon':
            #moon distribution is 2d only...
            #too easy though, not very interesting
            self.Ex = lambda x : moon_E(x)
            self.sample = lambda bs : moon_sample(bs)

        elif dist_name == "GMM":
            self.mus = mus
            self.covs = covs
            self.pis = pis
            gmm_instance = general_GMM(mus, covs, pis)
            self.Ex = gmm_instance.get_energy_function()
            self.sample = lambda n: gmm_instance.get_samples(n)
            
        elif dist_name == 'toy':
            # The toy likelihood model is defined for 2D inputs,
            # where the two dimensions represent parameters (e.g. phase and frequency).
            # The likelihood is based on the squared error between a generated waveform 
            # and a fixed template waveform.
            #
            # We first define a time vector t and model parameters for the waveform.
            t = torch.arange(0, 100, 0.1)  # time vector: from 0 to 100 in steps of 0.1
            phase = 0.0                   # fixed phase offset used to generate the template
            freq = 1.0                    # fixed frequency for the template waveform
            dfreq = 0.1                   # additional frequency modulation term
            ddfreq = 0.001                # second-order frequency modulation term
            
            # Generate the template waveform using the fixed parameters.
            # Here we use NumPy to compute the sine values and then convert to a torch tensor.
            template_np = np.array([np.sin(phase + freq * ti + dfreq * ti**2 + ddfreq * ti**3)
                                      for ti in t])
            template = torch.tensor(template_np, dtype=torch.float32)
            
            # Define the energy function for the toy model.
            # x is assumed to be a tensor of shape (batch_size, 2), where:
            #   x[:, 0] represents the phase parameter and x[:, 1] the frequency parameter.
            # For each sample, we generate a waveform h(x) = sin(phase + freq*t + dfreq*t^2 + ddfreq*t^3),
            # then compute the squared error from the fixed template, summed over time,
            # and scaled by 0.5.
            def toy_E(x):
                # x: [batch_size, 2]
                # Unsqueeze parameters to shape [batch_size, 1] so that they can broadcast with t.
                phase_val = x[:, 0].unsqueeze(1)       # shape: (batch_size, 1)
                freq_val  = x[:, 1].unsqueeze(1)         # shape: (batch_size, 1)
                # t is shape (T,), so unsqueeze to (1, T) for broadcasting.
                t_b = t.unsqueeze(0)                     # shape: (1, T)
                # Compute the waveform for each sample.
                h = torch.sin(phase_val + freq_val * t_b + dfreq * t_b**2 + ddfreq * t_b**3)
                # template has shape (T,), so unsqueeze it to (1, T)
                template_exp = template.unsqueeze(0)
                # Compute the squared error over time and sum; then multiply by 0.5.
                diff = h - template_exp
                E = 0.5 * (diff.pow(2).sum(1))
                return E
        
            # Define a sampling function for the toy model.
            # Here, we simply sample the 2D parameters (phase, frequency) from a standard Gaussian.
            def toy_sample(bs):
                # Phase: sample uniformly between -pi and pi.
                phase_sample = torch.empty(bs, 1).uniform_(-torch.pi, torch.pi)
                # Frequency: sample uniformly between 0.8 and 1.8.
                freq_sample = torch.empty(bs, 1).uniform_(0.8, 1.8)
                return torch.cat((phase_sample, freq_sample), dim=1)
            
            # Set the energy function, sampling function, and basic statistics.
            self.Ex = lambda x: toy_E(x)
            self.sample = lambda bs: toy_sample(bs)
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32)
            self.std = torch.tensor([1.0, 1.0], dtype=torch.float32)
        
        elif dist_name == 'UCI German':
            #German data 25d
            #Linear regression dataset doesn't have exact sample so use Gaussian as initialization
            data_path = 'UCI_data/german.data-numeric'
            self.d = 25
            self.sig = 10 #variance of prior
            self.X, self.Y = load_regression_data(data_path,25,convert_lable=True)
            self.Ex = lambda x : LR_E(x,self.X,self.Y,self.sig)
            self.sample = lambda bs : gaussian_sample(bs,d=25)
            self.mean = torch.tensor([-1.1880, -0.7250,  0.4188, -0.4144,  0.1206, -0.3568, -0.1697, -0.1490,
         0.0191,  0.1849, -0.1032, -0.2239,  0.1299,  0.0284, -0.1359, -0.2556,
         0.2780, -0.2965,  0.3064,  0.2698,  0.1225, -0.0557, -0.0865, -0.0305,
        -0.0212])
            self.std = torch.tensor([0.2797, 0.2441, 0.1684, 0.1366, 0.1739, 0.1919, 0.2086, 0.1718, 0.1541,
        0.1624, 0.1466, 0.1640, 0.1671, 0.1550, 0.1592, 0.3534, 0.1918, 0.1804,
        0.2100, 0.1836, 0.1912, 0.1985, 0.1586, 0.1808, 0.1821])
                       
        elif dist_name == 'UCI Australian':
            #15d Australian data
            data_path = 'UCI_data/australian.dat'
            self.d = 15
            self.sig = 10 #variance of prior
            self.X, self.Y = load_regression_data(data_path,15,convert_lable=False)
            self.Ex = lambda x : LR_E(x,self.X,self.Y,self.sig)
            self.sample = lambda bs : gaussian_sample(bs,d=15)
            self.mean = torch.tensor([-1.9549e-01, -3.7447e-05,  5.9428e-03, -1.8608e-01,  3.8344e-01,
         7.5737e-01,  8.1806e-02,  2.6807e-01,  1.7463e+00,  1.6302e-01,
         6.8301e-01, -1.5089e-01,  1.5342e-01, -3.5001e-01,  2.6625e+00],)
            self.std = torch.tensor([0.1825, 0.1316, 0.1409, 0.1361, 0.1338, 0.1545, 0.1503, 0.1667, 0.1556,
        0.1751, 0.2859, 0.1323, 0.1278, 0.1490, 0.8491])
                        
        elif dist_name == 'UCI Heart':
            data_path = 'UCI_data/heart.dat'
            self.d = 14
            self.sig = 10 #variance of prior
            self.X, self.Y = load_regression_data(data_path,14)
            self.Ex = lambda x : LR_E(x,self.X,self.Y,self.sig)
            self.sample = lambda bs : gaussian_sample(bs,d=14)
            self.mean = torch.tensor([-0.2656, -0.1787,  0.7862,  0.7384,  0.4948,  0.4155, -0.3108,  0.3319,
        -0.5338,  0.4196,  0.4362,  0.2894,  1.2079,  0.7211])
            self.std = torch.tensor([0.2076, 0.2446, 0.2662, 0.2163, 0.2150, 0.2244, 0.2132, 0.2072, 0.2583,
        0.2130, 0.2728, 0.2514, 0.2683, 0.2172])
            
        else:
            print('dataset name not recognized')
                                        
                                        
                                