import numpy as np
import arviz
import functools
from statsmodels.tsa.stattools import acf as autocorr
import statsmodels.api as sm
from mcmc import mcmc_algos
from scipy.stats import entropy

def neff(arr):
    n = len(arr)
    acf = autocorr(arr, nlags=n, fft=True)
    sums = 0
    for k in range(1, len(acf)):
        sums = sums + (n-k)*acf[k]/n

    return n/(1+2*sums)

def acf(x):
    acf = sm.tsa.acf(x,nlags=len(x))
    truncate = next(i for i,j in enumerate(acf) if j<0) # not so good for small n
    
    return acf[:truncate+1]

def evaluate(t,x,y,z,e,para1,n_sample,name):        
    for i in range(len(para1)):
        for j in range(n_sample):
            per_accept = np.array((x[i][j]/len(y[i][j]))*100)
            print('Given parameter {} = {}, {:.2f} % accepted overall, Time taken: {:.3f}s, \nKL Divergence: {:.4f} and ESS for x0: {:.3f}, x1: {:.3f}'
                  .format(name, para1[i], np.mean(per_accept), t[i][j], z[i][j], e[i][j][0], e[i][j][1]))

def evaluate2(t,x,y,z,e,para1,para2,n_sample):
    counter = 0
    for i in range(len(para1)):
        for j in range(len(para2)):
            if para1[i] < 5 or ((para1[i] == 5 and para2[j] < 0.5) or (para1[i] == 10 and para2[j] == 0.1)):
                counter += 1
                for k in range(n_sample):
                    per_accept = np.array(x[counter-1]*100)
                    print('Given parameter path length: {} and step size: {:.3f}, {:.2f} % accepted overall, Time taken: {:.3f}s, \nKL Divergence: {:.4f} and ESS for x0: {:.3f}, x1: {:.3f}'
                          .format(para1[i], para2[j], per_accept, t[counter-1], z[counter-1][k], e[counter-1][0][k][0], e[counter-1][0][k][1]))  

def kl_div(hist_0, x_edges, y_edges, sample, burnin):
    hist_A, _, _ = np.histogram2d(sample[burnin:,0], sample[burnin:,1], bins=[x_edges, y_edges], density=True)

    # Normalize histograms to get joint probability distributions
    prob_A = hist_A / np.sum(hist_A)
    prob_0 = hist_0 / np.sum(hist_0)
    
    # Flatten the probability matrices
    prob_A_flat = prob_A.flatten()
    prob_0_flat = prob_0.flatten()

    # Add a small value to avoid division by zero in KL divergence calculation
    epsilon = 1e-10
    prob_A_flat += epsilon
    prob_0_flat += epsilon

    # Calculate KL divergence (1st argument is P, 2nd argument is Q)
    return entropy(prob_A_flat, prob_0_flat)