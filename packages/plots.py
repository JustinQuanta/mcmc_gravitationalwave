import matplotlib.pyplot as plt
import numpy as np
import utils
from scipy.stats import multivariate_normal

c_true, c_contour = 'purple', '0.75'

def sampling(name, n, dist, num_bin, colors):
    S = dist.sample(n)
    fig = plt.figure()
    plt.title('{}'.format(name))
    plt.plot(S[:, 0], S[:, 1], 'o', alpha=0.4, color=colors)
    plt.axis('equal')
    plt.show()
    hist_0, _, _ = np.histogram2d(S[:, 0], S[:, 1], bins=[dist.x_edges(num_bin), dist.y_edges(num_bin)], density=True)
    return S, hist_0

def plot_compare(name, dist, num_bin, theory, sample, c):
    for i in range(len(c)):
        # Plot the histograms
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist2d(theory[:, 0], theory[:, 1], bins=[dist.x_edges(num_bin), dist.y_edges(num_bin)], density=True)
        plt.colorbar()
        plt.title('Theoretical {}'.format(name))
            
        plt.subplot(1, 2, 2)
        plt.hist2d(sample[i][:,:,0].flatten(), sample[i][:,:,1].flatten(), bins=[dist.x_edges(num_bin), dist.y_edges(num_bin)], density=True)
        plt.colorbar()
        plt.title('2D Distribution with parameter(s): {:.3f}'.format(c[i]))
        plt.show()

def plot_compare2(name, theory, sample, c, phase_t, freq_t, phase, freq):
    for i in range(len(c)):
        # Plot the histograms
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.contourf(phase_t, freq_t, theory, 20, cmap='viridis')
        plt.colorbar()
        plt.title('Theoretical {} model with phase: {:.3f} and freq: {:.1f}'.format(name, phase, freq))
            
        plt.subplot(1, 2, 2)
        plt.hist2d(sample[i][:,:,0].flatten(), sample[i][:,:,1].flatten(), bins=[phase_t, freq_t], density=True)
        plt.colorbar()
        plt.title('2D Distribution with parameter(s): {:.3f}'.format(c[i]))
        plt.show()


def plot_compare3(name, dist, num_bin, theory, sample, para1, para2):
    counter = 0
    for i in range(len(para1)):
        for j in range(len(para2)):
            counter += 1
            # Plot the histograms
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist2d(theory[:, 0], theory[:, 1], bins=[dist.x_edges(num_bin), dist.y_edges(num_bin)], density=True)
            plt.colorbar()
            plt.title('Theoretical {}'.format(name))
                
            plt.subplot(1, 2, 2)
            plt.hist2d(sample[counter-1][:,:,0].flatten(), sample[counter-1][:,:,1].flatten(), bins=[dist.x_edges(num_bin), dist.y_edges(num_bin)], density=True)
            plt.colorbar()
            plt.title('2D Distribution with path length: {:.3f} and step size: {:.3f}'.format(para1[i],para2[j]))
            plt.show()


def auto_corr_plot(name, sample, c):
    for i in range(len(c)):
        plt.figure()
        for j in range(len(sample[0][:,0,0])):
            plt.plot(utils.acf(sample[i][j,:,0]), label='x0', color='blue')
            plt.plot(utils.acf(sample[i][j,:,1]), label='x1', color='red')
            plt.ylim([0,1])
            plt.title('AC plot of {} with parameter(s) {:.3f}'.format(name, c[i]))
            plt.legend()
        plt.show()

def auto_corr_plot2(name, sample, para1, para2):
    counter = 0
    for i in range(len(para1)):
        for j in range(len(para2)):
            counter +=1
            plt.figure()
            for k in range(len(sample[0][0,:,0])):
                plt.plot(utils.acf(sample[counter-1][:,k,0]), label='x0', color='blue')
                plt.plot(utils.acf(sample[counter-1][:,k,1]), label='x1', color='red')
                plt.ylim([0,1])
                plt.title('AC plot of {} with path length: {} and step size: {:.3f}'.format(name, para1[i],para2[j]))
                plt.legend()
    plt.show()


def plot_gaussian_contours(mus, covs, colors=['blue', 'red'], spacing=5,
        x_lims=[-4,4], y_lims=[-3,3], res=100):

    X = np.linspace(x_lims[0], x_lims[1], res)
    Y = np.linspace(y_lims[0], y_lims[1], res)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for i in range(len(mus)):
        mu = mus[i]
        cov = covs[i]
        F = multivariate_normal(mu, cov)
        Z = F.pdf(pos)
        plt.contour(X, Y, Z, spacing, colors=colors[0])

    return plt


def contour_compare(name, sample, c1, mus, covs):
    for i in range(len(c1)):
        plot_gaussian_contours(mus, covs, colors=[c_contour, c_contour], x_lims=[-12,12], y_lims=[-12,12], res=200)
        plt.plot(sample[i][:,:,0].flatten(), sample[i][:,:,1].flatten(), marker='o', alpha=0.6)
        plt.title('{} Distribution with parameter : {:.3f}'.format(name, c1[i]))
        plt.tight_layout()
        
        plt.show()

            
def contour_compare2(name, sample, c1, c2, mus, covs):
    counter = 0
    for i in range(len(c1)):
        for j in range(len(c2)):
            #if c1[i] < 5 or ((c1[i] == 5 or c1[i] == 10) and c2[j] < 0.5):
                plot_gaussian_contours(mus, covs, colors=[c_contour, c_contour], x_lims=[-12,12], y_lims=[-12,12], res=200)
                plt.plot(sample[counter][:,:,0].flatten(), sample[counter][:,:,1].flatten(), marker='o', alpha=0.6)
                plt.title('{} Distribution with path length: {:.3f} and step size: {:.3f}'.format(name, c1[i],c2[j]))
                plt.tight_layout()
                
                plt.show()
                counter += 1


class plots(object):
    def __init__(self,dist_name, dist=0, num_bin=0, theory=0, sample=0, mus=0, covs=0 , x_lims=0, y_lims=0, c1=0, c2=0, colors=0, name=0, phase_t=0, freq_t=0, phase=0, freq=0):
        self.dist = dist
        self.num_bin = num_bin
        self.theory = theory
        self.sample = sample
        self.phase_t = phase_t
        self.freq_t = freq_t
        self.phase = phase
        self.freq = freq
        self.c1 = c1
        self.c2 = c2

        if dist_name == 'theory':
            self.colors = colors
            self.theory = lambda x,n : sampling(x, n, self.dist, self.num_bin, self.colors)
        
        elif dist_name == 'compare':
            self.compare = lambda x : plot_compare(x, self.dist, self.num_bin, self.theory, self.sample, self.c1)
            self.compare2 = lambda x : plot_compare2(x, self.theory, self.sample, self.c1, self.phase_t, self.freq_t, self.phase, self.freq)
            self.compare3 = lambda x : plot_compare3(x, self.dist, self.num_bin, self.theory, self.sample, self.c1, self.c2)

        elif dist_name == 'ac':
            self.acplot = lambda x : auto_corr_plot(x, self.sample, self.c1)
            self.acplot2 = lambda x : auto_corr_plot2(x, self.sample, self.c1, self.c2)

        elif dist_name == 'contour':
            self.mus = mus
            self.covs = covs
            self.contourp = lambda x : contour_compare(x, self.sample, self.c1, self.mus, self.covs)
            self.contourp2 = lambda x : contour_compare2(x, self.sample, self.c1, self.c2, self.mus, self.covs)
        
        else:
            print('no such plotting!')
            