import torch
from funcs.log_t import logt
from funcs.kl_div import kl
from funcs.gaussian import normal_prob,log_normal_prob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as pl
import matplotlib.colors as colors
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import geopy.distance
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
import sklearn
import scikitplot as skplt
import copy


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    #plt.show()

def compute_mmd(args,bnn,test_loader):
    plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],'font.size': 16,'lines.linewidth':2})
    reconstructed=np.vstack(bnn.generate(len(test_loader.dataset),1))
    test_data=test_loader.dataset.X
    score=maximum_mean_discrepancy(reconstructed, test_data, kernel=sklearn.metrics.pairwise.rbf_kernel)
    return score

def test_latent_space(args, bnn, test_loader):
    z = []
    mus = []
    log_vars = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(args.device)
            latent, mu, log_var = bnn.get_latent(data)
            z.append(latent.detach().numpy())
            mus.append(mu.numpy())
            log_vars.append(log_var.numpy())
        z=np.vstack(z)
        mus=np.vstack(mus)
        log_vars=np.vstack(log_vars)
        for i in range(0,5):
            plt.hist(mus[:,i], bins='auto',density=True)
            x = np.arange(-5, 5, .01)
            mean=0
            variance=1
            f = np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))
            plt.plot(x,f)
            #plt.show()
        return 0

def auroc(args, bnn, test_loader):
    scores = []
    with torch.no_grad():
        for i in range(0, int(len(test_loader.dataset.X) / 100)):
            data = torch.tensor(test_loader.dataset.X[i * 100:(i + 1) * 100, :])
            data = data.to(args.device)
            probs,mus,log_vars = bnn(data,args.m_te)
            if(not args.BayesianDec):
                probs = torch.vstack([torch.mean((p - data) ** 2, axis=1) for p in probs]).numpy()
                scores.append(np.mean(probs,axis=0))
            else:
                probs=torch.vstack([torch.exp(-torch.mean((p-data)**2,axis=1)/(2*args.sigma**2)) for p in probs]).numpy()
                probs=np.mean(probs,axis=0)
                scores.append(probs)
    if (not args.BayesianDec):
        scores=np.hstack(scores)
        scores=scores/np.max(scores)
    else:
        scores=np.hstack(scores)/args.m_te
    scores[np.isnan(scores)]=1
    scores[scores==np.inf] = 0
    labels=test_loader.dataset.OOD
    labels=labels[0:scores.shape[0]]
    fper, tper, thresholds =  sklearn.metrics.roc_curve(labels, scores)
    plot_roc_curve(fper, tper)
    print(sklearn.metrics.roc_auc_score(labels, scores))
    return fper, tper, sklearn.metrics.roc_auc_score(labels, scores)

def test(args, bnn, test_loader,epoch=0):
    nll,mse,nllt = 0,0,0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data= data.to(args.device)
            probs,mus,log_vars = bnn(data,args.m_te)
            if(batch_idx==0):
                data_numpy=np.reshape(data.cpu().numpy(),(args.te_batch_size,32,32))
                proposed_numpy = [np.reshape(p.cpu().numpy(), (args.te_batch_size, 32, 32)) for p in probs]
                for j in range(0,10):
                    fig, axs = plt.subplots(1, args.m_te+1)
                    axs[0].imshow(data_numpy[j,:,:])
                    for i in range(0,args.m_te):
                        axs[i+1].imshow(proposed_numpy[i][j,:,:])
                    #plt.show()
                    plt.savefig('reconstructions_res/Reconstructed_EPOCH'+str(epoch)+'.png')
                    plt.clf()
            log_p_x = torch.stack([log_normal_prob(data, p, args.sigma) for p in probs])
            log_t_avg_prob = logt(args.t, torch.sum(torch.exp(log_p_x-np.log(args.m_te)), 0))
            log_avg_prob = torch.logsumexp(torch.add(log_p_x, -np.log(args.m_te)), axis=0)
            nll += torch.mean(log_avg_prob)
            nllt += torch.mean(log_t_avg_prob)
            mse += torch.mean(torch.mean(torch.stack([torch.sqrt(torch.sum(torch.pow(p - torch.squeeze(data), 2), axis=1)) for p in probs]), 0))
        test_nll = nll / len(test_loader.dataset)
        test_nllt = nllt / len(test_loader.dataset)
        test_mse = mse / len(test_loader.dataset)
        return test_nll.cpu().data.numpy(), test_nllt.cpu().data.numpy(), test_mse.cpu().data.numpy()

def maximum_mean_discrepancy(x, y, kernel=sklearn.metrics.pairwise.rbf_kernel ):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.

    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

    where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

    Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
    """
    gamma=0.1
    cost = np.mean(kernel(x, x,gamma=gamma))
    cost += np.mean(kernel(y, y,gamma=gamma))
    cost -= 2 * np.mean(kernel(x, y,gamma=gamma))
    # We do not allow the loss to become negative.
    print('MMD:')
    print(cost)
    return cost
