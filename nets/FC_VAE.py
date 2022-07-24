import torch
import torch.nn as nn
import torchbnn as bnn
import tensorflow_probability as tfp
import math
import numpy as np
import nets.mylayer as mylayer
tfd = tfp.distributions

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

class VAE(nn.Module):
    def __init__(self,BayesEnc=False,BayesianDec=False,dim=32):
        super(VAE, self).__init__()
        self.dim=dim
        self.hid_dim=16
        if(not BayesEnc):
            self.fcce = torch.nn.Linear(in_features=self.dim*self.dim, out_features=512)
            self.fcce1 = torch.nn.Linear(in_features=512, out_features=256)
            self.fcce2 = torch.nn.Linear(in_features=256, out_features=self.hid_dim*2)
        else:
            prior_sigma=.1
            self.fcce = bnn.BayesLinear(prior_mu=0, prior_sigma=.02, in_features=self.dim*self.dim, out_features=512)
            self.fcce1 = bnn.BayesLinear(prior_mu=0, prior_sigma=.02, in_features=512, out_features=256)
            self.fcce2 = bnn.BayesLinear(prior_mu=0, prior_sigma=.02, in_features=256, out_features=self.hid_dim*2)
            self.fcce.prior_sigma = prior_sigma
            self.fcce.prior_log_sigma = math.log(prior_sigma)
            self.fcce1.prior_sigma = prior_sigma
            self.fcce1.prior_log_sigma = math.log(prior_sigma)
            self.fcce2.prior_sigma = prior_sigma
            self.fcce2.prior_log_sigma = math.log(prior_sigma)
        if (not BayesianDec):
            self.fccd = torch.nn.Linear(in_features=self.hid_dim, out_features=256)
            self.fcc1d = torch.nn.Linear(in_features=256, out_features=512)
            self.fcc2d = torch.nn.Linear(in_features=512, out_features=self.dim*self.dim)
        else:
            prior_sigma = .1
            self.fccd = bnn.BayesLinear(prior_mu=0, prior_sigma=.01, in_features=self.hid_dim, out_features=256)
            self.fcc1d = bnn.BayesLinear(prior_mu=0, prior_sigma=.01, in_features=256, out_features=512)
            self.fcc2d = bnn.BayesLinear(prior_mu=0, prior_sigma=.01, in_features=512, out_features=self.dim*self.dim)
            self.fccd.prior_sigma = prior_sigma
            self.fccd.prior_log_sigma = math.log(prior_sigma)
            self.fcc1d.prior_sigma = prior_sigma
            self.fcc1d.prior_log_sigma = math.log(prior_sigma)
            self.fcc2d.prior_sigma = prior_sigma
            self.fcc2d.prior_log_sigma = math.log(prior_sigma)
        self.activ = nn.ELU()
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample


    def forward(self, x_input,m):
        x_input=x_input.type(torch.float32).view(x_input.shape[0] ,self.dim*self.dim)
        prob_list = []
        mu_list=[]
        log_var_list = []
        hid1 = self.activ(self.fcce(x_input))
        hid2 = self.activ(self.fcce1(hid1))
        hid3 = torch.flatten((self.fcce2(hid2)), start_dim=1, end_dim=- 1).view(x_input.shape[0] ,2,-1)
        mu = hid3[:, 0, :]  # the first feature values as mean
        log_var = hid3[:, 1, :]  # the other feature values as variance
        z = self.reparameterize(mu, log_var)
        for ind_m in range(m):
            hid_4=self.activ(self.fccd(z))
            hid5 = self.activ(self.fcc1d(hid_4))
            hid6 = self.fcc2d(hid5)
            out=hid6.view(x_input.shape[0],-1)
            prob_list.append(out)
            mu_list.append(mu)
            log_var_list.append(log_var)
        return prob_list,mu_list,log_var_list


    def get_latent(self, x_input):
        hid1 = self.activ(self.cnn1e(x_input))
        hid2 = self.activ(self.cnn2e(hid1))
        hid3 =  self.cnn3e(hid2).view(-1, 2,  self.dim)
        mu = hid3[:, 0, :]  # the first feature values as mean
        log_var = hid3[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return z,mu,log_var

    def generate(self, batch_size,m):
        prob_list = []
        mu=torch.tensor(0).repeat(batch_size,self.dim)
        log_var=torch.tensor(0).repeat(batch_size,self.dim)
        for ind_m in range(m):
            z = self.reparameterize(mu, log_var)
            hid4 = self.activ(self.cnn1d(z))
            hid5 = self.activ(self.cnn2d(hid4))
            out = self.ReLU(self.cnn3d(hid5))
            prob_list.append(out.detach().numpy())
        return prob_list