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
    def __init__(self,in_size,BayesEnc=False,BayesianDec=False,dim=10):
        super(VAE, self).__init__()
        self.dim=dim
        self.hid_dim=16
        in_dim=1
        out_ch=16
        if(not BayesEnc):
            self.cnn1e = torch.nn.Conv2d(in_channels=in_dim, out_channels=out_ch, kernel_size=3, stride=2)
            self.cnn2e = torch.nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2)
            self.cnn3e = torch.nn.Conv2d(in_channels=out_ch, out_channels=4, kernel_size=3, stride=2)
            self.fcce = torch.nn.Linear(in_features=144, out_features=self.hid_dim * 2)
        else:
            prior_sigma=.1
            self.cnn1e = bnn.BayesConv2d(prior_mu=0, prior_sigma=.02, in_channels=in_dim, out_channels=out_ch, kernel_size=4, stride=2)
            self.cnn2e = bnn.BayesConv2d(prior_mu=0, prior_sigma=.02, in_channels=out_ch, out_channels=out_ch, kernel_size=2, stride=2)
            self.cnn3e = bnn.BayesConv2d(prior_mu=0, prior_sigma=.02, in_channels=out_ch, out_channels=4, kernel_size=2, stride=2)
            self.fcce = bnn.BayesLinear(prior_mu=0, prior_sigma=.02, in_features=144, out_features=self.hid_dim * 2)
            self.cnn1e.prior_sigma = prior_sigma
            self.cnn1e.prior_log_sigma = math.log(prior_sigma)
            self.cnn2e.prior_sigma = prior_sigma
            self.cnn2e.prior_log_sigma = math.log(prior_sigma)
            self.cnn3e.prior_sigma = prior_sigma
            self.cnn3e.prior_log_sigma = math.log(prior_sigma)
        if (not BayesianDec):
            self.fcc1d = torch.nn.Linear(in_features=self.hid_dim, out_features=4 * 7 * 7)
            self.cnn1d = torch.nn.ConvTranspose2d(in_channels=4, out_channels=out_ch, kernel_size=2, stride=2)
            self.cnn2d = torch.nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=2, stride=2)
            self.cnn3d = torch.nn.ConvTranspose2d(in_channels=out_ch, out_channels=in_dim, kernel_size=2, padding=0, stride=2)
        else:
            prior_sigma = .1
            self.fcc1d = bnn.BayesLinear(prior_mu=0, prior_sigma=.02, in_features=self.hid_dim, out_features=4 * 6 * 6)
            self.cnn1d = mylayer.BayesConv2dTranspose(prior_mu=0, prior_sigma=.02, in_channels=4, out_channels=out_ch, kernel_size=3, stride=2,padding=0,output_padding=0)
            self.cnn2d = mylayer.BayesConv2dTranspose(prior_mu=0, prior_sigma=.02, in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2,padding=0)
            self.cnn3d = mylayer.BayesConv2dTranspose(prior_mu=0, prior_sigma=.02, in_channels=out_ch, out_channels=in_dim, kernel_size=2, stride=2,padding=0)
            self.cnn1d.prior_sigma = prior_sigma
            self.cnn1d.prior_log_sigma = math.log(prior_sigma)
            self.cnn2d.prior_sigma = prior_sigma
            self.cnn2d.prior_log_sigma = math.log(prior_sigma)
            self.cnn3d.prior_sigma = prior_sigma
            self.cnn3d.prior_log_sigma = math.log(prior_sigma)
        self.pool=torch.nn.MaxPool2d(kernel_size=(2,2))
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
        x_input=x_input.type(torch.float32).view(x_input.shape[0] ,1 , 56,  56)
        prob_list = []
        mu_list=[]
        log_var_list = []
        hid1 = self.activ(self.cnn1e(x_input))
        hid2 = self.activ(self.cnn2e(hid1))
        hid3 = torch.flatten(self.activ(self.cnn3e(hid2)), start_dim=1, end_dim=- 1)
        hid4 = self.fcce(hid3).view(x_input.shape[0], 2,  -1)
        mu = hid4[:, 0, :]  # the first feature values as mean
        log_var = hid4[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization 16, 3, 3
        z = self.reparameterize(mu, log_var)
        for ind_m in range(m):
            fc_hid=self.activ(self.fcc1d(z)).view(x_input.shape[0], 4,6,6)
            hid5 = self.activ(self.cnn1d(fc_hid))
            hid6 = self.activ(self.cnn2d(hid5))
            hid7 = self.cnn3d(hid6)
            out=hid7.view(x_input.shape[0],-1)
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