import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from offlinerl.utils.net.common import miniblock


class SVAEDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, Guassain_hidden_sizes=(256,256), LOG_MAX_STD=2, LOG_MIN_STD=-20, EPS=1e-8):
        super(SVAEDecoder,self).__init__()
        self.obs_dim = obs_dim
        self.Guassain_hidden_sizes = list(Guassain_hidden_sizes).copy()
        self.LOG_MAX_STD = LOG_MAX_STD
        self.LOG_MIN_STD = LOG_MIN_STD
        self.EPS = EPS
        self.Guassain_input_dim = latent_dim
        self.Guassain_mlp = miniblock(self.Guassain_input_dim, self.Guassain_hidden_sizes[0], None)
        if len(Guassain_hidden_sizes)>=2:
            for i in range(1,len(Guassain_hidden_sizes)):
                self.Guassain_mlp += miniblock(Guassain_hidden_sizes[i-1], Guassain_hidden_sizes[i], None)
        self.Guassain_mlp = nn.Sequential(*self.Guassain_mlp)
        self.Guassain_mu_mlp = [nn.Linear(self.Guassain_hidden_sizes[-1], obs_dim)]
        self.Guassain_logstd_mlp = [nn.Linear(self.Guassain_hidden_sizes[-1], obs_dim)]
        self.Guassain_mu_mlp = nn.Sequential(*self.Guassain_mu_mlp)
        self.Guassain_logstd_mlp = nn.Sequential(*self.Guassain_logstd_mlp)

    def gaussian_likelihood(self,x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return torch.mean(pre_sum, dim=-1)

    def forward(self, latent, ref_obs=None): 
        out = self.Guassain_mlp(latent)
        mu = self.Guassain_mu_mlp(out)
        log_std = self.Guassain_logstd_mlp(out)
        log_std = torch.clip(log_std, self.LOG_MIN_STD, self.LOG_MAX_STD)
        std = torch.exp(log_std)
        if ref_obs is not None:
            likelihood = self.gaussian_likelihood(ref_obs, mu, log_std)
        else:
            likelihood = None
        return mu, std, likelihood

    def apply_squashing_func(self, mu, pi, logp_pi):
        logp_pi -= torch.sum(2 * (np.log(2) - pi - F.softplus(-2 * pi)), dim=-1)
        # Squash those unbounded actions!
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        return mu, pi, logp_pi









