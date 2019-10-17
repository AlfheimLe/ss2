import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import math
from torch.nn import init
from torch.autograd import Variable

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var



def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)
def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)

class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
    def forward(self, norm_mean, norm_log_sigma):
        pass
    @staticmethod
    def __sample_norm(mu, log_sigma):
        """

        :param mu:
        :param log_sigma:
        :return: latent normal z ~ N(mu, sigma^2)
        """
        std_z = torch.randn(mu.size())
        if mu.is_cuda:
            std_z = std_z.cuda()
        return mu + torch.exp(log_sigma) * std_z

class Predict(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super(Predict, self).__init__()
        self.densy = nn.Linear(in_features, hid_features)
        self.out = nn.Linear(hid_features, out_features)
    def forward(self, x):
        x = F.relu(self.densy(x))
        x = F.sigmoid(self.out(x))
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # [self.x_dim, self.h_dim, self.z_dim, self.len] = dims
        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.len = args.y_dim
        self.firstcell = Predict(self.x_dim, self.h_dim, 1)
        self.grucell_layer = nn.ModuleList([nn.GRUCell(self.x_dim+1, self.h_dim) for i in range(self.len)])
        #self.grucell = nn.GRUCell(self.x_dim+1, self.h_dim)
        predict_layers = [Predict(self.h_dim, self.h_dim, 1) for i in range(self.len)]
        self.predict = nn.ModuleList(predict_layers)
        self.sample = GaussianSample(self.h_dim, self.z_dim)

    def forward(self, x):
        t = torch.zeros(x.shape[0], self.len).cuda()
        z = torch.zeros(x.shape[0], self.h_dim).cuda()
        t[:, 0] = self.firstcell(x).t()
        for i ,predict, grucell in zip(range(self.len), self.predict, self.grucell_layer):
        #for i, predict in zip(range(self.len), self.predict):
            tmp_t = (t[:, i] >= 0.5).float()
            tmp_x = torch.cat([x, tmp_t.reshape(tmp_t.size(0), 1)], dim=1)
            mu = grucell(tmp_x, z)
            #mu = self.grucell(tmp_x, z)
            sigma = 0.005*torch.ones_like(mu)
            eps = torch.randn(mu.size())
            if mu.is_cuda:
                eps = eps.cuda()
            std = sigma.mul(0.5).exp_()
            z = mu.addcmul(std, eps)
            if i < self.len-1:
                t[:, i+1] = predict(z).t()
        latent, mu, log_sigma = self.sample(z)
        return latent, mu, log_sigma, t


class Decoder_x(nn.Module):
    '''
    reconstruct the data points
    '''
    def __init__(self, args):
        super(Decoder_x, self).__init__()
        #[self.z_dim, self.h_dim, self.x_dim, self.len] = dims
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.len = args.y_dim
        self.x_dim = args.x_dim
        self.workers = args.workers
        self.grucell_layer = nn.ModuleList([nn.GRUCell(self.z_dim+1, self.h_dim)for i in range(self.len)])
        #self.grucell = nn.GRUCell(self.z_dim+1, self.h_dim)
        self.reconstruct = nn.Linear(self.h_dim, self.x_dim)
        predict_layers = [Predict(self.h_dim, self.h_dim, self.workers) for i in range(self.len)]
        self.predict = nn.ModuleList(predict_layers)
    def forward(self, z, t):
        z_theta = torch.zeros(z.size(0), self.h_dim).cuda()
        workers_preds = torch.zeros(self.workers, z.size(0), self.len).cuda()
        for i, predict, grucell in zip(range(self.len), self.predict, self.grucell_layer):
        #for i, predict in zip(range(self.len), self.predict):
            tmp_z = torch.cat([z, t[:, i].reshape(z.size(0), 1)], dim=1)
            mu_theta = grucell(tmp_z, z_theta)
            #mu_theta = self.grucell(tmp_z, z_theta)
            log_sigma_theta = 0.005*torch.ones_like(mu_theta)
            eps = torch.rand(mu_theta.size())
            if z.is_cuda:
                eps = eps.cuda()
            std = log_sigma_theta.mul(0.5).exp_()
            z_theta = mu_theta.addcmul(std, eps)
            workers_preds[:, :, i] = predict(z_theta).t()
        recon = F.sigmoid(self.reconstruct(z_theta))
        return recon, workers_preds

class Decoder_y(nn.Module):
    """
    reconstruct the K workers label the data points
    """
    def __init__(self, args):
        super(Decoder_y, self).__init__()


class DSGM(nn.Module):
    def __init__(self, args):
        super(DSGM, self).__init__()
        x_dim = args.x_dim
        z_dim = args.z_dim
        h_dim = args.h_dim
        self.workers = args.workers
        self.y_dim = args.y_dim

        #p model sequantial generative model

        self.decoder = Decoder_x(args)
        self.encoder = Encoder(args)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, workers=None):
        z, z_mu, z_sigma, t = self.encoder(x)
        tmp_t = (t>0.5).float()
        new_x, worker_preds = self.decoder(z, tmp_t)
        log_p_x_tz = torch.sum(x * torch.log(new_x), dim=1)
        # sampler = D.Bernoulli(t)
        # t = sampler.sample()
        #越小越好
        if workers is None:
            log_p_y_tz = 0
            workers_bar = 1 / self.y_dim * torch.ones(x.size(0), self.y_dim).cuda()
        else:
            #a=F.binary_cross_entropy(workers, worker_preds, reduction='none')
            log_p_y_tz = -torch.sum(torch.sum(F.binary_cross_entropy(worker_preds, workers,  reduction='none'),  dim=0), dim=1)
            workers_bar = workers.mean(dim=0)
        log_kl_qp_t = torch.sum(t*(torch.log(t+1e-6)-torch.log(workers_bar+1e-6)), dim=1)
        #log_kl_qp_t = -torch.sum(F.binary_cross_entropy(t, workers_bar, reduction='none'), dim=1)
        log_p_z = log_standard_gaussian(z)
        log_q_z_tx = log_gaussian(z, z_mu, z_sigma)
        loss = torch.mean(log_q_z_tx-log_p_z-log_p_y_tz-log_p_x_tz+log_kl_qp_t)
        return t, loss
        # latent, mu, log_sigma, t = self.encoder(x)
        # KL1 = self._kld1(latent, (mu, log_sigma)).sum()
        # t = (t > 0.5).float()
        # # tmp = D.Bernoulli(t)
        # # real_t = tmp.sample()
        # reconstruction, workers_pred = self.decoder(latent, t)
        # rec_loss = -F.mse_loss(reconstruction, x)
        # total_loss = -KL1 + rec_loss
        #
        # if workers is None:
        #     workers_bar = 1/self.y_dim*torch.ones(x.size(0), self.y_dim).cuda()
        # else:
        #     total_loss -= F.mse_loss(workers, workers_pred)
        #     workers_bar = workers.mean(dim=0)
        # KL2 = (t*torch.log(t+1e-6)-t*torch.log(workers_bar+1e-6)).sum()
        # total_loss -= KL2
        #
        # return t, total_loss
    def _kld1(self, z, q_param):
        (mu, log_var) = q_param
        qz = log_gaussian(z, mu, log_var)
        pz = log_standard_gaussian(z)
        return qz-pz

