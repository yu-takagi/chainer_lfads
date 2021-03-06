# -*- coding: utf-8 -*-
import lfads
import chainer.functions as F
import chainer.links as L
import math
import numpy as np
from chainer import Chain, ChainList
from chainer.functions import gaussian_kl_divergence
from chainer.functions import gaussian_nll
from chainer.functions.math import exponential
from chainer import cuda, Variable

class GaussianEncoder(lfads.Encoder):

    def __init__(self, n_layers, in_size, hidden_dims, use_dropout):
        super(GaussianEncoder, self).__init__()
        with self.init_scope():
            self.gru = L.NStepBiGRU(n_layers=n_layers, in_size=in_size,
                        out_size=hidden_dims, dropout=use_dropout)

    def __call__(self, xs, hx=None):
        hy, ys = self.gru(hx=hx, xs=xs)
        return hy, ys

class GaussianGenerator(lfads.Generator):
    def __init__(self, in_size, hidden_dims,
                 f_dims, g_dims, x_dims,
                 ar_tau, ar_noise_variance,batch_size,xp):
        super(GaussianGenerator, self).__init__()
        with self.init_scope():
            self.hidden_dims = hidden_dims
            self.gru = L.StatelessGRU(in_size=in_size, out_size=hidden_dims)
            self.l_g0_mu = L.Linear(None, g_dims)
            self.l_g0_ln_var = L.Linear(None, g_dims)
            self.l_f = L.Linear(None, f_dims)
            self.l_x_mu = L.Linear(None, x_dims)
            self.l_x_ln_var = L.Linear(None, x_dims)
            self.l_u_mu = L.Linear(None, in_size)
            self.l_u_ln_var = L.Linear(None, in_size)
            self.g_dims = g_dims
            self.u_dims = in_size
            # self.hoge = L.Linear(1,log_evar_dims)

            # process variance, the variance at time t over all instantiations of AR(1)
            log_evar = F.log(xp.asarray(ar_noise_variance,dtype=xp.float32))
            self.log_evar = log_evar

            # \tau, the autocorrelation time constant of the AR(1) process
            log_atau = xp.log(ar_tau)

            # alpha in x_t = \mu + alpha x_tm1 + \eps
            # alpha = exp(-1/tau)
            # alpha = exp(-1/exp(logtau))
            # alpha = exp(-exp(-logtau))
            alphas = xp.exp(-xp.exp(-log_atau),dtype=xp.float32)
            alphas_p1 = xp.array(xp.float32(1.0) + alphas)
            alphas_m1 = xp.array(xp.float32(1.0) - alphas)
            self.alphas = alphas

            # process noise
            # pvar = evar / (1- alpha^2)
            # logpvar = log ( exp(logevar) / (1 - alpha^2) )
            # logpvar = logevar - log(1-alpha^2)
            # logpvar = logevar - (log(1-alpha) + log(1+alpha))
            log_pvar = log_evar - xp.log(alphas_m1) - xp.log(alphas_p1)
            self.log_pvar = log_pvar

    def __call__(self, xs, hx=None):
        xp = cuda.get_array_module(xs)
        if hx is None:
            hx = Variable(xp.zeros((xs.data.shape[0],self.hidden_dims), dtype=xp.float32))
            hy = self.gru(h=hx, x=xs)
        else:
            hy = self.gru(h=hx, x=xs)
        return hy

    def sample_g0(self, zs):
        mu = self.l_g0_mu(zs)
        ln_var = self.l_g0_ln_var(zs)
        g_0 = F.gaussian(mu, ln_var)
        batchsize = len(mu.data)
        kl_g0 = gaussian_kl_divergence(mu, ln_var) / batchsize
        return g_0, kl_g0

    def sample_u_1(self, zs, batch_size=0, prior_sample=False):
        xp = cuda.get_array_module(zs)
        if prior_sample==False:
            # sampling
            mu = self.l_u_mu(zs)
            ln_var = self.l_u_ln_var(zs)
            u_1 = F.gaussian(mu, ln_var)

            # calculate kl
            logq = -gaussian_nll(u_1, mu, ln_var)
            mu_cond = Variable(xp.zeros_like(u_1,dtype=xp.float32))
            log_pvar = F.tile(self.log_pvar, (u_1.data.shape[0],self.u_dims))
            logp = -gaussian_nll(u_1, mu_cond, log_pvar)
            batchsize = len(mu.data)
            kl_u_1 = (logq - logp) / batchsize
            return u_1, kl_u_1
        else:
            # sampling
            log_pvar = F.tile(self.log_pvar, (batch_size,self.u_dims))
            mus = xp.zeros((batch_size,self.u_dims),dtype=xp.float32)
            u_1 = F.gaussian(Variable(mus), log_pvar)
            return u_1


    def sample_u_i(self, zs, ui_prev, batch_size=0, prior_sample=False):
        if prior_sample==False:
            # sampling
            mu = self.l_u_mu(zs)
            ln_var = self.l_u_ln_var(zs)
            u_i = F.gaussian(mu, ln_var)

            # calculate kl
            logq = -gaussian_nll(u_i, mu, ln_var)
            # W = self.hoge.W need reshape
            # logp = -gaussian_nll(u_i, self.alphas * ui_prev, W*log_evar)
            log_evar = F.tile(self.log_evar, (u_i.data.shape[0],self.u_dims))
            logp = -gaussian_nll(u_i, self.alphas * ui_prev, log_evar)
            batchsize = len(mu.data)
            kl_u_i = (logq - logp) / batchsize
            return u_i, kl_u_i
        else:
            # sampling
            log_evar = F.tile(self.log_evar, (batch_size, self.u_dims))
            u_i =  F.gaussian(self.alphas * ui_prev, log_evar)
            return u_i

    def sample_x_hat(self, zs, xs=[], nrep=1,calc_rec_loss=True):
        mu = self.l_x_mu(zs)
        ln_var = self.l_x_ln_var(zs)
        x_hat = []
        for i in range(0,nrep):
            x_hat.append(F.gaussian(mu, ln_var))
        x_hat = F.mean(F.vstack(x_hat),axis=0)
        if calc_rec_loss == True:
            batchsize = len(mu.data)
            rec_loss = gaussian_nll(xs, mu, ln_var) / batchsize
            return x_hat, rec_loss
        else:
            return x_hat


class GaussianController(lfads.Controller):

    def __init__(self, in_size, hidden_dims):
        super(GaussianController, self).__init__()
        with self.init_scope():
            self.hidden_dims = hidden_dims
            self.gru = L.StatelessGRU(in_size=in_size, out_size=hidden_dims)

    def __call__(self, xs, hx=None):
        xp = cuda.get_array_module(xs)
        if hx is None:
            hx = Variable(xp.zeros((xs.data.shape[0],self.hidden_dims), dtype=xp.float32))
            hy = self.gru(h=hx, x=xs)
        else:
            hy = self.gru(h=hx, x=xs)
        return hy
