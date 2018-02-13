import sys
import util
import os
from chainer import cuda, Variable
from lfads import LFADS
import chainer.functions as F
import numpy as np
import six
import h5py

def main(args):

    # load model
    model = LFADS.load(args.model)
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # generate completely new samples
    print("generate completely new samples from prior distribution...")

    # sample initial condition (g0)
    xp = cuda.cupy
    mus = xp.zeros((args.batch_size,model.generator.g_dims),dtype=xp.float32)
    sigmas = xp.ones((args.batch_size,model.generator.g_dims),dtype=xp.float32)
    g0_bxd = F.gaussian(Variable(mus), Variable(sigmas))

    # inffered inputs are sampled from a Gaussian autoregressive prior
    xs = []
    for i in range(args.nsample):
        print("now generating %d'th sample among %d" % (i, args.nsample) )
        if i == 0:
            u_i_bxd = model.generator.sample_u_1(0,batch_size=args.batch_size,prior_sample=True)
            g_i_bxd = model.generator(F.concat([g0_bxd,u_i_bxd],axis=1))
        else:
            u_i_bxd = model.generator.sample_u_i(0,u_i_bxd,batch_size=args.batch_size,prior_sample=True)
            g_i_bxd = model.generator(F.concat([g_i_bxd,u_i_bxd],axis=1), hx=g_i_bxd)
        f_i = model.generator.l_f(g_i_bxd)
        x_i = model.generator.sample_x_hat(f_i,calc_rec_loss=False)
        xs.append(cuda.to_cpu(x_i.data))

    # save
    hf = h5py.File('data.h5', 'w')
    hf.create_dataset('data', data=xs)
    hf.flush()
    hf.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train LFADS for gaussian input')
    parser.add_argument('model', help='destination of model')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size(number of samples for generating )')
    parser.add_argument('--nsample', type=int, default=300, help='number of samples for generation')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--data-path', type=str, default='./data/', help='file directory')
    parser.add_argument('--data-fname-stem', type=str, default='', help='filename')

    main(parser.parse_args())
