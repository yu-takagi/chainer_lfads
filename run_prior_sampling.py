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
    g0_1x1 = F.gaussian(0, 1)
    g0_bxd = F.tile(g0_1x1, (args.batch_size,args.model.generator.g_dims))

    # inffered inputs are sampled from a Gaussian autoregressive prior
    xs = []
    for i in range(nsample):
        print("now generating %d'th sample among %d" % (i, nsamples) )
        if i == 0:
            u_i_bxd = model.generator.sample_u_1(_,batch_size=batch_size,prior_sample=True)
            g_i_bxd = model.generator(F.concat([g0_bxd,u_i_bxd],axis=1))
        else:
            u_i_bxd = model.generator.sample_u_i(_,u_i_bxd,batch_size=batch_size,prior_sample=True)
            g_i_bxd = model.generator(F.concat([g_i_bxd,u_i_bxd],axis=1), hx=g_i_bxd)
        f_i = model.generator.l_f(g_i_bxd)
        x_i = model.generator.sample_x_hat(f_i,rec_loss=False)
        xs.append(cuda.to_cpu(x_i.data))
    hf = h5py.File('data.h5', 'w')
    hf.create_dataset('data', data=xs)
    hf.flush()
    hf.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train LFADS for gaussian input')
    parser.add_argument('model', help='destination of model')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size(number of samples for averaging )')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--data-path', type=str, default='./data/', help='file directory')
    parser.add_argument('--data-fname-stem', type=str, default='', help='filename')

    main(parser.parse_args())
