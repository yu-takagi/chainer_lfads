import sys
import util
import os
from chainer import cuda, Variable
from lfads import LFADS
from chainer import cuda, Variable
import chainer.functions as F
import numpy as np
import six
import h5py

def main(args):

    # load data
    print >> sys.stderr, 'Loading data...'
    dataset = util.read_datasets(args.data_path, args.data_fname_stem)
    train_data = dataset["train_data"]
    test_data = dataset["valid_data"]

    # load model
    model = LFADS.load(args.model)
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # posterior sampling
    # encoder
    ndata = train_data.shape[0]
    x_hat_all = []
    for i in range(ndata):
        print("now %d'th data among %d" % (i, ndata) )
        x_data = train_data[i,:,:].astype(np.float32)
        x_data = x_data[np.newaxis,:,:]
        x_data = np.tile(x_data,(args.batch_size,1,1))

        # copy data to GPU
        if args.gpu is not None:
            x_data = cuda.to_gpu(x_data)

        # create variable
        xs = []
        [xs.append(Variable(x.astype(np.float32))) for x in x_data]

        # encoder
        _, h_bxtxd = model.encoder(xs)
        h_bxtxd = F.stack(h_bxtxd,0)
        d_dims = h_bxtxd.data.shape[2]

        # generator
        g0_bxd, _ = model.generator.sample_g0(F.concat([h_bxtxd[:,0,-d_dims/2:],h_bxtxd[:,-1,:d_dims/2]],axis=1))
        f0_bxd = model.generator.l_f(g0_bxd)

        # controller
        x_hat = []
        kl_u_total = 0
        rec_loss_total = 0

        for j in range(0, h_bxtxd[0].data.shape[0]):
            if j == 0:
                con_i = model.controller(F.concat((f0_bxd, h_bxtxd[:,j,:d_dims/2],h_bxtxd[:,j,d_dims/2:]),axis=1))
                u_i_bxd, _ = model.generator.sample_u_1(con_i)
                g_i_bxd = model.generator(F.concat([g0_bxd,u_i_bxd],axis=1))
            else:
                con_i = model.controller(F.concat([f_i, h_bxtxd[:,j,:d_dims/2],h_bxtxd[:,j,d_dims/2:]],axis=1), hx=con_i)
                u_i_bxd, _ = model.generator.sample_u_i(con_i,u_i_bxd)
                g_i_bxd = model.generator(F.concat([g_i_bxd,u_i_bxd],axis=1), hx=g_i_bxd)
            f_i = model.generator.l_f(g_i_bxd)
            x_hat_i, _ = model.generator.sample_x_hat(Variable(x_data[:,j,:]),f_i)
            x_hat_i = F.mean(x_hat_i,axis=0)
            x_hat.append(cuda.to_cpu(x_hat_i.data))
        x_hat_all.append(x_hat)
    hf = h5py.File('data.h5', 'w')
    hf.create_dataset('data', data=x_hat_all)
    hf.flush()
    hf.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train LFADS for gaussian input')
    parser.add_argument('model', help='destination of model')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size(number of samples for averaging )')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--data-path', type=str, default='./data/', help='file directory')
    parser.add_argument('--data-fname-stem', type=str, default='007_day1', help='filename')

    main(parser.parse_args())
