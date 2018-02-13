#!/usr/bin/env python

import util
import train_lfads
from lfads import LFADS
from gaussian_lfads import GaussianEncoder, GaussianGenerator, GaussianController
import os
import logging
import numpy as np
from chainer import cuda

def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    try:
        os.makedirs(args.model)
    except:
        pass

    # set up logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.model, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # load data
    logger.info('Loading data...')
    dataset = util.read_datasets(args.data_path, args.data_fname_stem)

    # save hyperparameters
    with open(os.path.join(args.model, 'params'), 'w') as f:
        for k, v in vars(args).items():
            print >> f, '{}\t{}'.format(k, v)

    # create test set
    input_dims = dataset["train_data"].shape[2]
    con_input_dim = args.gen_f_dims + (args.enc_h_dims*2)
    gen_input_dims = args.gen_g_dims + args.gen_u_dims

    encoder = GaussianEncoder(args.enc_n_layer, input_dims, args.enc_h_dims, args.enc_dropout)
    controller = GaussianController(con_input_dim, args.con_h_dims)
    generator = GaussianGenerator(gen_input_dims, args.gen_h_dims,
                                  args.gen_f_dims, args.gen_g_dims, input_dims, args.gen_u_dims, args.ar_tau, args.ar_noise_variance, args.batch_size)
    lfads = LFADS(encoder, controller, generator)

    lfads.save_model_def(args.model)

    train_lfads.train(lfads, dataset, args.optim, dest_dir=args.model, batch_size=args.batch_size,
                      max_epoch=args.epoch, gpu=args.gpu, save_every=args.save_every,
                      alpha_init=args.alpha_init, alpha_delta=args.alpha_delta)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train LFADS for gaussian input')

    parser.add_argument('model', help='destination of model')

    # NN architecture
    parser.add_argument('--enc-n-layer', type=int, default=1, help='number of hidden layer of encoder')
    parser.add_argument('--enc-h-dims', type=int, default=50, help='dimension of hidden variable of encoder')
    parser.add_argument('--enc-dropout', type=int, default=0, help='rate of dropout')

    parser.add_argument('--con-h-dims', type=int, default=20, help='dimension of hidden variable of controller')

    parser.add_argument('--gen-input-dims', type=int, default=20, help='dimension of hidden variable')
    parser.add_argument('--gen-h-dims', type=int, default=20, help='dimension of hidden variable')
    parser.add_argument('--gen-f-dims', type=int, default=20, help='dimension of factor')
    parser.add_argument('--gen-g-dims', type=int, default=20, help='dimension of generator RNN')
    parser.add_argument('--gen-u-dims', type=int, default=20, help='dimension of inferred input')
    parser.add_argument('--ar-tau', type=int, default=1, help='tau for autoregressive prior')
    parser.add_argument('--ar-noise-variance', type=int, default=10, help='noise variance for autoregressive prior')

    # training options
    parser.add_argument('--batch-size', type=int, default=24, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--optim', nargs='+', default=['Adam'], help='optimization method supported by chainer (optional arguments can be omitted)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    parser.add_argument('--save-every', type=int, default=100, help='save model every this number of epochs')

    parser.add_argument('--alpha-init', type=float, default=1., help='initial value of weight of KL loss')
    parser.add_argument('--alpha-delta', type=float, default=0., help='delta value of weight of KL loss')

    parser.add_argument('--data-path', type=str, default='./data/', help='file directory')
    parser.add_argument('--data-fname-stem', type=str, default='', help='filename')

    main(parser.parse_args())
