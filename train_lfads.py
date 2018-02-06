import logging
import time
from collections import OrderedDict
import numpy as np
import pickle
from chainer import cuda, Variable
import chainer.functions as F
import util

def train(model, dataset, optimizer, dest_dir, batch_size=128, max_epoch=None, gpu=None, save_every=1, alpha_init=1., alpha_delta=0.):
    """Common training procedure.

    :param model: model to train
    :param dataset: training & validation data
    :param optimizer: chainer optimizer
    :param dest_dir: destination directory
    :param batch_size: number of sample in minibatch
    :param max_epoch: maximum number of epochs to train (None to train indefinitely)
    :param gpu: ID of GPU (None to use CPU)
    :param save_every: save every this number of epochs (first epoch and last epoch are always saved)
    :param alpha_init: initial value of alpha
    :param alpha_delta: change of alpha at every batch
    """
    if gpu is not None:
        # set up GPU
        cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    logger = logging.getLogger()

    # set up optimizer
    opt_enc = util.list2optimizer(optimizer)
    opt_con= util.list2optimizer(optimizer)
    opt_gen = util.list2optimizer(optimizer)
    opt_enc.setup(model.encoder)
    opt_con.setup(model.controller)
    opt_gen.setup(model.generator)

    # training loop
    epoch = 0
    alpha = alpha_init
    test_losses = []
    train_losses = []
    train_data = dataset["train_data"]
    test_data = dataset["valid_data"]

    while True:
        if max_epoch is not None and epoch >= max_epoch:
            # terminate training
            break

        # create batches
        x_data, _ = util.get_batch(train_data, batch_size=batch_size)
        # print x_data.shape
        x_data = x_data.astype(np.float32)

        # copy data to GPU
        if gpu is not None:
            x_data = cuda.to_gpu(x_data)

        # create variable
        xs = []
        [xs.append(Variable(x.astype(np.float32))) for x in x_data]

        # set new alpha
        alpha += alpha_delta
        alpha = min(alpha, 1.)
        alpha = max(alpha, 0.)

        time_start = time.time()

        # encoder
        _, h_bxtxd = model.encoder(xs)
        h_bxtxd = F.stack(h_bxtxd,0)
        d_dims = h_bxtxd.data.shape[2]

        # generator
        g0_bxd, kl_g0 = model.generator.sample_g0(F.concat([h_bxtxd[:,0,-d_dims/2:],h_bxtxd[:,-1,:d_dims/2]],axis=1))
        f0_bxd = model.generator.l_f(g0_bxd)

        # controller
        x_hat = []
        kl_u_total = 0
        rec_loss_total = 0

        for i in range(0, h_bxtxd[0].data.shape[0]):
            if i == 0:
                con_i = model.controller(F.concat((f0_bxd, h_bxtxd[:,i,:d_dims/2],h_bxtxd[:,i,d_dims/2:]),axis=1))
                u_i_bxd, kl_u = model.generator.sample_u_1(con_i)
                g_i_bxd = model.generator(F.concat([g0_bxd,u_i_bxd],axis=1))
            else:
                con_i = model.controller(F.concat([f_i, h_bxtxd[:,i,:d_dims/2],h_bxtxd[:,i,d_dims/2:]],axis=1), hx=con_i)
                u_i_bxd, kl_u = model.generator.sample_u_i(con_i,u_i_bxd)
                g_i_bxd = model.generator(F.concat([g_i_bxd,u_i_bxd],axis=1), hx=g_i_bxd)
            f_i = model.generator.l_f(g_i_bxd)
            x_hat_i, rec_loss_i = model.generator.sample_x_hat(Variable(x_data[:,i,:]),f_i)
            x_hat.append(x_hat_i)
            rec_loss_total += rec_loss_i
            kl_u_total += kl_u

        # update
        model.cleargrads()
        loss = kl_g0 + kl_u_total + rec_loss_total
        model.encoder.cleargrads()
        model.controller.cleargrads()
        model.generator.cleargrads()
        loss.backward()
        opt_enc.update()
        opt_con.update()
        opt_gen.update()

        # report training status

        time_end = time.time()
        time_delta = time_end - time_start

        # report training status
        status = OrderedDict()
        status['epoch'] = epoch
        status['time'] = int(time_delta * 1000)     # time in msec
        status['loss'] = '{:.4}'.format(float(loss.data))      # training loss
        status['alpha'] = alpha
        status['rec_loss'] = '{:.4}'.format(float(rec_loss_total.data))    # reconstruction loss
        status['kl_g0'] = '{:.4}'.format(float(kl_g0.data))    # KL-divergence loss for g0
        status['kl_u_total'] = '{:.4}'.format(float(kl_u_total.data))    # KL-divergence loss for us
        logger.info(_status_str(status))

        # # save model
        if epoch % save_every == 0 or (max_epoch is not None and epoch == max_epoch - 1):
            model.save(dest_dir, epoch)

        epoch += 1

def _status_str(status):
    lst = []
    for k, v in status.items():
        lst.append(k + ':')
        lst.append(str(v))
    return '\t'.join(lst)
