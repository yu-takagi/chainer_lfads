import logging
import time
from collections import OrderedDict
import numpy as np
import pickle
from chainer import cuda, Variable
import chainer.functions as F
import util

def train(model, dataset, optimizer, dest_dir, batch_size=128, max_epoch=None, gpu=None, save_every=5,
          test_every=5, alpha_init=1., alpha_delta=0., l2_weight_gen=0., l2_weight_con=0.):
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
    opt_enc.setup(model.encoder)
    if hasattr(model,'controller'):
        opt_con = util.list2optimizer(optimizer)
        opt_con.setup(model.controller)
    opt_gen = util.list2optimizer(optimizer)
    opt_gen.setup(model.generator)

    # training loop
    epoch = 0
    alpha = alpha_init
    test_losses = []
    train_losses = []
    train_data = dataset["train_data"]
    test_data = dataset["valid_data"]
    split = 'test'

    while True:
        if max_epoch is not None and epoch >= max_epoch:
            # terminate training
            break

        # Every ten epochs, try validation set
        if split == 'train':
            x_data, _ = util.get_batch(train_data, batch_size=batch_size)
        else:
            x_data, _ = util.get_batch(test_data, batch_size=batch_size)

        # create batches
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

        # main
        x_hat = []
        rec_loss_total = 0
        if hasattr(model,'controller'):
            kl_u_total = 0

        for i in range(0, h_bxtxd[0].data.shape[0]):
            if i == 0:
                if hasattr(model,'controller'):
                    con_i = model.controller(F.concat((f0_bxd, h_bxtxd[:,i,:d_dims/2],h_bxtxd[:,i,d_dims/2:]),axis=1))
                    u_i_bxd, kl_u = model.generator.sample_u_1(con_i)
                    kl_u_total += kl_u
                    g_i_bxd = model.generator(u_i_bxd,hx=g0_bxd)
                else:
                    g_i_bxd = model.generator(F.concat((h_bxtxd[:,i,:d_dims/2],h_bxtxd[:,i,d_dims/2:]),axis=1),hx=g0_bxd)
            else:
                if hasattr(model,'controller'):
                    con_i = model.controller(F.concat([f_i, h_bxtxd[:,i,:d_dims/2],h_bxtxd[:,i,d_dims/2:]],axis=1), hx=con_i)
                    u_i_bxd, kl_u = model.generator.sample_u_i(con_i,u_i_bxd)
                    kl_u_total += kl_u
                    g_i_bxd = model.generator(u_i_bxd,hx=g_i_bxd)
                else:
                    g_i_bxd = model.generator(F.concat([h_bxtxd[:,i,:d_dims/2],h_bxtxd[:,i,d_dims/2:]],axis=1), hx=g_i_bxd)

            f_i = model.generator.l_f(g_i_bxd)
            x_hat_i, rec_loss_i = model.generator.sample_x_hat(f_i,xs=Variable(x_data[:,i,:]),nrep=1)
            x_hat.append(x_hat_i)
            rec_loss_total += rec_loss_i

        # calculate loss
        if hasattr(model,'controller'):
            loss = rec_loss_total + alpha * (kl_g0 + kl_u_total)
        else:
            loss = rec_loss_total + alpha * kl_g0

        l2_loss = 0;
        if l2_weight_gen > 0:
            l2_W_gen = F.sum(F.square(model.generator.gru.W.W))
            l2_W_r_gen = F.sum(F.square(model.generator.gru.W_r.W))
            l2_W_z_gen = F.sum(F.square(model.generator.gru.W_z.W))
            l2_gen = l2_weight_gen * (l2_W_gen + l2_W_r_gen + l2_W_z_gen)
            l2_loss += l2_gen
        if hasattr(model,'controller') and l2_weight_con>0:
            l2_W_con = F.sum(F.square(model.controller.gru.W.W))
            l2_W_r_con = F.sum(F.square(model.controller.gru.W_r.W))
            l2_W_z_con = F.sum(F.square(model.controller.gru.W_z.W))
            l2_con = l2_weight_con * (l2_W_con + l2_W_r_con + l2_W_z_con)
            l2_loss += l2_con
        loss += l2_loss

        # update
        if split == 'train':
            model.cleargrads()
            model.encoder.cleargrads()
            if hasattr(model,'controller'):
                model.controller.cleargrads()
            model.generator.cleargrads()
            loss.backward()
            opt_enc.update()
            if hasattr(model,'controller'):
                opt_con.update()
            opt_gen.update()

        # report training status

        time_end = time.time()
        time_delta = time_end - time_start

        # report training status
        status = OrderedDict()
        status['epoch'] = epoch
        status['time'] = int(time_delta * 1000)     # time in msec
        status['alpha'] = alpha

        status[split+'_loss'] = '{:.4}'.format(float(loss.data))      # total training loss
        status[split+'_rec_loss'] = '{:.4}'.format(float(rec_loss_total.data))    # reconstruction loss
        status[split+'_kl_g0'] = '{:.4}'.format(float(kl_g0.data))    # KL-divergence loss for g0
        if hasattr(model,'controller'):
            status[split+'_kl_u_total'] = '{:.4}'.format(float(kl_u_total.data))    # KL-divergence loss for us
            if l2_weight_con > 0:
                status[split+'_l2_loss_con'] = '{:.4}'.format(float(l2_con.data))    # L2 loss for controller
        if l2_weight_gen > 0:
            status[split+'_l2_loss_gen'] = '{:.4}'.format(float(l2_gen.data))    # L2 loss for generator
        logger.info(_status_str(status))

        # # save model
        if ((epoch % save_every) == 0 or (max_epoch is not None and epoch == max_epoch - 1)) and split=='train':
            model.save(dest_dir, epoch)

        if split == 'train' and epoch % test_every == 0:
            split = 'test'
        else:
            split = 'train'
            epoch += 1

def _status_str(status):
    lst = []
    for k, v in status.items():
        lst.append(k + ':')
        lst.append(str(v))
    return '\t'.join(lst)
