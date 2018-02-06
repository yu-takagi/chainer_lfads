# LFADS implemented with Chainer
Implemenation of [Latent Factor Analysis via Dynamical Systems](https://www.biorxiv.org/content/early/2017/06/20/152884) with [Chainer](https://chainer.org/).
LFADS is an sequential adaptation of a VAE (Variational Auto-Encoder). It can de-noise and find a low-dimensional representations for time series data. Although it is especially designed for investigating neuroscience data, it can be applied for any time series data.
MIT license. Contributions welcome.

## Requirements
python 2.x, chainer 3.3.0, numpy, h5py

## Train an LFADS model
You have to prepare the dataset in the "data" directory.
```
python run_lfads.py dataset --gpu 0 --epoch 1000 --batch-size 64
```

## Evaluate a trained model

##### Take samples from posterior then average (denoising operation)
```
python run_posterior_sampling.py ./dataset/epoch1000 --gpu 0
```

## TODO
* Implementing a sampling from prior (generation of completely new samples)
* L2 loss
* Monitoring validation error during training
* Preparing datasets and demos for github
