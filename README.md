# LFADS implemented with Chainer
Implemenation of [Latent Factor Analysis via Dynamical Systems](https://www.biorxiv.org/content/early/2017/06/20/152884) with [Chainer](https://chainer.org/).

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