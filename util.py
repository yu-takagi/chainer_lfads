import numpy as np
import os
import h5py
from chainer import cuda, Variable
import chainer.optimizers as O
from collections import defaultdict

def list2optimizer(lst):
    """Create chainer optimizer object from list of strings, such as ['SGD', '0.01']"""
    optim_name = lst[0]
    optim_args = map(float, lst[1:])
    optimizer = getattr(O, optim_name)(*optim_args)
    return optimizer

def read_datasets(data_path, data_fname_stem):
  """Read dataset sin HD5F format.

  This function assumes the dataset_dict is a mapping ( string ->
  to data_dict ).  It calls write_data for each data dictionary,
  post-fixing the data filename with the key of the dataset.

  Args:
    data_path: The path to the save directory.
    data_fname_stem: The filename stem of the file in which to write the data.
  """

  dataset_dict = {}
  fnames = os.listdir(data_path)

  print ('loading data from ' + data_path + ' with stem ' + data_fname_stem)
  for fname in fnames:
    if fname.startswith(data_fname_stem):
      data_dict = read_data(os.path.join(data_path,fname))
      idx = len(data_fname_stem) + 1
      key = fname[idx:]
      data_dict['data_dim'] = data_dict['train_data'].shape[2]
      data_dict['num_steps'] = data_dict['train_data'].shape[1]
      dataset_dict = data_dict

  if len(dataset_dict) == 0:
    raise ValueError("Failed to load any datasets, are you sure that the "
                     "'--data_dir' and '--data_filename_stem' flag values "
                     "are correct?")
  print (str(len(dataset_dict)) + ' datasets loaded')
  return dataset_dict

def read_data(data_fname):
  """ Read saved data in HDF5 format.

  Args:
    data_fname: The filename of the file from which to read the data.
  Returns:
    A dictionary whose keys will vary depending on dataset (but should
    always contain the keys 'train_data' and 'valid_data') and whose
    values are numpy arrays.
  """

  try:
    with h5py.File(data_fname, 'r') as hf:
      data_dict = {k: np.array(v) for k, v in hf.items()}
      return data_dict
  except IOError:
    print("Cannot open %s for reading." % data_fname)
    raise

def get_batch(data_extxd, ext_input_extxi=None, batch_size=None,
              example_idxs=None):
    """Get a batch of data, either randomly chosen, or specified directly.
    Args:
      data_extxd: The data to model, numpy tensors with shape:
        # examples x # time steps x # dimensions
      ext_input_extxi (optional): The external inputs, numpy tensor with shape:
        # examples x # time steps x # external input dimensions
      batch_size:  The size of the batch to return
      example_idxs (optional): The example indices used to select examples.
    Returns:
      A tuple with two parts:
        1. Batched data numpy tensor with shape:
        batch_size x # time steps x # dimensions
        2. Batched external input numpy tensor with shape:
        batch_size x # time steps x # external input dims
    """
    assert batch_size is not None or example_idxs is not None, "Problems"
    E, T, D = data_extxd.shape
    if example_idxs is None:
      example_idxs = np.random.choice(E, batch_size)

    ext_input_bxtxi = None
    if ext_input_extxi is not None:
      ext_input_bxtxi = ext_input_extxi[example_idxs,:,:]

    return data_extxd[example_idxs,:,:], ext_input_bxtxi

