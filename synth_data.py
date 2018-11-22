import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import hmmlearn
import os
import h5py
import seaborn as sns
sns.set()
np.random.seed(0)
plt.rcParams['figure.figsize'] = (19.0,2.0)

def write_data(data_fname, data_dict):
    """Write data in HD5F format.

    Args:
    data_fname: The filename of teh file in which to write the data.
    data_dict:  The dictionary of data to write. The keys are strings
      and the values are numpy arrays.
    """

    dir_name = os.path.dirname(data_fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with h5py.File(data_fname, 'w') as hf:
        for k, v in data_dict.items():
            clean_k = k.replace('/', '_')
            if clean_k is not k:
                print('Warning: saving variable with name: ', k, ' as ', clean_k)
            else:
                print('Saving variable with name: ', clean_k)
                hf.create_dataset(clean_k, data=v, compression=False)


if __name__ == '__main__':
    ### #############################################################################
    # Generate sample data
    n_samples = 200000
    time = np.linspace(0, 4096, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape) +1 # Add noise
    S /= S.std(axis=0)  # Standardize data
    plt.plot(S[0:300,:])
    plt.savefig('./fig/latent.png') # -----(2)

    # Mix data
    A = np.random.uniform(-1,1,(20,3))
    X = np.dot(S, A.T)  # Generate observations
    plt.plot(X[0:300,:])
    plt.savefig('./fig/mixed.png') # -----(2)

    # Reshape
    X_rs = np.reshape(X,(-1,200,20))
    S_rs = np.reshape(S,(-1,200,3))

    # Train test split
    latent_train = S_rs[:500,:,:]
    latent_valid = S_rs[500:,:,:]
    noisy_data_train = X_rs[:500,:,:]
    noisy_data_valid = X_rs[500:,:,:]

    # save data
    data = {'train_truth': latent_train,
          'valid_truth': latent_valid,
          'train_data' : noisy_data_train,
          'valid_data' : noisy_data_valid,
          }
    data_path = './data/dat_gaussian'
    write_data(data_path, data)
