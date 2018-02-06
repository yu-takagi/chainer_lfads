import cPickle as pickle
import os
import copy
import numpy as np
from chainer import Chain, Variable
import chainer.functions as F
from chainer.serializers import save_hdf5, load_hdf5


class Encoder(Chain):

    def __init__(self, **links):
        super(Encoder, self).__init__(**links)

    def __call__(self, xs, hx):
        raise NotImplementedError

class Controller(Chain):

    def __init__(self, **links):
        super(Controller, self).__init__(**links)

    def __call__(self, self, xs, hx):
        raise NotImplementedError

class Generator(Chain):

    def __init__(self, **links):
        super(Generator, self).__init__(**links)

    def __call__(self, xs, hx):
        raise NotImplementedError

class LFADS(Chain):

    MODEL_DEF_NAME = 'model_def.pickle'

    def __init__(self, encoder, controller, generator):
        super(LFADS, self).__init__(
            encoder=encoder,
            controller=controller,
            generator=generator
        )

    def __call__(self):
        raise NotImplementedError

    def save_model_def(self, model_base_path):
        obj = copy.deepcopy(self)
        for p in obj.params():
            p.data = None
        path = os.path.join(model_base_path, self.MODEL_DEF_NAME)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def save(self, model_base_path, epoch):
        assert os.path.exists(os.path.join(model_base_path, self.MODEL_DEF_NAME)), 'Must call save_model_def() first'
        model_path = os.path.join(model_base_path, 'epoch{}'.format(epoch))
        save_hdf5(model_path, self)

    @classmethod
    def load(cls, path):
        model_base_path = os.path.dirname(path)
        model_def_path = os.path.join(model_base_path, cls.MODEL_DEF_NAME)
        with open(model_def_path, 'rb') as f:
            model = pickle.load(f)  # load model definition
            load_hdf5(path, model)  # load parameters
        return model
