from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from six import add_metaclass
from torchvision.utils import save_image

import pyro
from pyro.contrib.examples import util
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.infer.ea.batched_linear import BatchedLinear
from pyro.infer.ea.evolution import Evolution
from pyro.infer.ea.parallelized_elbo import Parallelized_ELBO
from pyro.optim import Adam
from utils.mnist_cached import DATA_DIR, RESULTS_DIR


logging.basicConfig(format='%(message)s')
logging.getLogger('pyro').setLevel(logging.INFO)

"""
Comparison of VAE implementation in PyTorch and Pyro. This example can be
used for profiling purposes.

The PyTorch VAE example is taken (with minor modification) from pytorch/examples.
Source: https://github.com/pytorch/examples/tree/master/vae
"""

TRAIN = 'train'
TEST = 'test'
OUTPUT_DIR = RESULTS_DIR


# VAE encoder network
class Encoder(nn.Module):
    def __init__(self, batches):
        super(Encoder, self).__init__()
        self.batches = batches
        self.fc1 = BatchedLinear(784, 400, batches)
        self.fc21 = BatchedLinear(400, 20, batches)
        self.fc22 = BatchedLinear(400, 20, batches)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.shape[0] != self.batches:
            x = x.expand(torch.Size((self.batches,)) + x.shape)
        x = x.reshape(self.batches, -1, 784)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), torch.exp(self.fc22(h1))


# VAE Decoder network
class Decoder(nn.Module):
    def __init__(self, batches):
        super(Decoder, self).__init__()
        self.fc3 = BatchedLinear(20, 400, batches)
        self.fc4 = BatchedLinear(400, 784, batches)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))


@add_metaclass(ABCMeta)
class VAE(object):
    """
    Abstract class for the variational auto-encoder. The abstract method
    for training the network is implemented by subclasses.
    """

    def __init__(self, args, train_loader, test_loader, cuda):
        self.args = args
        self.vae_encoder = Encoder(args.population_size)
        self.vae_decoder = Decoder(args.population_size)
        if cuda:
            self.vae_encoder = self.vae_encoder.cuda()
            self.vae_decoder = self.vae_decoder.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cuda = cuda
        self.mode = TRAIN

    def set_train(self, is_train=True):
        if is_train:
            self.mode = TRAIN
            self.vae_encoder.train()
            self.vae_decoder.train()
        else:
            self.mode = TEST
            self.vae_encoder.eval()
            self.vae_decoder.eval()

    @abstractmethod
    def compute_loss_and_gradient(self, x):
        """
        Given a batch of data `x`, run the optimizer (backpropagate the gradient),
        and return the computed loss.

        :param x: batch of data or a single datum (MNIST image).
        :return: loss computed on the data batch.
        """
        return

    def model_eval(self, x):
        """
        Given a batch of data `x`, run it through the trained VAE network to get
        the reconstructed image.

        :param x: batch of data or a single datum (MNIST image).
        :return: reconstructed image, and the latent z's mean and variance.
        """
        z_mean, z_var = self.vae_encoder(x)
        if self.mode == TRAIN:
            z = dist.Normal(z_mean, z_var.sqrt()).sample()
        else:
            z = z_mean
        return self.vae_decoder(z), z_mean, z_var

    def train(self, epoch):
        self.set_train(is_train=True)
        train_loss = 0
        for batch_idx, (x, _) in enumerate(self.train_loader):
            if self.cuda:
                x = x.cuda()
            loss = self.compute_loss_and_gradient(x)
            train_loss += loss
        print('====> Epoch: {} \nTraining loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.set_train(is_train=False)
        test_loss = 0
        for i, (x, _) in enumerate(self.test_loader):
            if self.cuda:
                x = x.cuda()
            with torch.no_grad():
                recon_x = self.model_eval(x)[0][0]
                test_loss += self.compute_loss_and_gradient(x)
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n],
                                        recon_x.reshape(self.args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.detach().cpu(),
                           os.path.join(OUTPUT_DIR, 'reconstruction_' + str(epoch) + '.png'),
                           nrow=n)

        print('Test set loss: {:.4f}'.format(test_loss))


class PyroVAEImpl(VAE):
    """
    Implementation of VAE using Pyro. Only the model and the guide specification
    is needed to run the optimizer (the objective function does not need to be
    specified as in the PyTorch implementation).
    """

    def __init__(self, *args, **kwargs):
        pyro.clear_param_store()
        self.population_size = kwargs.pop('population_size')
        self.batch_size = kwargs.pop('batch_size')
        self.selection_size = kwargs.pop('selection_size')
        self.num_particles = kwargs.pop('num_particles')
        self.decay_schedule = kwargs.pop('decay_schedule')
        self.user_inputs = kwargs.pop('user_inputs')
        self.mutation_val = self.decay_schedule[0][0]
        self._t = 0
        self._t_prev = None
        self.optim_type = kwargs.pop('optim')
        super(PyroVAEImpl, self).__init__(*args, **kwargs)
        if self.optim_type == 'ea':
            self.optimizer = self.ea_optimizer()
        else:
            self.optimizer = self.svi_optimizer()

    def model(self, data):
        decoder = pyro.module('decoder', self.vae_decoder)
        with pyro.iarange('data', data.size(0), dim=-3):
            with pyro.iarange('zdim', 20, dim=-2):
                z = pyro.sample('latent', dist.Normal(data.new_tensor(0.), data.new_tensor(1.)))
                img = decoder.forward(z.squeeze(-1)).unsqueeze(-2)
            with pyro.iarange('img_shape', 784, dim=-1):
                data = data.reshape(data.size(0), 784).unsqueeze(-2)
                pyro.sample('obs',
                            dist.Bernoulli(img),
                            obs=data)

    def guide(self, data):
        encoder = pyro.module('encoder', self.vae_encoder)
        with pyro.iarange('data', data.size(0), dim=-3):
            with pyro.iarange('zdim', 20, dim=-2):
                z_mean, z_var = encoder.forward(data)
                z_mean = z_mean.unsqueeze(-1)
                z_var = z_var.unsqueeze(-1)
                pyro.sample('latent', dist.Normal(z_mean, z_var))

    def compute_loss_and_gradient(self, x):
        if self.mode == TRAIN:
            if self.optim_type == 'ea':
                loss = self.optimizer.step(x)
            else:
                loss = self.optimizer.step(x) / self.population_size
            self._t += 1
        else:
            if self.optim_type == 'ea':
                loss = self.optimizer.evaluate_loss(x)[0]
            else:
                loss = self.optimizer.evaluate_loss(x) / self.population_size
        print("ELBO loss: {}".format(loss))
        loss /= self.args.batch_size * 784
        return loss

    def mutation_fns(self, param):
        if self._t == self._t_prev:
            return lambda x: dist.Normal(x, x.new_tensor(self.mutation_val)).sample()
        if self.user_inputs and self._t % 400 == 0:
            decay = raw_input("decay: ")
            mutation_val = raw_input("mutation_val: ")
            if decay:
                if not mutation_val:
                    mutation_val = self.mutation_val
                self.decay_schedule = [(float(mutation_val), float(decay))]
                self.mutation_val = self.decay_schedule[0][0]
        decay = 0.999
        for mutation_val, decay in self.decay_schedule:
            if self.mutation_val <= mutation_val:
                decay = decay
        self.mutation_val = decay * self.mutation_val
        print("mutation: {}".format(self.mutation_val))
        self._t_prev = self._t
        return lambda x: dist.Normal(x, x.new_tensor(self.mutation_val)).sample()

    def ea_optimizer(self):
        loss = Parallelized_ELBO(self.model,
                                 self.guide,
                                 num_chains=self.population_size,
                                 max_iarange_nesting=3)
        return Evolution(loss,
                         self.mutation_fns,
                         num_particles=self.num_particles,
                         population_size=self.population_size,
                         selection_size=self.selection_size)

    def svi_optimizer(self):
        optimizer = Adam({'lr': 0.001})
        return SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(), max_iarange_nesting=3)


def setup(args):
    pyro.set_rng_seed(args.rng_seed)
    train_loader = util.get_data_loader(dataset_name='MNIST',
                                        data_dir=DATA_DIR,
                                        batch_size=args.batch_size,
                                        is_training_set=True,
                                        shuffle=True)
    test_loader = util.get_data_loader(dataset_name='MNIST',
                                       data_dir=DATA_DIR,
                                       batch_size=args.batch_size,
                                       is_training_set=False,
                                       shuffle=True)
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(RESULTS_DIR, 'ea')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    pyro.clear_param_store()
    return train_loader, test_loader


def main(args):
    train_loader, test_loader = setup(args)
    vae = PyroVAEImpl(args,
                      train_loader,
                      test_loader,
                      optim=args.optim,
                      cuda=args.cuda,
                      decay_schedule=list(zip([float(x) for x in args.mutation_schedule],
                                              [float(x) for x in args.decay_schedule])),
                      user_inputs=args.user_inputs,
                      num_particles=args.num_particles,
                      batch_size=args.batch_size,
                      population_size=args.population_size,
                      selection_size=args.selection_size)
    print('Running VAE implementation using: {}'.format(args.optim))
    for i in range(args.num_epochs):
        vae.train(i)
        if not args.skip_eval:
            vae.test(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE using MNIST dataset')
    parser.add_argument('-n', '--num-epochs', nargs='?', default=2, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch-size', nargs='?', default=128, type=int)
    parser.add_argument('--rng-seed', nargs='?', default=0, type=int)
    parser.add_argument('--num-particles', nargs='?', default=30, type=int)
    parser.add_argument('-d', '--decay-schedule', action='append', default=[0.995])
    parser.add_argument('-m', '--mutation-schedule', action='append', default=[0.01])
    parser.add_argument('--user-inputs', action='store_true')
    parser.add_argument('--population-size', default=300, type=int)
    parser.add_argument('--selection-size', default=30, type=int)
    parser.add_argument('--optim', default='ea', type=str)
    parser.add_argument('--skip-eval', action='store_true')
    parser.set_defaults(skip_eval=False)
    parser.set_defaults(cuda=False)
    parser.set_defaults(user_inputs=False)
    args = parser.parse_args()
    main(args)
