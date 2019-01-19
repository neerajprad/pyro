"""
This example shows how to marginalize out discrete model variables in Pyro.

This combines Stochastic Variational Inference (SVI) with a
variable elimination algorithm, where we use enumeration to exactly
marginalize out some variables from the ELBO computation. We might
call the resulting algorithm collapsed SVI or collapsed SGVB (i.e
collapsed Stochastic Gradient Variational Bayes). In the case where
we exactly sum out all the latent variables, this algorithm reduces
to a form of gradient-based Maximum Likelihood Estimation.

To marginalize out discrete variables ``x`` in Pyro's SVI:

1. Verify that the variable dependency structure in your model
    admits tractable inference, i.e. the dependency graph among
    enumerated variables should have narrow treewidth.
2. Annotate each target each such sample site in the model
    with ``infer={"enumerate": "parallel"}``
3. Ensure your model can handle broadcasting of the sample values
    of those variables
4. Use the ``TraceEnum_ELBO`` loss inside Pyro's ``SVI``.
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import sys
from time import time

import torch
import torch.nn as nn
from torch.distributions import constraints

import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, infer_discrete
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO, stream=sys.stdout)


MODELS = [
    {'cuda': False},
    {'cuda': True},
]


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 88 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y. We'll drive
# probs_* with the guide, enumerate over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has
# narrow treewidth, therefore admitting efficient inference by message passing.
# Pyro's TraceEnum_ELBO will find an efficient message passing scheme if one
# exists.
def model_0(sequences, lengths, args, batch_size=None, include_prior=True):
    assert not torch._C._get_tracing_state()
    num_sequences, max_length, data_dim = sequences.shape
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                              .to_event(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                              .expand([args.hidden_dim, data_dim])
                              .to_event(2))
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    for i in pyro.plate("sequences", len(sequences), batch_size):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro.markov(range(length)):
            # On the next line, we'll overwrite the value of x with an updated
            # value. If we wanted to record all x values, we could instead
            # write x[t] = pyro.sample(...x[t-1]...).
            x = pyro.sample("x_{}_{}".format(i, t), dist.Categorical(probs_x[x]),
                            infer={"enumerate": "parallel"})
            with tones_plate:
                pyro.sample("y_{}_{}".format(i, t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                            obs=sequence[t])


# To see how enumeration changes the shapes of these sample sites, we can use
# the Trace.format_shapes() to print shapes at each site:
# $ python examples/hmm.py -m 0 -n 1 -b 1 -t 5 --print-shapes
# ...
#  Sample Sites:
#   probs_x dist          | 16 16
#          value          | 16 16
#   probs_y dist          | 16 88
#          value          | 16 88
#     tones dist          |
#          value       88 |
# sequences dist          |
#          value        1 |
#   x_178_0 dist          |
#          value    16  1 |
#   y_178_0 dist    16 88 |
#          value       88 |
#   x_178_1 dist    16  1 |
#          value 16  1  1 |
#   y_178_1 dist 16  1 88 |
#          value       88 |
#   x_178_2 dist 16  1  1 |
#          value    16  1 |
#   y_178_2 dist    16 88 |
#          value       88 |
#   x_178_3 dist    16  1 |
#          value 16  1  1 |
#   y_178_3 dist 16  1 88 |
#          value       88 |
#   x_178_4 dist 16  1  1 |
#          value    16  1 |
#   y_178_4 dist    16 88 |
#          value       88 |
#
# Notice that enumeration (over 16 states) alternates between two dimensions:
# -2 and -3.  If we had not used pyro.markov above, each enumerated variable
# would need its own enumeration dimension.


# Next let's make our simple model faster in two ways: first we'll support
# vectorized minibatches of data, and second we'll support the PyTorch jit
# compiler.  To add batch support, we'll introduce a second plate "sequences"
# and randomly subsample data to size batch_size.  To add jit support we
# silence some warnings and try to avoid dynamic program structure.
def model_1(sequences, lengths, args, num_sequences, include_prior=True):
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        n, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (n,)
        assert lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                              .to_event(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                              .expand([args.hidden_dim, data_dim])
                              .to_event(2))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    # We subsample batch_size items out of num_sequences items. Note that since
    # we're using dim=-1 for the notes plate, we need to batch over a different
    # dimension, here dim=-2.
    with pyro.plate("sequences", num_sequences, subsample=sequences, dim=-2):
        x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                with tones_plate:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                                obs=sequences[:, t])


# Let's see how batching changes the shapes of sample sites:
# $ python examples/hmm.py -m 1 -n 1 -t 5 --batch-size=10 --print-shapes
# ...
#  Sample Sites:
#   probs_x dist             | 16 16
#          value             | 16 16
#   probs_y dist             | 16 88
#          value             | 16 88
#     tones dist             |
#          value          88 |
# sequences dist             |
#          value          10 |
#       x_0 dist       10  1 |
#          value    16  1  1 |
#       y_0 dist    16 10 88 |
#          value       10 88 |
#       x_1 dist    16 10  1 |
#          value 16  1  1  1 |
#       y_1 dist 16  1 10 88 |
#          value       10 88 |
#       x_2 dist 16  1 10  1 |
#          value    16  1  1 |
#       y_2 dist    16 10 88 |
#          value       10 88 |
#       x_3 dist    16 10  1 |
#          value 16  1  1  1 |
#       y_3 dist 16  1 10 88 |
#          value       10 88 |
#       x_4 dist 16  1 10  1 |
#          value    16  1  1 |
#       y_4 dist    16 10 88 |
#          value       10 88 |
#
# Notice that we're now using dim=-2 as a batch dimension (of size 10),
# and that the enumeration dimensions are now dims -3 and -4.


# Next let's add a dependency of y[t] on y[t-1].
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
#
def model_2(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                              .to_event(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                              .expand([args.hidden_dim, 2, data_dim])
                              .to_event(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with tones_plate as tones:
                    y = pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x, y, tones]),
                                    obs=sequences[batch, t]).long()


# Next consider a Factorial HMM with two hidden states.
#
#    w[t-1] ----> w[t] ---> w[t+1]
#        \ x[t-1] --\-> x[t] --\-> x[t+1]
#         \  /       \  /       \  /
#          \/         \/         \/
#        y[t-1]      y[t]      y[t+1]
#
# Note that since the joint distribution of each y[t] depends on two variables,
# those two variables become dependent. Therefore during enumeration, the
# entire joint space of these variables w[t],x[t] needs to be enumerated.
# For that reason, we set the dimension of each to the square root of the
# target hidden dimension.
def model_3(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    with poutine.mask(mask=include_prior):
        probs_w = pyro.sample("probs_w",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                              .to_event(1))
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                              .to_event(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                              .expand([hidden_dim, hidden_dim, data_dim])
                              .to_event(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        w, x = 0, 0
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                with tones_plate as tones:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])


# By adding a dependency of x on w, we generalize to a
# Dynamic Bayesian Network.
#
#     w[t-1] ----> w[t] ---> w[t+1]
#        |  \       |  \       |   \
#        | x[t-1] ----> x[t] ----> x[t+1]
#        |   /      |   /      |   /
#        V  /       V  /       V  /
#     y[t-1]       y[t]      y[t+1]
#
# Note that message passing here has roughly the same cost as with the
# Factorial HMM, but this model has more parameters.
def model_4(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    hidden_dim = int(args.hidden_dim ** 0.5)  # split between w and x
    hidden = torch.arange(hidden_dim, dtype=torch.long)
    with poutine.mask(mask=include_prior):
        probs_w = pyro.sample("probs_w",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                              .to_event(1))
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1)
                              .expand_by([hidden_dim])
                              .to_event(2))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                              .expand([hidden_dim, hidden_dim, data_dim])
                              .to_event(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        # Note the broadcasting tricks here: we declare a hidden torch.arange and
        # ensure that w and x are always tensors so we can unsqueeze them below,
        # thus ensuring that the x sample sites have correct distribution shape.
        w = x = torch.tensor(0, dtype=torch.long)
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                w = pyro.sample("w_{}".format(t), dist.Categorical(probs_w[w]),
                                infer={"enumerate": "parallel"})
                x = pyro.sample("x_{}".format(t),
                                dist.Categorical(probs_x[w.unsqueeze(-1), x.unsqueeze(-1), hidden]),
                                infer={"enumerate": "parallel"})
                with tones_plate as tones:
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[w, x, tones]),
                                obs=sequences[batch, t])


# Next let's consider a neural HMM model.
#
#     x[t-1] --> x[t] --> x[t+1]   } standard HMM +
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]   } neural likelihood
#
# First let's define a neural net to generate y logits.
class TonesGenerator(nn.Module):
    def __init__(self, args, data_dim):
        self.args = args
        self.data_dim = data_dim
        super(TonesGenerator, self).__init__()
        self.x_to_hidden = nn.Linear(args.hidden_dim, args.nn_dim)
        self.y_to_hidden = nn.Linear(args.nn_channels * data_dim, args.nn_dim)
        self.conv = nn.Conv1d(1, args.nn_channels, 3, padding=1)
        self.hidden_to_logits = nn.Linear(args.nn_dim, data_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # Check dimension of y so this can be used with and without enumeration.
        if y.dim() < 2:
            y = y.unsqueeze(0)

        # Hidden units depend on two inputs: a one-hot encoded categorical variable x, and
        # a bernoulli variable y. Whereas x will typically be enumerated, y will be observed.
        # We apply x_to_hidden independently from y_to_hidden, then broadcast the non-enumerated
        # y part up to the enumerated x part in the + operation.
        x_onehot = y.new_zeros(x.shape[:-1] + (self.args.hidden_dim,)).scatter_(-1, x, 1)
        y_conv = self.relu(self.conv(y.unsqueeze(-2))).reshape(y.shape[:-1] + (-1,))
        h = self.relu(self.x_to_hidden(x_onehot) + self.y_to_hidden(y_conv))
        return self.hidden_to_logits(h)


# We will create a single global instance later.
tones_generator = None


# The neural HMM model now uses tones_generator at each time step.
def model_5(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length

    # Initialize a global module instance if needed.
    global tones_generator
    if tones_generator is None:
        tones_generator = TonesGenerator(args, data_dim)
    pyro.module("tones_generator", tones_generator)

    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                              .to_event(1))
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        y = torch.zeros(data_dim)
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                # Note that since each tone depends on all tones at a previous time step
                # the tones at different time steps now need to live in separate plates.
                with pyro.plate("tones_{}".format(t), data_dim, dim=-1):
                    y = pyro.sample("y_{}".format(t),
                                    dist.Bernoulli(logits=tones_generator(x, y)),
                                    obs=sequences[batch, t])


# Next let's consider a second-order HMM model
# in which x[t+1] depends on both x[t] and x[t-1].
#
#                     _______>______
#         _____>_____/______        \
#        /          /       \        \
#     x[t-1] --> x[t] --> x[t+1] --> x[t+2]
#        |        |          |          |
#        V        V          V          V
#     y[t-1]     y[t]     y[t+1]     y[t+2]
#
#  Note that in this model (in contrast to the previous model) we treat
#  the transition and emission probabilities as parameters (so they have no prior).

def model_6(sequences, lengths, args, batch_size=None, include_prior=False):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length
    hidden_dim = args.hidden_dim
    hidden = torch.arange(hidden_dim, dtype=torch.long)

    if not args.raftery_parameterization:
        # Explicitly parameterize the full tensor of transition probabilities, which
        # has hidden_dim cubed entries.
        probs_x = pyro.param("probs_x", torch.rand(hidden_dim, hidden_dim, hidden_dim),
                             constraint=constraints.simplex)
    else:
        # Use the more parsimonious "Raftery" parameterization of
        # the tensor of transition probabilities. See reference:
        # Raftery, A. E. A model for high-order markov chains.
        # Journal of the Royal Statistical Society. 1985.
        probs_x1 = pyro.param("probs_x1", torch.rand(hidden_dim, hidden_dim),
                              constraint=constraints.simplex)
        probs_x2 = pyro.param("probs_x2", torch.rand(hidden_dim, hidden_dim),
                              constraint=constraints.simplex)
        mix_lambda = pyro.param("mix_lambda", torch.tensor(0.5), constraint=constraints.unit_interval)
        # we use broadcasting to combine two tensors of shape (hidden_dim, hidden_dim) and
        # (hidden_dim, 1, hidden_dim) to obtain a tensor of shape (hidden_dim, hidden_dim, hidden_dim)
        probs_x = mix_lambda * probs_x1 + (1.0 - mix_lambda) * probs_x2.unsqueeze(-2)

    probs_y = pyro.param("probs_y", torch.rand(hidden_dim, data_dim),
                         constraint=constraints.unit_interval)
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x_curr, x_prev = torch.tensor(0), torch.tensor(0)
        # we need to pass the argument `history=2' to `pyro.markov()`
        # since our model is now 2-markov
        for t in pyro.markov(range(lengths.max()), history=2):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                probs_x_t = probs_x[x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), hidden]
                x_prev, x_curr = x_curr, pyro.sample("x_{}".format(t), dist.Categorical(probs_x_t),
                                                     infer={"enumerate": "parallel"})
                with tones_plate:
                    probs_y_t = probs_y[x_curr.squeeze(-1)]
                    pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y_t),
                                obs=sequences[batch, t])


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def run(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = poly.load_data(poly.JSB_CHORALES)

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequences'])))
    sequences = data['train']['sequences']
    lengths = data['train']['sequence_lengths']
    num_sequences = len(sequences)

    if args.batch_size:
        lengths = lengths[:args.batch_size]
        sequences = sequences[:args.batch_size]

    # find all the notes that are present at least once in the training set
    present_notes = torch.ones(88, dtype=torch.uint8, device=sequences.device)
    present_notes[args.clamp_notes:] = 0
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths.clamp_(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    first_available_dim = -2 if model is model_0 else -3
    if args.print_shapes:
        guide_trace = poutine.trace(guide).get_trace(
            sequences, lengths, args=args, num_sequences=num_sequences)
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences, lengths, args=args, num_sequences=num_sequences)
        logging.info(model_trace.format_shapes())

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=1 if model is model_0 else 2)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.

    logging.info('Step\tLoss')
    # do not record time for the jit step
    svi.step(sequences, lengths, args=args, num_sequences=num_sequences)
    train_times, sim_times = [], []
    for step in range(args.num_steps):
        t_start = time()
        loss = svi.step(sequences, lengths, args=args, num_sequences=num_sequences)
        t_end = time()
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))
        train_time = t_end - t_start
        logging.info('train time = {}'.format(train_time))
        train_times.append(train_time)

    inferred_model = infer_discrete(model, first_available_dim=first_available_dim - 1, temperature=0)

    for _ in range(args.num_steps):
        t_start = time()
        with pyro.validation_enabled(False):
            poutine.trace(inferred_model).get_trace(sequences, lengths, args=args,
                                                    num_sequences=num_sequences)
        t_end = time()
        sim_time = t_end - t_start
        logging.info('sim time = {}'.format(sim_time))
        sim_times.append(sim_time)
    return train_times, sim_times


def profile_hidden_dim(args):
    for model in MODELS:
        out_file = "hidden_dim_cuda" if model["cuda"] else "hidden_dim_cpu"
        if model["cuda"]:
            args.cuda = True
        with open(out_file + '.csv', 'w', newline='') as f:
            fieldnames = ["hidden dim size", "time per step", "event"]
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for hidden_dim in range(4, 102, 4):
                args.hidden_dim = hidden_dim
                print("Profiling hidden dim size = {}, args = {}".format(hidden_dim, args))
                train_times, sim_times = run(args)
                for t in train_times:
                    writer.writerow([hidden_dim, t, "training"])
                for t in sim_times:
                    writer.writerow([hidden_dim, t, "simulation"])


def profile_batch_size(args):
    for model in MODELS:
        out_file = "batch_size_cuda" if model["cuda"] else "batch_size_cpu"
        if model["cuda"]:
            args.cuda = True
        with open(out_file + '.csv', 'w', newline='') as f:
            fieldnames = ["batch size", "time per step", "event"]
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for batch_size in range(4, 102, 4):
                args.batch_size = batch_size
                print("Profiling batch size = {}, args = {}".format(batch_size, args))
                train_times, sim_times = run(args)
                for t in train_times:
                    writer.writerow([batch_size, t, "training"])
                for t in sim_times:
                    writer.writerow([batch_size, t, "simulation"])


def profile_plate_dim(args):
    for model in MODELS:
        out_file = "plate_dim_cuda" if model["cuda"] else "plate_dim_cpu"
        if model["cuda"]:
            args.cuda = True
        with open(out_file + '.csv', 'w', newline='') as f:
            fieldnames = ["plate dim size", "time per step", "event"]
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for plate_dim in range(4, 89, 4):
                args.clamp_notes = plate_dim
                print("Profiling plate dim size = {}, args = {}".format(plate_dim, args))
                train_times, sim_times = run(args)
                for t in train_times:
                    writer.writerow([plate_dim, t, "training"])
                for t in sim_times:
                    writer.writerow([plate_dim, t, "simulation"])


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-d", "--hidden-dim", default=32, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true', default=True)
    parser.add_argument("--clamp-notes", type=int, default=88)
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    parser.add_argument('--profile', type=str, default='batch_size')
    args = parser.parse_args()
    if args.profile == "hidden_dim":
        profile_hidden_dim(args)
    elif args.profile == "batch_size":
        profile_batch_size(args)
    elif args.profile == "plate_dim":
        profile_plate_dim(args)
