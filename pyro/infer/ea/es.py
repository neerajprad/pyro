from __future__ import absolute_import, division, print_function


import logging

import torch

import pyro
import pyro.poutine as poutine


class ES(object):
    """
    A common interface for using Genetic Algorithms in Pyro.

    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param loss: evaluation function to minimize
    :param mutation_fns: mutation function per sample site
    :param population_size: population size at each generation
    """
    def __init__(self,
                 elbo,
                 mutation_fns):
        self.elbo = elbo
        self.mutation_fns = mutation_fns
        self.logger = logging.getLogger(__name__)
        self._reset()

    def _reset(self):
        self.parents = []
        self.parent_loss = None
        self._t = 0

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        return self.elbo.get_loss(*args, **kwargs)

    def evaluate_parent_loss(self, data, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        return self.elbo.get_loss_conditioned_on(data, *args, **kwargs)

    def _mutate(self, parents, copy=False):
        var = {}
        noise = {}
        with torch.no_grad():
            for site, value in parents.items():
                true_param = pyro.get_param_store().get_param(site).unconstrained()
                true_param.zero_()
                if not copy:
                    var[site], noise[site] = self.mutation_fns(site)(true_param)
                    mutated = value + noise[site]
                    true_param += mutated
                else:
                    true_param += value
        return noise, var

    def step(self, *args, **kwargs):
        if not self.parents:
            with poutine.trace(param_only=True) as param_capture:
                self.evaluate_loss(*args, **kwargs)
            population = {site["name"]: site["value"].unconstrained().detach().clone() for site in
                                  param_capture.trace.nodes.values()}
            parents = {k: v.mean(dim=0).requires_grad_(True) for k, v in population.items()}
            self.optim = torch.optim.Adam(parents.values(), lr=1e-3)
        else:
            parents = {k: v.detach().requires_grad_(True) for k,v in self.parents.items()}
        self.optim.zero_grad()

        noise, var = self._mutate(parents)

        losses = self.evaluate_loss(*args, **kwargs).detach()
        for k, v in parents.items():
            v.grad = -torch.stack([1/var[k] * noise[k][i] * losses[i] for i in range(len(losses))]).mean(0)
        self.optim.step()
        self.parents = parents
        self._mutate(parents, copy=True)
        self.parent_loss = self.evaluate_loss(*args, **kwargs).detach().mean()
        return self.parent_loss.clone()
