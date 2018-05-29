from __future__ import absolute_import, division, print_function


import logging

import sys
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
                 mutation_fns,
                 population_size,
                 num_particles=1,
                 inheritance_decay=1.):
        self.elbo = elbo
        self.population_size = population_size
        self.mutation_fns = mutation_fns
        self.num_particles = num_particles
        self.decay = inheritance_decay
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

    def _mutate(self, parents):
        var = {}
        noise = {}
        with torch.no_grad():
            for site, value in parents.items():
                true_param = pyro.get_param_store().get_param(site).unconstrained()
                true_param.zero_()
                var[site], noise[site] = self.mutation_fns(site)
                mutated = value + noise[site]
                true_param += mutated
        return noise, var

    def _log_summary(self, name, losses):
        self.logger.info(name)
        self.logger.info("min: {}\tp25: {}\tp50: {}\tp75:{}\tmax: {}"
                         .format(losses[0],
                                 losses[int(len(losses)/4.)],
                                 losses[int(len(losses)/2.)],
                                 losses[int(len(losses) * 3/4)],
                                 losses[len(losses) - 1],
                                 ))

    def step(self, *args, **kwargs):
        parents = {}
        if not self.parents:
            with poutine.trace(param_only=True) as param_capture:
                self.evaluate_loss(*args, **kwargs)
            population = {site["name"]: site["value"].unconstrained().detach().clone() for site in
                                  param_capture.trace.nodes.values()}
            parents = {site["name"]: site["value"].mean(dim=0) for site in population}
            self.optim = torch.optim.Adam(parents.values, lr=1e-3)
        else:
            parents = self.parents

        noise, var = self._mutate(parents)

        losses = [self.evaluate_loss(*args, **kwargs).detach()]
        if self.num_particles > 1:
            for _ in range(self.num_particles-1):
                losses.append(self.evaluate_loss(*args, **kwargs).detach())
        losses = torch.stack(losses, dim=0).mean(dim=0)
        for k, v in parents.items():
            v.grad = torch.mean(1/var[k] * noise[k] * losses)
        self.optim.step()
        self.parents = parents
        self.parent_loss = self.evaluate_parent_loss(self.parents)
        return self.parent_loss.clone()
