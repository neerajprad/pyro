from __future__ import absolute_import, division, print_function


import logging

import sys
import torch

import pyro
import pyro.poutine as poutine


class Evolution(object):
    """
    A common interface for using Genetic Algorithms in Pyro.

    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param loss: evaluation function to minimize
    :param mutation_fns: mutation function per sample site
    :param population_size: population size at each generation
    :param selection_size: number of individuals selected at each
        generation to advance to the next.
    """
    def __init__(self,
                 elbo,
                 mutation_fns,
                 population_size,
                 selection_size,
                 num_particles=1,
                 inheritance_decay=1.):
        self.elbo = elbo
        self.population_size = population_size
        self.selection_size = selection_size
        self.mutation_fns = mutation_fns
        self.num_particles = num_particles
        self.decay = inheritance_decay
        self.logger = logging.getLogger(__name__)
        self._reset()

    def _reset(self):
        self.parents = []
        self.elite = None
        self.elite_loss = None
        self.parent_loss = None
        self._t = 0

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        return self.elbo.get_loss(*args, **kwargs)

    def _mutate(self, param_sites, elite):
        true_params = {}
        with torch.no_grad():
            if isinstance(self.mutation_fns, dict):
                mutfn = self.mutation_fns.get
            else:
                mutfn = self.mutation_fns
            for site, value in param_sites.items():
                true_param = pyro.get_param_store().get_param(site).unconstrained()
                true_param.zero_()
                mutated = value
                mutated = mutfn(site)(mutated)
                if elite:
                    mutated = torch.cat([elite[site].unsqueeze(0), mutated], dim=0)
                true_param += mutated
                true_params[site] = true_param
        return true_params

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
        initial_candidates = {}
        parent_idxs = None
        if not self.parents:
            with poutine.trace(param_only=True) as param_capture:
                self.evaluate_loss(*args, **kwargs)
            initial_candidates = {site["name"]: site["value"].unconstrained().detach().clone() for site in
                                  param_capture.trace.nodes.values()}
            print("params size: {}".format(sys.getsizeof(initial_candidates)))
        else:
            for k, v in self.parents.items():
                value = v.detach().clone()
                num_candidates = value.shape[0]
                parent_idxs = torch.randint(0,
                                            num_candidates,
                                            (self.population_size-1,)).type(torch.long)
                initial_candidates[k] = value[parent_idxs]

        # add elite to population
        if parent_idxs is not None:
            parent_idxs = torch.cat([torch.LongTensor([0]), parent_idxs])
        population = self._mutate(initial_candidates, self.elite)

        losses = [self.evaluate_loss(*args, **kwargs).detach()]
        if self.num_particles > 1:
            for _ in range(self.num_particles-1):
                losses.append(self.evaluate_loss(*args, **kwargs).detach())
        losses = torch.stack(losses, dim=0).mean(dim=0)
        if self.parent_loss is not None:
            parent_losses = self.parent_loss[parent_idxs]
            losses = (1 - self.decay) * parent_losses + losses
        sorted_losses, sort_index = torch.sort(losses)
        top_n = sort_index[:self.selection_size]
        top_loss = losses[top_n]
        self.logger.info("\nGeneration: {}".format(self._t))
        self._log_summary("Overall population", sorted_losses)
        self._log_summary("Selected population", top_loss)
        if self.parent_loss is not None:
            self._log_summary("Parent population", self.parent_loss)
        next_generation = {k: v[top_n] for k, v in population.items()}
        self.elite, self.parents = {}, {}
        for k, v in next_generation.items():
            self.elite[k] = v[0]
            self.parents[k] = v[1:]
        self.elite_loss = sorted_losses[0]
        self.parent_loss = sorted_losses
        self._t += 1
        return self.elite_loss.clone()
