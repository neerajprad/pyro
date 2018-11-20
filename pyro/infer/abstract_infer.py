from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass

import pyro.poutine as poutine
from pyro.distributions import Categorical, Empirical
from pyro.ops.stats import split_gelman_rubin, effective_sample_size


class EmpiricalMarginal(Empirical):
    """
    Marginal distribution, that wraps over a TracePosterior object to provide a
    a marginal over one or more latent sites or the return values of the
    TracePosterior's model. If multiple sites are specified, they must have the
    same tensor shape.

    :param TracePosterior trace_posterior: a TracePosterior instance representing
        a Monte Carlo posterior.
    :param list sites: optional list of sites for which we need to generate
        the marginal distribution. Note that for multiple sites, the shape
        for the site values must match (needed by the underlying ``Empirical``
        class).
    """
    def __init__(self, trace_posterior, sites=None, validate_args=None):
        assert isinstance(trace_posterior, TracePosterior), \
            "trace_dist must be trace posterior distribution object"
        super(EmpiricalMarginal, self).__init__(validate_args=validate_args)
        if sites is None:
            sites = "_RETURN"
        self._populate_traces(trace_posterior, sites)

    def _populate_traces(self, trace_posterior, sites):
        assert isinstance(sites, (list, str))
        for tr, log_weight in zip(trace_posterior.exec_traces, trace_posterior.log_weights):
            value = tr.nodes[sites]["value"] if isinstance(sites, str) else \
                torch.stack([tr.nodes[site]["value"] for site in sites], 0)
            self.add(value, log_weight=log_weight)


@add_metaclass(ABCMeta)
class TracePosterior(object):
    """
    Abstract TracePosterior object from which posterior inference algorithms inherit.
    When run, collects a bag of execution traces from the approximate posterior.
    This is designed to be used by other utility classes like `EmpiricalMarginal`,
    that need access to the collected execution traces.
    """
    def __init__(self, num_chains=1):
        self.num_chains = num_chains
        self._reset()

    def _reset(self):
        self.log_weights = []
        self.exec_traces = []
        # For each chain, store sample indices that correspond to the chain
        self.idxs_by_chain = []
        self._categorical = None

    @abstractmethod
    def _traces(self, *args, **kwargs):
        """
        Abstract method implemented by classes that inherit from `TracePosterior`.

        :return: Generator over ``(exec_trace, weight)``.
        """
        raise NotImplementedError("inference algorithm must implement _traces")

    def __call__(self, *args, **kwargs):
        random_idx = self.idxs_by_chain[:][self._categorical.sample()]
        trace = self.exec_traces[random_idx].copy()
        for name in trace.observation_nodes:
            trace.remove_node(name)
        return trace

    def run(self, *args, **kwargs):
        """
        Calls `self._traces` to populate execution traces from a stochastic
        Pyro model.

        :param args: optional args taken by `self._traces`.
        :param kwargs: optional keywords args taken by `self._traces`.
        """
        self._reset()
        with poutine.block():
            for i, vals in enumerate(self._traces(*args, **kwargs)):
                if self.num_chains > 1:
                    tr, logit, chain_id = vals
                    assert chain_id < self.num_chains
                else:
                    tr, logit = vals
                    chain_id = 0
                self.exec_traces.append(tr)
                self.log_weights.append(logit)
                self.ids_by_chain[chain_id].append(i)
        self._categorical = Categorical(logits=torch.tensor(self.log_weights))
        return self

    def diagnostics(self, sites=None):
        if sites is None:
            assert len(self.exec_traces) > 0
            sites = self.exec_traces[0].stochastic_nodes()
        diagnostics = {}
        for site in sites:
            marginal, weights = EmpiricalMarginal(self, sites=[site]).get_samples_and_weights()
            if weights.max() != weights.min():
                raise ValueError("Diagnostics not implemented for differently weighted samples.")
            chain_samples = []
            for idxs in self.idxs_by_chain:
                sample_idxs = torch.tensor(idxs, dtype=torch.int64, device=marginal.device())
                chain_samples.append(marginal[sample_idxs])
            chain_samples = torch.stack(chain_samples)
            diagnostics[site] = {
                "n_eff": effective_sample_size(chain_samples),
                "r_hat": split_gelman_rubin(chain_samples),
            }
        return diagnostics


class TracePredictive(TracePosterior):
    """
    Generates and holds traces from the posterior predictive distribution,
    given model execution traces from the approximate posterior. This is
    achieved by constraining latent sites to randomly sampled parameter
    values from the model execution traces and running the model forward
    to generate traces with new response ("_RETURN") sites.

    :param model: arbitrary Python callable containing Pyro primitives.
    :param TracePosterior posterior: trace posterior instance holding
        samples from the model's approximate posterior.
    :param int num_samples: number of samples to generate.
    """
    def __init__(self, model, posterior, num_samples):
        self.model = model
        self.posterior = posterior
        self.num_samples = num_samples
        super(TracePredictive, self).__init__()

    def _traces(self, *args, **kwargs):
        if not self.posterior.exec_traces:
            self.posterior.run(*args, **kwargs)
        for _ in range(self.num_samples):
            model_trace = self.posterior()
            replayed_trace = poutine.trace(poutine.replay(self.model, model_trace)).get_trace(*args, **kwargs)
            yield (replayed_trace, 0.)
