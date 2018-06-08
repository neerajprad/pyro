import warnings

import torch

import pyro
from pyro.distributions.util import sum_rightmost, scale_tensor
from pyro.infer import Trace_ELBO, is_validation_enabled
import pyro.poutine as poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, torch_isnan


class Parallelized_ELBO(Trace_ELBO):
    def __init__(self,
                 model,
                 guide,
                 num_particles=1,
                 num_chains=1,
                 max_iarange_nesting=float('inf'),
                 strict_enumeration_warning=True):
        self.model = model
        self.guide = guide
        self.num_chains = num_chains
        self.num_particles = num_particles
        self.max_iarange_nesting = max_iarange_nesting
        self.chain_dim = None
        self.model, self.guide = self._parallelize([self.model, self.guide])
        super(Parallelized_ELBO, self).__init__(num_particles=self.num_particles,
                                                max_iarange_nesting=self.max_iarange_nesting,
                                                strict_enumeration_warning=strict_enumeration_warning)

    def _parallelize(self, fns):
        def _parallelized(fn, dim):
            def parallel_model(*args, **kwargs):
                with pyro.iarange("num_chains", self.num_chains, dim=dim-1):
                    with pyro.iarange("num_particles", self.num_particles, dim=dim):
                        return fn(*args, **kwargs)
            return parallel_model

        parallelized_fns = fns
        dim = -(self.max_iarange_nesting + 1)
        for i in range(len(parallelized_fns)):
            parallelized_fns[i] = _parallelized(parallelized_fns[i], dim)
        self.max_iarange_nesting += 2
        self.chain_dim = self.max_iarange_nesting
        return [poutine.broadcast(fn) for fn in parallelized_fns]

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace)
            enumerated_sites = [name for name, site in guide_trace.nodes.items()
                                if site["type"] == "sample" and site["infer"].get("enumerate")]
            if enumerated_sites:
                warnings.warn('\n'.join([
                    'Trace_ELBO found sample sites configured for enumeration:'
                    ', '.join(enumerated_sites),
                    'If you want to enumerate sites, you need to use TraceEnum_ELBO instead.']))
        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)

        model_trace.compute_log_prob()
        guide_trace.compute_score_parts()
        if is_validation_enabled():
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)

        return model_trace, guide_trace

    def trace_log_prob(self, trace, rightmost=float("inf")):
        log_p = []
        for name, site in trace.nodes.items():
            if site["type"] == "sample":
                args, kwargs = site["args"], site["kwargs"]
                site_log_p = site["fn"].log_prob(site["value"], *args, **kwargs)
                site_log_p = sum_rightmost(scale_tensor(site_log_p, site["scale"]), rightmost)
                log_p.append(site_log_p)
        return torch.stack(log_p, dim=-1).sum(dim=-1)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        model_trace, guide_trace = self._get_traces(model, guide, *args, **kwargs)
        elbo = (self.trace_log_prob(model_trace, self.chain_dim - 1) -
                self.trace_log_prob(guide_trace, self.chain_dim - 1)) / self.num_particles
        if self.num_chains > 1:
            assert elbo.shape == torch.Size((self.num_chains,))

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def get_loss(self, *args, **kwargs):
        return self.loss(self.model, self.guide, *args, **kwargs)
