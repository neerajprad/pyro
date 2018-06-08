import torch

import pyro
import pyro.distributions as dist
from pyro.infer.ea.parallelized_elbo import Parallelized_ELBO


def test_parallel_elbo():
    data = torch.ones(1000, 2)

    def model(data):
        with pyro.iarange("components", 2, dim=-1):
            p = pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
        with pyro.iarange("data", data.shape[0], dim=-2):
            pyro.sample("obs", dist.Bernoulli(p), obs=data)

    def guide(data):
        with pyro.iarange("components", 2, dim=-1):
            pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))

    pyro.clear_param_store()
    elbo = Parallelized_ELBO(model,
                             guide,
                             num_particles=10,
                             num_chains=5,
                             max_iarange_nesting=2,
                             strict_enumeration_warning=True)
    assert elbo.get_loss(data).shape == torch.Size((5,))
