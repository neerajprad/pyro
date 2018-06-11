import pytest
import torch
import logging
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import TraceEnum_ELBO, Trace_ELBO, SVI
from pyro.infer.ea.es import ES
from pyro.infer.ea.ga import GA
from pyro.infer.ea.parallelized_elbo import Parallelized_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine

logging.basicConfig(format='%(message)s')
logging.getLogger('pyro').setLevel(logging.INFO)


@pytest.mark.filterwarnings("ignore:Encountered NaN")
def test_normal_normal():
    population_size = 60

    def model(data):
        loc = torch.tensor([0., 0.])
        scale = torch.tensor([1., 1.])
        with pyro.iarange('components', 2, dim=-1):
            p_latent = pyro.sample('p_latent',
                                   dist.Normal(loc, scale))
        with pyro.iarange("data", size=data.shape[0], dim=-2):
            pyro.sample('obs',
                        dist.Normal(p_latent, 1.),
                        obs=data)
        return p_latent

    def guide(data):
        loc = pyro.param("loc", torch.zeros(population_size, 1, 1, 2))
        scale = pyro.param("scale", torch.ones(population_size, 1, 1, 2),
                           constraint=constraints.positive)
        assert loc.shape == torch.Size((population_size, 1, 1, 2))
        with pyro.iarange('components', 2, dim=-1):
            p_latent = pyro.sample('p_latent',
                                   dist.Normal(loc, scale))
        return p_latent

    true_loc = torch.tensor([0.9, 0.1])
    true_scale = torch.tensor([0.5, 0.5])
    data = dist.Normal(true_loc, true_scale).sample(sample_shape=(torch.Size((1000,))))

    def mutation_fns(param):
        return lambda x: dist.Normal(x, x.new_tensor(0.05)).sample()

    loss = Parallelized_ELBO(model, guide, num_particles=10, num_chains=population_size, max_iarange_nesting=2)
    evol = GA(loss, mutation_fns, population_size=population_size, selection_size=15)
    with pyro.validation_enabled(False):
        for i in range(100):
            evol.step(data)
    print(evol.elite)


@pytest.mark.filterwarnings("ignore:Encountered NaN")
def test_normal_normal_es():
    population_size = 300

    def model(data):
        loc = torch.tensor([0., 0.])
        scale = torch.tensor([1., 1.])
        with pyro.iarange('components', 2, dim=-1):
            p_latent = pyro.sample('p_latent',
                                   dist.Normal(loc, scale))
        with pyro.iarange("data", size=data.shape[0], dim=-2):
            pyro.sample('obs',
                        dist.Normal(p_latent, 1.),
                        obs=data)
        return p_latent

    def guide(data):
        loc = pyro.param("loc", torch.zeros(population_size, 1, 1, 2))
        scale = pyro.param("scale", torch.ones(population_size, 1, 1, 2),
                           constraint=constraints.positive)
        assert loc.shape == torch.Size((population_size, 1, 1, 2))
        with pyro.iarange('components', 2, dim=-1):
            p_latent = pyro.sample('p_latent',
                                   dist.Normal(loc, scale))
        return p_latent

    true_loc = torch.tensor([0.9, 0.1])
    true_scale = torch.tensor([0.5, 0.5])
    data = dist.Normal(true_loc, true_scale).sample(sample_shape=(torch.Size((1000,))))

    def mutation_fns(param):
        return lambda x: (0.05, dist.Normal(x.new_zeros(x.shape), x.new_tensor(0.05)).sample())

    loss = Parallelized_ELBO(model, guide, num_particles=100, num_chains=population_size, max_iarange_nesting=2)
    evol = ES(loss, mutation_fns, lr=1e-3)
    with pyro.validation_enabled(False):
        for i in range(2000):
            print(evol.step(data))
            print(evol.parents)


def test_normal_normal_elbo():
    def model(data):
        loc = torch.tensor([0., 0.])
        scale = torch.tensor([1., 1.])
        p_latent = pyro.sample('p_latent',
                               dist.Normal(loc, scale).independent(1))
        with pyro.iarange("data", size=data.shape[0], dim=-1):
            pyro.sample('obs',
                        dist.Normal(p_latent, 1.).independent(1),  # 100x1x2     (1000x2)
                        obs=data)
        return p_latent

    def guide(data):
        loc = pyro.param("loc", torch.tensor([0., 0.]))
        scale = pyro.param("scale", torch.tensor([1., 1.]), constraint=constraints.positive)
        p_latent = pyro.sample('p_latent',
                               dist.Normal(loc, scale).independent(1))
        return p_latent

    pyro.clear_param_store()
    true_loc = torch.tensor([-2.5, 2.5])
    true_scale = torch.tensor([0.5, 0.5])
    data = dist.Normal(true_loc, true_scale).sample(sample_shape=(torch.Size((10000,))))
    print("data std: {}".format(torch.std(data, dim=0)))
    elbo_loss = Trace_ELBO()
    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, elbo_loss)
    for i in range(800):
        print(svi.step(data))
    print(pyro.param("loc"))
    print(pyro.param("scale"))

    "Check ELBO with num_particles = 1"
    for _ in range(10):
        elbo_loss = Trace_ELBO().loss(model,
                                      poutine.condition(guide, {"loc": pyro.param("loc"),
                                                                "scale": pyro.param("scale")}),
                                      data)
        print(elbo_loss)
