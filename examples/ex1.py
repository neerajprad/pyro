import torch

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS


def model(data):
    alpha = pyro.param('alpha', torch.tensor([1.1, 1.1]))
    beta = pyro.param('beta', torch.tensor([1.1, 1.1]))
    p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
    pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
    return p_latent

pyro.set_rng_seed(0)
true_probs = torch.tensor([0.9, 0.1])
data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
kernel = NUTS(model, step_size=0.02)
mcmc_run = MCMC(kernel, num_samples=300, warmup_steps=100).run(data)
EmpiricalMarginal(mcmc_run, sites='p_latent')


