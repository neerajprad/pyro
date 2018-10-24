import pyro
import torch
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc import HMC, MCMC


def model():
    true_probs = torch.tensor([0.9, 0.1]).cuda()
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    # wrapped by `pyro.param` to test if it works
    alpha = pyro.param('alpha', true_probs.new_tensor([1.1, 1.1]))
    beta = pyro.param('beta', true_probs.new_tensor([1.1, 1.1]))
    p_latent = pyro.sample('p_latent', dist.Beta(alpha, beta))
    pyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
    return p_latent


if __name__ == "__main__":
    hmc_kernel = HMC(model, trajectory_length=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=800, warmup_steps=500, num_chains=2).run()
    posterior = EmpiricalMarginal(mcmc_run, sites='p_latent')
