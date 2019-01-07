import torch
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import HMC, MCMC

true_probs = torch.tensor([0.9, 0.1])
data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))

def model():
    alpha = torch.tensor([1.1, 1.1])
    beta = torch.tensor([1.1, 1.1])
    p_latent = pyro.sample('p_latent', dist.Beta(alpha, beta))
    with pyro.plate("data", data.shape[0], dim=-2):
        pyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
    return p_latent


hmc_kernel = HMC(model, trajectory_length=1, max_plate_nesting=2)
mcmc_run = MCMC(hmc_kernel, num_samples=800, warmup_steps=500, num_chains=2).run()
posterior = mcmc_run.marginal(["p_latent"]).empirical["p_latent"]
