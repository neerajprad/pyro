import torch
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import HMC, MCMC
from pyro.infer.mcmc.trace_kernel import TraceKernel
import pyro.poutine as poutine

data = torch.tensor([1.0], device='cuda')


class PriorKernel(TraceKernel):
    """
    Disregards the value of the current trace (or observed data) and
    samples a value from the model's prior.
    """
    def __init__(self, model):
        self.model = model
        self.data = None

    def setup(self, warmup_steps, data):
        self.data = data

    def cleanup(self):
        self.data = None

    def initial_trace(self):
        return poutine.trace(self.model).get_trace(self.data)

    def sample(self, trace):
        return self.initial_trace()


def normal_normal_model(data):
    x = torch.tensor([0.0])
    y = pyro.sample('y', dist.Normal(x, torch.tensor([1.0], device='cuda')))
    pyro.sample('obs', dist.Normal(y, torch.tensor([1.0]), device='cuda'), obs=data)
    return y


if __name__ == "__main__":
    kernel = PriorKernel(normal_normal_model)
    mcmc = MCMC(kernel=kernel, num_samples=800, warmup_steps=100, num_chains=2, mp_context='spawn').run(data)
    marginal = mcmc.marginal().empirical["_RETURN"]
