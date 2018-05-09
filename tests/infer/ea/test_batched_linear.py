import torch
import torch.nn as nn

from pyro.infer.ea.batched_linear import BatchedLinear
import pyro.distributions as dist


class Encoder(nn.Module):
    def __init__(self, batches):
        super(Encoder, self).__init__()
        self.batches = batches
        self.fc1 = BatchedLinear(784, 400, batches)
        self.fc21 = BatchedLinear(400, 20, batches)
        self.fc22 = BatchedLinear(400, 20, batches)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.shape[0] != self.batches:
            x = x.expand(torch.Size((self.batches,)) + x.shape)
        x = x.reshape(self.batches, -1, 784)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), torch.exp(self.fc22(h1))


# VAE Decoder network
class Decoder(nn.Module):
    def __init__(self, batches):
        super(Decoder, self).__init__()
        self.fc3 = BatchedLinear(20, 400, batches)
        self.fc4 = BatchedLinear(400, 784, batches)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))


def test_batched_linear():
    encoder = Encoder(40)
    decoder = Decoder(40)
    img = torch.randn(5, 28, 28)
    mu, sigma = encoder.forward(img)
    assert mu.shape == torch.Size((40, 5, 20))
    img = decoder.forward(dist.Normal(mu, sigma).sample())
    assert img.shape == torch.Size((40, 5, 784))