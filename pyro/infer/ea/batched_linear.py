import math

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class BatchedLinear(nn.Module):
    r"""Batched linear module.
    """

    def __init__(self, in_features, out_features, batches, bias=True):
        super(BatchedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batches = batches
        self.weight = Parameter(torch.Tensor(batches, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(batches, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ret = input.matmul(self.weight)
        if self.bias is not None:
            ret = ret + self.bias
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, batches={}, bias={}'.format(
            self.in_features, self.out_features, self.batches, self.bias is not None
        )
