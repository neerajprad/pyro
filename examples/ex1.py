import torch


@torch.jit.trace(*([torch.randn(100, 10) for _ in range(10)]))
def ke(*r):
    sum_r = 0.
    for x in r:
        sum_r += x.pow(2).sum()
    return 0.5 * sum_r


for i in range(100):
    arr = []
    for _ in range(10):
        arr.append(torch.randn(100, 10))

    print(ke(*arr))