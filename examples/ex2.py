import torch
import torch.multiprocessing as mp


N = 2
DEVICE = "cpu"


def fn(x):
    x["in"] = x["out"]
    x["out"] = x["out"] + 1
    return x


def stream():
    x = {"in": torch.tensor(1., device=DEVICE), "out": torch.tensor(1., device=DEVICE)}
    for _ in range(10):
        x = fn(x)
        yield x


def proc(id, queue):
    for x in stream():
        queue.put((id, x))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue()
    procs = []
    for i in range(N):
        procs.append(ctx.Process(name=str(i), target=proc, args=(i, queue)))
    for p in procs:
        p.daemon = True
        p.start()
    for _ in range(N * 10):
        print("received: ", queue.get())