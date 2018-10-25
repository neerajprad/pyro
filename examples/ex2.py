import torch
import torch.multiprocessing as mp


DEVICE = "cpu"


def proc(queue):
    queue.put(torch.FloatTensor([1.], device=DEVICE))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=proc, args=(queue,))
    p.daemon = True
    p.start()
    print("received: ", queue.get())
