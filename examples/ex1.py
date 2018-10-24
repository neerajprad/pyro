import torch


def _process(queue):
    input_ = queue.get()
    print('get')
    queue.put(input_)
    print('put')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    input_ = torch.ones(1).cuda()
    queue = torch.multiprocessing.Queue()
    process = torch.multiprocessing.Process(target=_process, args=(queue,))
    process.start()
    queue.put(input_)
    process.join()
    result = queue.get()
    print('end')
    print(result)