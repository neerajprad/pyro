"""
This module offers a modified interface for MCMC inference with the following objectives:
  - making MCMC independent of Pyro specific trace data structure, to facilitate
    integration with other PyTorch based libraries.
  - bringing the interface closer to that of NumPyro to make it easier to write
    code that works with different backends.
  - minimal memory consumption with multiprocessing and CUDA.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import signal
import threading
import warnings
from collections import OrderedDict, defaultdict

import six
import torch
import torch.multiprocessing as mp
from six.moves import queue

import pyro
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.logger import initialize_logger, DIAGNOSTIC_MSG, TqdmHandler, ProgressBar
from pyro.infer.mcmc.util import diagnostics, initialize_model, summary

MAX_SEED = 2**32 - 1


def logger_thread(log_queue, warmup_steps, num_samples, num_chains, disable_progbar=False):
    """
    Logging thread that asynchronously consumes logging events from `log_queue`,
    and handles them appropriately.
    """
    progress_bars = ProgressBar(warmup_steps, num_samples, disable=disable_progbar, num_bars=num_chains)
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.addHandler(TqdmHandler())
    num_samples = [0] * num_chains
    try:
        while True:
            try:
                record = log_queue.get(timeout=1)
            except queue.Empty:
                continue
            if record is None:
                break
            metadata, msg = record.getMessage().split("]", 1)
            _, msg_type, logger_id = metadata[1:].split()
            if msg_type == DIAGNOSTIC_MSG:
                pbar_pos = int(logger_id.split(":")[-1])
                num_samples[pbar_pos] += 1
                if num_samples[pbar_pos] == warmup_steps:
                    progress_bars.set_description("Sample [{}]".format(pbar_pos + 1), pos=pbar_pos)
                diagnostics = json.loads(msg, object_pairs_hook=OrderedDict)
                progress_bars.set_postfix(diagnostics, pos=pbar_pos, refresh=False)
                progress_bars.update(pos=pbar_pos)
            else:
                logger.handle(record)
    finally:
        progress_bars.close()


class _Worker(object):
    def __init__(self, chain_id, result_queue, log_queue, kernel, num_samples, warmup_steps,
                 initial_params=None, hook=None):
        self.chain_id = chain_id
        self.kernel = kernel
        if initial_params is not None:
            self.kernel.initial_params = initial_params
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.rng_seed = (torch.initial_seed() + chain_id) % MAX_SEED
        self.log_queue = log_queue
        self.result_queue = result_queue
        self.default_tensor_type = torch.Tensor().type()
        self.hook = hook

    def run(self, *args, **kwargs):
        pyro.set_rng_seed(self.rng_seed)
        torch.set_default_tensor_type(self.default_tensor_type)
        # XXX we clone CUDA tensor args to resolve the issue "Invalid device pointer"
        # at https://github.com/pytorch/pytorch/issues/10375
        args = [arg.clone().detach() if (torch.is_tensor(arg) and arg.is_cuda) else arg for arg in args]
        kwargs = kwargs
        logger = logging.getLogger("pyro.infer.mcmc")
        logger_id = "CHAIN:{}".format(self.chain_id)
        log_queue = self.log_queue
        logger = initialize_logger(logger, logger_id, None, log_queue)
        logging_hook = _add_logging_hook(logger, None, self.hook)

        try:
            for sample in _gen_samples(self.kernel, self.warmup_steps, self.num_samples, logging_hook,
                                       *args, **kwargs):
                self.result_queue.put_nowait((self.chain_id, sample))
            self.result_queue.put_nowait((self.chain_id, None))
        except Exception as e:
            logger.exception(e)
            self.result_queue.put_nowait((self.chain_id, e))


def _gen_samples(kernel, warmup_steps, num_samples, hook, *args, **kwargs):
    kernel.setup(warmup_steps, *args, **kwargs)
    params = kernel.initial_params
    for i in range(warmup_steps):
        params = kernel.sample(params)
        hook(kernel, params, 'warmup', i)
    for i in range(num_samples):
        params = kernel.sample(params)
        hook(kernel, params, 'sample', i)
        yield params
    yield kernel.diagnostics()
    kernel.cleanup()


def _add_logging_hook(logger, progress_bar=None, hook=None):
    def _add_logging(kernel, params, stage, i):
        diagnostics = json.dumps(kernel.logging())
        logger.info(diagnostics, extra={"msg_type": DIAGNOSTIC_MSG})
        if progress_bar:
            progress_bar.set_description(stage, refresh=False)
        if hook:
            hook(kernel, params, stage, i)

    return _add_logging


class _UnarySampler(object):
    """
    Single process runner class optimized for the case `num_chains=1`.
    """

    def __init__(self, kernel, num_samples, warmup_steps, disable_progbar, initial_params=None, hook=None):
        self.kernel = kernel
        if initial_params is not None:
            self.kernel.initial_params = initial_params
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.logger = None
        self.disable_progbar = disable_progbar
        self.hook = hook
        super(_UnarySampler, self).__init__()

    def terminate(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        logger = logging.getLogger("pyro.infer.mcmc")
        progress_bar = ProgressBar(self.warmup_steps, self.num_samples, disable=self.disable_progbar)
        logger = initialize_logger(logger, "", progress_bar)
        hook_w_logging = _add_logging_hook(logger, progress_bar, self.hook)
        for sample in _gen_samples(self.kernel, self.warmup_steps, self.num_samples, hook_w_logging,
                                   *args, **kwargs):
            yield sample, 0  # sample, chain_id (default=0)
        progress_bar.close()


class _MultiSampler(object):
    """
    Parallel runner class for running MCMC chains in parallel. This uses the
    `torch.multiprocessing` module (itself a light wrapper over the python
    `multiprocessing` module) to spin up parallel workers.
    """
    def __init__(self, kernel, num_samples, warmup_steps, num_chains, mp_context,
                 disable_progbar, initial_params=None, hook=None):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.hook = hook
        self.workers = []
        self.ctx = mp
        if mp_context:
            if six.PY2:
                raise ValueError("multiprocessing.get_context() is "
                                 "not supported in Python 2.")
            self.ctx = mp.get_context(mp_context)
        self.result_queue = self.ctx.Queue()
        self.log_queue = self.ctx.Queue()
        self.logger = initialize_logger(logging.getLogger("pyro.infer.mcmc"),
                                        "MAIN", log_queue=self.log_queue)
        self.num_samples = num_samples
        self.initial_params = initial_params
        self.log_thread = threading.Thread(target=logger_thread,
                                           args=(self.log_queue, self.warmup_steps, self.num_samples,
                                                 self.num_chains, disable_progbar))
        self.log_thread.daemon = True
        self.log_thread.start()

    def init_workers(self, *args, **kwargs):
        self.workers = []
        for i in range(self.num_chains):
            init_params = {k: v[i] for k, v in self.initial_params.items()} if self.initial_params is not None else None
            worker = _Worker(i, self.result_queue, self.log_queue, self.kernel, self.num_samples, self.warmup_steps,
                             initial_params=init_params, hook=self.hook)
            worker.daemon = True
            self.workers.append(self.ctx.Process(name=str(i), target=worker.run,
                                                 args=args, kwargs=kwargs))

    def terminate(self, terminate_workers=False):
        if self.log_thread.is_alive():
            self.log_queue.put_nowait(None)
            self.log_thread.join(timeout=1)
        # Only kill workers if exception is raised. worker processes are daemon
        # processes that will otherwise be terminated with the main process.
        # Note that it is important to not
        if terminate_workers:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()

    def run(self, *args, **kwargs):
        # Ignore sigint in worker processes; they will be shut down
        # when the main process terminates.
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.init_workers(*args, **kwargs)
        # restore original handler
        signal.signal(signal.SIGINT, sigint_handler)
        active_workers = self.num_chains
        exc_raised = True
        try:
            for w in self.workers:
                w.start()
            while active_workers:
                try:
                    chain_id, val = self.result_queue.get(timeout=5)
                except queue.Empty:
                    continue
                if isinstance(val, Exception):
                    # Exception trace is already logged by worker.
                    raise val
                if val is not None:
                    yield val, chain_id
                else:
                    active_workers -= 1
            exc_raised = False
        finally:
            self.terminate(terminate_workers=exc_raised)


class MCMC(object):
    """
    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.

    .. note:: The case of `num_chains > 1` uses python multiprocessing to
        run parallel chains in multiple processes. This goes with the usual
        caveats around multiprocessing in python, e.g. the model used to
        initialize the ``kernel`` must be serializable via `pickle`, and the
        performance / constraints will be platform dependent (e.g. only
        the "spawn" context is available in Windows). This has also not
        been extensively tested on the Windows platform.

    :param kernel: An instance of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded. If not provided, default is
        half of `num_samples`.
    :param int num_chains: Number of MCMC chains to run in parallel. Depending on
        whether `num_chains` is 1 or more than 1, this class internally dispatches
        to either `_UnarySampler` or `_MultiSampler`.
    :param dict initial_params: dict containing initial tensors in unconstrained
        space to initiate the markov chain. The leading dimension's size must match
        that of `num_chains`. If not specified, parameter values will be sampled from
        the prior.
    :param hook_fn: Python callable that takes in `(kernel, samples, stage, i)`
        as arguments. stage is either `sample` or `warmup` and i refers to the
        i'th sample for the given stage. This can be
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above. Use `mp_context="spawn"` for
        CUDA.
    :param bool disable_progbar: Disable progress bar and diagnostics update.
    :param bool disable_validation: Disables distribution validation check. This is
        disabled by default, since divergent transitions will lead to exceptions.
        Switch to `True` for debugging purposes.
    :param dict transforms: dictionary that specifies a transform for a sample site
        with constrained support to unconstrained space.
    """
    def __init__(self, kernel, num_samples, warmup_steps=None, initial_params=None,
                 num_chains=1, hook_fn=None, mp_context=None, disable_progbar=False,
                 disable_validation=True, transforms=None):
        self.warmup_steps = num_samples if warmup_steps is None else warmup_steps  # Stan
        self.num_samples = num_samples
        self.kernel = kernel
        self.transforms = transforms
        self.disable_validation = disable_validation
        self._samples = None
        if isinstance(self.kernel, (HMC, NUTS)) and self.kernel.potential_fn is not None:
            if initial_params is None:
                raise ValueError("Must provide valid initial parameters to begin sampling"
                                 " when using `potential_fn` in HMC/NUTS kernel.")
        if num_chains > 1:
            # check that initial_params is different for each chain
            if initial_params:
                for v in initial_params.values():
                    if v.shape[0] != num_chains:
                        raise ValueError("The leading dimension of tensors in `initial_params` "
                                         "must match the number of chains.")
                if mp_context is None and six.PY3:
                    # change multiprocessing context to 'spawn' for CUDA tensors.
                    if list(initial_params.values())[0].is_cuda:
                        mp_context = "spawn"

            # verify num_chains is compatible with available CPU.
            available_cpu = max(mp.cpu_count() - 1, 1)  # reserving 1 for the main process.
            if num_chains > available_cpu:
                warnings.warn("num_chains={} is more than available_cpu={}. "
                              "Resetting number of chains to available CPU count."
                              .format(num_chains, available_cpu))
                num_chains = available_cpu
                # adjust initial_params accordingly
                if num_chains == 1:
                    initial_params = {k: v[0] for k, v in initial_params.items()}
                else:
                    initial_params = {k: v[:num_chains] for k, v in initial_params.items()}

        self.num_chains = num_chains
        self._diagnostics = [None] * num_chains

        if num_chains > 1:
            self.sampler = _MultiSampler(kernel, num_samples, self.warmup_steps, num_chains, mp_context,
                                         disable_progbar, initial_params=initial_params, hook=hook_fn)
        else:
            self.sampler = _UnarySampler(kernel, num_samples, self.warmup_steps, disable_progbar,
                                         initial_params=initial_params, hook=hook_fn)

    def run(self, *args, **kwargs):
        num_samples = [0] * self.num_chains
        z_acc = defaultdict(lambda: [[] for _ in range(self.num_chains)])
        with pyro.validation_enabled(not self.disable_validation):
            for x, chain_id in self.sampler.run(*args, **kwargs):
                if num_samples[chain_id] == self.num_samples:
                    self._diagnostics[chain_id] = x
                else:
                    num_samples[chain_id] += 1
                    for k, v in x.items():
                        z_acc[k][chain_id].append(v)

        z_acc = {k: [torch.stack(l) for l in v] for k, v in z_acc.items()}
        z_acc = {k: v[0] if self.num_chains == 1 else torch.stack(v) for k, v in z_acc.items()}

        # If transforms is not explicitly provided, infer automatically using
        # model args, kwargs.
        if self.transforms is None and isinstance(self.kernel, (HMC, NUTS)):
            if self.kernel.transforms is not None:
                self.transforms = self.kernel.transforms
            elif self.kernel.model:
                _, _, self.transforms, _ = initialize_model(self.kernel.model,
                                                            model_args=args,
                                                            model_kwargs=kwargs)
            else:
                self.transforms = {}

        # transform samples back to constrained space
        for name, transform in self.transforms.items():
            z_acc[name] = transform.inv(z_acc[name])
        self._samples = z_acc

        # terminate the sampler (shut down worker processes)
        self.sampler.terminate(True)

    def get_samples(self, num_samples=None, group_by_chain=False):
        """
        Get samples from the MCMC run, potentially resampling with replacement.

        :param int num_samples: Number of samples to return. If `None`, all the samples
            from an MCMC chain are returned in their original ordering.
        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            all samples will have num_chains as the size of their leading dimension.
        :return: dictionary of samples keyed by site name.
        """
        samples = self._samples
        if num_samples is None:
            # reshape to collapse chain dim when group_by_chain=False
            if not group_by_chain and self.num_chains > 1:
                samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
        else:
            if not samples:
                raise ValueError("No samples found from MCMC run.")
            if not group_by_chain and self.num_chains > 1:
                samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
                batch_dim = 0
            else:
                batch_dim = 1
            sample_tensor = list(samples.values())[0]
            batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
            idxs = torch.randint(0, batch_size, size=(num_samples,), device=device)
            samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}
        return samples

    def diagnostics(self):
        """
        Gets some diagnostics statistics such as effective sample size, split
        Gelman-Rubin, or divergent transitions from the sampler.
        """
        diag = diagnostics(self._samples, num_chains=self.num_chains)
        for diag_name in self._diagnostics[0]:
            diag[diag_name] = {'chain {}'.format(i): self._diagnostics[i][diag_name]
                               for i in range(self.num_chains)}
        return diag

    def summary(self, prob=0.9):
        """
        Prints a summary table displaying diagnostics of samples obtained from
        posterior. The diagnostics displayed are mean, standard deviation, median,
        the 90% Credibility Interval, :func:`~pyro.ops.stats.effective_sample_size`,
        :func:`~pyro.ops.stats.split_gelman_rubin`.

        :param float prob: the probability mass of samples within the credibility interval.
        """
        summary(self._samples, prob=prob, num_chains=self.num_chains)
