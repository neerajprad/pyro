import csv
import os
import sys
from subprocess import check_output

from tests.common import EXAMPLES_DIR

MODELS = [
    'hmm.py --model=1 --jit --profile',
    'hmm.py --model=1 --jit --cuda --profile',
]


def run_process(filename, args):
    out = check_output([sys.executable, filename] + args).decode('utf-8')
    train_times, sim_times = [], []
    for line in out.split('\n'):
        line = line.strip()
        if 'train time' in line:
            train_time = float(line.split('=')[1].strip())
            train_times.append(train_time)
        if 'sim time' in line:
            sim_time = float(line.split('=')[1].strip())
            sim_times.append(sim_time)
    assert len(train_times) == int(args[-1])
    assert len(sim_times) == int(args[-1])    
    return train_times, sim_times


if __name__ == "__main__":
    perf_results = {}
    for model in MODELS:
        example = model.split()
        filename, args = example[0], example[1:]
        filename = os.path.join(EXAMPLES_DIR, filename)
        out_file = "results_cuda" if "cuda" in model else "results_cpu"
        with open(out_file + '.csv', 'w', newline='') as f:
            fieldnames = ["hidden dim size", "time per step", "event"]
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for hidden_dim in range(4, 102, 4):
                print("Profiling hidden dim size = {}".format(hidden_dim))
                num_steps = 60
                run_args = args + ['--hidden-dim', str(hidden_dim), '--num-steps', str(num_steps)]
                train_times, sim_times = run_process(filename, run_args)
                for t in train_times:
                    writer.writerow([hidden_dim, t, "training"])
                for t in sim_times:
                    writer.writerow([hidden_dim, t, "simulation"])
