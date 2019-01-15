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
    train_time, sim_time = None, None
    for line in out.split('\n'):
        line = line.strip()
        if 'training time' in line:
            train_time = float(line.split('=')[1].strip())
        if 'simulation time' in line:
            sim_time = float(line.split('=')[1].strip())
    return train_time, sim_time


if __name__ == "__main__":
    perf_results = {}
    for model in MODELS:
        example = model.split()
        filename, args = example[0], example[1:]
        filename = os.path.join(EXAMPLES_DIR, filename)
        x, y, z = [], [], []
        out_file = "results_cuda" if "cuda" in model else "results_cpu"
        with open(out_file + '.csv', 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            for hidden_dim in range(4, 102, 4):
                print("Profiling hidden dim size = {}".format(hidden_dim))
                if hidden_dim <= 50:
                    num_steps = 100
                elif hidden_dim < 80:
                    num_steps = 50
                else:
                    num_steps = 20
                run_args = args + ['--hidden-dim', str(hidden_dim), '--num-steps', str(num_steps)]
                train_time_per_step, sim_time_per_step = run_process(filename, run_args)
                x.append(hidden_dim)
                y.append(train_time_per_step)
                z.append(sim_time_per_step)
                writer.writerow([hidden_dim, train_time_per_step, sim_time_per_step])
        perf_results[model] = (x, y, z)
    print(perf_results)
