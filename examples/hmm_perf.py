import argparse
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


def profile_hidden_dim():
    for model in MODELS:
        example = model.split()
        filename, args = example[0], example[1:]
        filename = os.path.join(EXAMPLES_DIR, filename)
        out_file = "hidden_dim_cuda" if "cuda" in model else "hidden_dim_cpu"
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


def profile_batch_size():
    for model in MODELS:
        example = model.split()
        filename, args = example[0], example[1:]
        filename = os.path.join(EXAMPLES_DIR, filename)
        out_file = "batch_size_cuda" if "cuda" in model else "batch_size_cpu"
        with open(out_file + '.csv', 'w', newline='') as f:
            fieldnames = ["batch size", "time per step", "event"]
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for batch_size in range(4, 100, 4):
                print("Profiling batch size = {}".format(batch_size))
                num_steps = 60
                run_args = args + ['--batch-size', str(batch_size), '--num-steps', str(num_steps)]
                train_times, sim_times = run_process(filename, run_args)
                for t in train_times:
                    writer.writerow([batch_size, t, "training"])
                for t in sim_times:
                    writer.writerow([batch_size, t, "simulation"])


def profile_plate_dim():
    for model in MODELS:
        example = model.split()
        filename, args = example[0], example[1:]
        filename = os.path.join(EXAMPLES_DIR, filename)
        out_file = "plate_dim_cuda" if "cuda" in model else "plate_dim_cpu"
        with open(out_file + '.csv', 'w', newline='') as f:
            fieldnames = ["plate dim size", "time per step", "event"]
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for plate_dim in range(4, 88, 44):
                print("Profiling plate dim size = {}".format(plate_dim))
                num_steps = 60
                run_args = args + ['--num-steps', str(num_steps), '--clamp-notes', str(plate_dim)]
                train_times, sim_times = run_process(filename, run_args)
                for t in train_times:
                    writer.writerow([plate_dim, t, "training"])
                for t in sim_times:
                    writer.writerow([plate_dim, t, "simulation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM profiler")
    parser.add_argument("--profile", type=str)
    args = parser.parse_args()
    if args.profile == "hidden_dim":
        profile_hidden_dim()
    elif args.profile == "batch_size":
        profile_batch_size()
    elif args.profile == "plate_dim":
        profile_plate_dim()

