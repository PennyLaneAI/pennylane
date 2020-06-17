# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmarking tool for PennyLane.
"""
# pylint: disable=import-outside-toplevel,invalid-name
import argparse
import importlib
import subprocess
import time

import numpy as np

import pennylane as qml


# benchmarking tool version
__version__ = "0.1.0"


# ANSI escape sequences for terminal colors
Colors = {
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[95m",
}
RESET = "\033[0m"


def col(text, color):
    """Wraps the given text in color ANSI sequences.

    Args:
        text  (str): text to print
        color (str): ANSI color code

    Returns:
        str: text wrapped with ANSI codes
    """
    return Colors[color] + text + RESET


def timing(func, *, number=10, repeat=5):
    """Time the given function.

    Args:
        func (Callable[[], Any]): function to time
        number (int): number of loops per test
        repeat (int): the number of times the timing test is run
    """
    import timeit

    print("{} loops, {} runs".format(number, repeat))
    res = timeit.repeat(func, number=number, repeat=repeat, globals=globals())
    print("Timing per loop:", col(str(np.array(res) / number), "yellow"))


def plot(title, kernels, labels, n_vals):
    """Plot timings as a function of the parameter n.

    Args:
        title (str): plot title
        kernels (Sequence[Callable[[Any], Any]]): parametrized benchmarks to time
        labels (Sequence[str]): names of the kernels
        n_vals (Sequence[Any]): values the benchmark parameter n takes
    """
    import perfplot

    perfplot.show(
        setup=lambda n: n, kernels=kernels, labels=labels, n_range=n_vals, xlabel="n", title=title,
    )


def profile(func, identifier, *, min_time=5):
    """Profile the given function.

    Args:
        func (Callable[[], Any]): function to profile
        identifier (str): identifying part of the name of the file containing the results
        min_time (float): func is called repeatedly until at least this many seconds have elapsed
    """
    import cProfile
    import pstats

    print("Minimum duration: {} seconds.".format(min_time))
    pr = cProfile.Profile()
    pr.enable()

    t0 = time.process_time()
    repeats = 0
    while time.process_time() - t0 < min_time:
        func()
        repeats += 1

    pr.disable()
    pr.dump_stats("pennylane_{}.pstats".format(identifier))
    ps = pstats.Stats(pr).strip_dirs().sort_stats("tottime")  # "cumulative")
    ps.print_stats()
    print("{} repeats.".format(repeats))


def cli():
    """Parse the command line arguments, perform the requested action.
    """
    parser = argparse.ArgumentParser(description="PennyLane benchmarking tool")
    parser.add_argument("--noinfo", action="store_true", help="suppress information output")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument(
        "-d",
        "--device",
        type=lambda x: x.split(","),
        default="default.qubit",
        help="comma-separated list of devices to run the benchmark on (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--qnode",
        type=lambda x: x.split(","),
        default="QNode",
        help="comma-separated list of QNode subclasses to run the benchmark on (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--wires",
        type=int,
        default=3,
        help="number of wires to run the benchmark on (default: %(default)s)",
    )
    parser.add_argument("cmd", choices=["time", "plot", "profile"], help="function to perform")
    parser.add_argument("benchmark", help="benchmark module name (without .py)")

    args = parser.parse_args()

    # look up information about the current HEAD Git commit
    res = subprocess.run(
        ["git", "log", "-1", "--pretty=%h %s"],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    title = res.stdout
    short_hash = title.split(" ", maxsplit=1)[0]

    print("Benchmarking PennyLane", qml.version())

    if args.verbose:
        print("Verbose mode on, results may not be representative.")

    if not args.noinfo:
        print("Commit:", col(title, "red"))
        qml.about()
        print()

    # import the requested benchmark module
    mod = importlib.import_module(args.benchmark)

    # check for additional device args
    devs = args.device
    dev_kwargs = []
    for idx, dev in enumerate(devs):
        dev_kwargs.append({})
        if "default.tensor" in dev:
            # check for additional args
            device_full = dev.split("-")
            devs[idx] = device_full[0]
            print(device_full)
            if len(device_full) > 1:
                dev_kwargs[idx]["representation"] = device_full[1]

    # execute the command
    if args.cmd == "plot":
        print(
            "Performance plot: '{}' benchmark on {}, {}".format(
                mod.Benchmark.name, args.device, args.qnode
            )
        )
        bms = [
            mod.Benchmark(qml.device(d, wires=args.wires, **k), qnode_type=q, verbose=args.verbose)
            for d, k in zip(devs, dev_kwargs)
            for q in args.qnode
        ]
        for k in bms:
            k.setup()
        plot(
            title,
            [k.benchmark for k in bms],
            [f"{args.benchmark} {k.device.short_name} {k.qnode_type}" for k in bms],
            mod.Benchmark.n_vals,
        )
        for k in bms:
            k.teardown()
        return

    for dev, k in zip(devs, dev_kwargs):
        dev = qml.device(dev, wires=args.wires, **k)
        for q in args.qnode:
            bm = mod.Benchmark(dev, qnode_type=q, verbose=args.verbose)
            bm.setup()
            text = col(f"'{bm.name}'", "blue") + " benchmark on " + col(f"{d}, {q}", "magenta")
            if args.cmd == "time":
                print("Timing:", text)
                timing(bm.benchmark)
            elif args.cmd == "profile":
                print("Profiling:", text)
                profile(bm.benchmark, identifier="_".join([short_hash, d, q]))
            else:
                raise ValueError("Unknown command.")
            bm.teardown()


if __name__ == "__main__":
    cli()
