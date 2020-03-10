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
import functools
import importlib
import subprocess
import sys

import numpy as np

import pennylane as qml


# benchmarking tool version
__version__ = "0.1.0"


def timing(func, *, number=10, repeat=5):
    """Time the given function.

    Args:
        func (Callable[[], Any]): function to time
        number (int): number of loops per test
        repeat (int): the number of times the timing test is run
    """
    import timeit

    print('{} loops, {} runs'.format(number, repeat))
    res = timeit.repeat(func, number=number, repeat=repeat, globals=globals())
    print("Timing per loop:", np.array(res) / number)


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
        setup=lambda n: n,
        kernels=kernels,
        labels=labels,
        n_range=n_vals,
        xlabel="n",
        title=title,
        # logx="auto",  # set to True or False to force scaling
        # logy="auto",
        # equality_check=numpy.allclose,  # set to None to disable "correctness" assertion
        # automatic_order=True,
        # colors=None,
        # target_time_per_measurement=1.0,
        # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
        # relative_to=1,  # plot the timings relative to one of the measurements
        # flops=lambda n: 3*n,  # FLOPS plots
    )


def profile(func, identifier, *, repeat=20):
    """Profile the given function.

    Args:
        func (Callable[[], Any]): function to profile
        identifier (str): identifying part of the name of the file containing the results
        repeat (int): the number of times the function is run
    """
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(repeat):
        func()

    pr.disable()
    pr.dump_stats("pennylane_{}.pstats".format(identifier))
    ps = pstats.Stats(pr).strip_dirs().sort_stats("tottime")  # "cumulative")
    ps.print_stats()


def cli():
    """Parse the command line arguments, perform the requested action.

    Returns:
      argparse.Namespace: parsed arguments in a namespace container
    """
    parser = argparse.ArgumentParser(description='PennyLane benchmarking tool')
    parser.add_argument('--version', action='version', version=__version__)
    #parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-d', '--device', type=lambda x: x.split(','), default=['default.qubit'], help='comma-separated list of devices to run the benchmark on (default: %(default)s)')
    parser.add_argument('-w', '--wires', type=int, default=3, help='number of wires to run the benchmark on (default: %(default)s)')
    parser.add_argument('cmd', choices=['time', 'plot', 'profile'], help='function to perform')
    parser.add_argument('benchmark', help='benchmark module')

    args = parser.parse_args()

    # look up information about the current HEAD Git commit
    res = subprocess.run(
        ["git", "log", "-1", "--pretty=%h %s"], stdout=subprocess.PIPE, encoding="utf-8", check=True
    )
    title = res.stdout
    short_hash = title.split(" ", maxsplit=1)[0]

    print("Benchmarking PennyLane", qml.version())
    print("Commit:", title)
    #qml.about()

    # import the requested benchmark module
    mod = importlib.import_module(args.benchmark)

    # execute the command
    if args.cmd == "time":
        for d in args.device:
            dev = qml.device(d, wires=args.wires)
            bm = mod.Benchmark(dev)
            bm.setup()
            print("Timing: '{}' benchmark on {}".format(bm.name, d))
            timing(bm.benchmark)
    elif args.cmd == "plot":
        print("Performance plot: '{}' benchmark on {}".format(mod.Benchmark.name, args.device))
        bms = [mod.Benchmark(qml.device(d, wires=args.wires)) for d in args.device]
        for k in bms:
            k.setup()
        plot(title, [k.benchmark for k in bms], [args.benchmark + ' ' + k.device.short_name for k in bms], mod.Benchmark.n_vals)
    elif args.cmd == "profile":
        for d in args.device:
            dev = qml.device(d, wires=args.wires)
            bm = mod.Benchmark(dev)
            bm.setup()
            print("Profiling: '{}' benchmark on {}".format(bm.name, d))
            profile(bm.benchmark, identifier=short_hash + '_' + d)
    else:
        raise ValueError("Unknown command.")



if __name__ == "__main__":
    cli()
