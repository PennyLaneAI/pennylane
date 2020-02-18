# Benchmarking utility for PennyLane
import functools
import importlib
import subprocess
import sys

import pennylane as qml


def timing(func):
    "Time the given function."
    import timeit

    res = timeit.repeat(func, number=10, repeat=5, globals=globals())
    print("Timing:", res)


def plot(title, kernels, n_range):
    "Plot timings as a function of the parameter n."
    import perfplot

    labels = [k.__doc__ for k in kernels]
    perfplot.show(
        setup=lambda n: n,
        kernels=kernels,
        labels=labels,
        n_range=n_range,
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


def profile(func):
    "Profile the given function."
    import cProfile, pstats

    pr = cProfile.Profile()
    pr.enable()

    for k in range(20):
        func()

    pr.disable()
    pr.dump_stats("pennylane_{}.pstats".format(qml.version()))
    ps = pstats.Stats(pr).strip_dirs().sort_stats("tottime")  # "cumulative")
    ps.print_stats()


if __name__ == "__main__":
    # import pdb; pdb.set_trace()

    n = len(sys.argv)
    if n != 3:
        print("Usage: benchmark.py {time,plot,profile} benchmark_module")
        sys.exit(0)

    res = subprocess.run(["git", "log", "-1", "--pretty=%h %s"], stdout=subprocess.PIPE)
    title = res.stdout.decode("utf-8")
    print("Benchmarking PennyLane", qml.version())
    print("Commit:", title)
    # qml.about()

    # import the requested benchmark module
    bm = importlib.import_module(sys.argv[2])

    cmd = sys.argv[1]
    if cmd == "time":
        print("Timing ", bm.benchmark.__doc__)
        timing(functools.partial(bm.benchmark, bm.n))
    elif cmd == "plot":
        plot(title, [bm.benchmark], range(bm.n_min, bm.n_max, bm.n_step))
    elif cmd == "profile":
        profile(functools.partial(bm.benchmark, bm.n))
    else:
        raise ValueError("Unknown command.")
