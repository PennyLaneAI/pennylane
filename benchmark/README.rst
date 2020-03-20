Benchmarking tool for PennyLane
===============================

``benchmark.py`` is a simple benchmarking script that can do timings, performance plots and
basic profiling. The benchmarks can be run on different PL devices, and the results compared.

The script provides a uniform way of evaluating PL code performance, and the way it changes
between commits. It is meant to be run on a Git repository containing the PL code, and will
automatically extract the short hash of the currently checked out commit to label the results.

The script has three main functions.


* Profiling::

      python3 benchmark.py profile bm_mutable_rotations
      bash view_pstats.sh pennylane_*.pstats

  The profiling stats are saved in a .pstats file, which can be viewed using various tools.
  For example, the ``view_pstats.sh`` BASH script uses ``gprof2dot`` to present the data as a
  call graph.

* Timing::

      python3 benchmark.py -d default.qubit,default.tensor time bm_mutable_rotations

* Producing performance plots::

      python3 benchmark.py -d default.qubit,default.tensor plot bm_mutable_rotations

  The performance plot shows the execution time of the benchmark as a function of a scalar
  "size parameter" ``n``, the exact meaning of which depends on each benchmark.


Installation
------------

The benchmarking tool has the same dependencies as PennyLane.
Additionally, performance plots require the ``perfplot`` Python package,
and the profiling stats visualization script requires the ``gprof2dot``
Python package and the ``Graphviz`` binary package.


Usage
-----

Use ``python3 benchmark.py -h`` to see a summary of the available commandline options.
The most important ones are

* ``-d DEVICES``: Comma-separated list of PL device names (without spaces), on which
  to execute the benchmark. The default is 'default.qubit'.

* ``-w WIRES``: Number of wires to run the benchmark on. The default is 3.


Included benchmarks
-------------------

* ``bm_entangling_layers``: Creates an immutable QNode consisting of strongly entangling layers,
  then evaluates it and its Jacobian. The size parameter ``n`` is the number of layers.
* ``bm_mutable_rotations``: Evaluates a mutable QNode consisting of ``k`` simple rotations on one qubit
  several times, varying ``k`` from 0 to ``n``. The qfunc is called each time due to the mutability.
