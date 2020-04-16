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

Comparing Revisions
-------------------

An additional script ``benchmark_revisions.py`` enables the comparative benchmarking of different
revisions of PennyLane. It calls upon ``benchmark.py`` and thus supports all its different arguments.
The revisions -- including branches, tags and commits -- are specified via ``-r revision1[,revision2[,revision3...]]``,
as in

.. code-block:: bash

  python3 benchmark_revisions.py -r master,0c8e90a -d default.qubit,default.tensor time bm_mutable_rotations

The chosen revisions will be downloaded and cached into the ``revisions`` subdirectory of the benchmarking folder.
They are not automatically removed, if you wish to free up space you have to remove them by hand. 

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


Note
----

Since the benchmarking tool is (for the time being) in the same Git repository as PennyLane,
checking out a specific PennyLane commit for benchmarking also may change the benchmarks.
Therefore it is recommended to make a copy of the latest version of the benchmarking tool in
another directory, and run that on the checked-out commit.


Included benchmarks
-------------------

* ``bm_entangling_layers``: Creates an immutable QNode consisting of strongly entangling layers,
  then evaluates it and its Jacobian. The size parameter ``n`` is the number of layers.
* ``bm_mutable_rotations``: Evaluates a mutable QNode consisting of ``k`` simple rotations on one qubit
  several times, varying ``k`` from 0 to ``n``. The qfunc is called each time due to the mutability.
* ``bm_iqp_circuit``:  Evaluates an IQP circuit on ``w`` wires with ``n*w`` gates chosen from the 
  set ``Z``, ``CZ`` and ``CCZ``.
