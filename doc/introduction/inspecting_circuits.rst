.. role:: html(raw)
   :format: html

.. _intro_inspecting_circuits:

Inspecting circuits
===================

A :func:`~pennylane.qjit`-compiled program does not run in Python, so Python's ``print`` statements
and debuggers only run once, while the program is compiled. Instead, PennyLane provides tools that
run as part of the compiled program, as well as access to the compiled program itself.

Printing values at runtime
--------------------------

The :func:`catalyst.debug.print` function prints values while the compiled program runs, including
the outcomes of mid-circuit measurements:

.. code-block:: python

    from catalyst import debug

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x):
        debug.print("x = {x}", x=x)
        qp.RX(x, wires=0)
        m = qp.measure(0)
        debug.print("measurement outcome: {m}", m=m)
        qp.cond(m, qp.X)(wires=1)
        return qp.expval(qp.Z(1))

>>> circuit(np.pi)
x = 3.141592653589793
measurement outcome: 1
Array(-1., dtype=float64)

Running Python callbacks
------------------------

The :func:`catalyst.debug.callback` decorator turns a Python function into one that is called with
runtime values while the compiled program runs. This can be used for logging or checking
intermediate values:

.. code-block:: python

    @debug.callback
    def check(x):
        print(f"Received x = {x:.2f}")

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x):
        check(x)
        qp.RX(x, wires=0)
        return qp.expval(qp.Z(0))

>>> circuit(0.5)
Received x = 0.50
Array(0.87758256, dtype=float64)

Inspecting the compiled program
-------------------------------

The intermediate representation of a compiled program is available as text. The ``mlir``
property contains the program before any compilation pass, and the ``mlir_opt`` property contains
the program after all compilation passes:

.. code-block:: python

    print(circuit.mlir)
    print(circuit.mlir_opt)

Additional resources
--------------------

For more details on debugging compiled programs, see the
:doc:`Catalyst sharp bits and debugging tips <catalyst:dev/sharp_bits>`. For more details on
compilation, see :doc:`compiling workflows <compiling_workflows>`.
