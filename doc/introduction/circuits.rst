 .. role:: html(raw)
   :format: html


.. _intro_vcircuits:

Quantum circuits
================

In PennyLane2, quantum circuits are represented as *quantum nodes*, or QNodes
for short. A QNode not only declares a quantum circuit, it also ties it to a
specific device that executes it.

Furthermore, in PennyLane2, all workflows involving QNodes must be decorated
with :func:`@qjit <~pennylane.qjit>`, which triggers our bespoke quantum
just-in-time compiler framework.

Workflows compiled with :func:`@qjit <~pennylane.qjit>` can simultaneously
interface with JAX and NumPy data types, though there are cases where using one
over the other could be preferred. More information on compiling circuits with
:func:`@qjit <~.qjit>` can be found in :doc:`compiling_circuits`.

.. _intro_vcirc_qfunc:

How circuits are constructed
----------------------------

1. Quantum functions
^^^^^^^^^^^^^^^^^^^^

The first step in constructing an executable quantum circuit is by defining a
special Python function called a *quantum function*. For example:

.. code-block:: python

    import pennylane as qp

    def circuit(x, y):
        qp.RZ(x, wires=0)
        qp.CNOT(wires=[0, 1])
        qp.RY(y, wires=1)
        return qp.expval(qp.Z(wires=1))

.. note::

    PennyLane uses the term *wires* instead of qubits.

Quantum circuit operators (e.g., :doc:`gates <operations>` or
:doc:`subroutines <subroutines>`) are placed one-per-line, where the operators
occur in the final circuit in the order they appear in the quantum function. The
final line of a Python function is typically the ``return`` statement, which in
quantum-circuit terms is the :doc:`terminal measurement <measurements>`.
Although, standard Python functions don't have to return anything, and neither
do quantum functions.

Additionally, quantum functions can accept classical input arguments and contain
control-flow like ``for`` loops, ``while`` loops, and ``if`` statements.

2. Devices
^^^^^^^^^^

A quantum circuit has to run on a device, be it real hardware or a simulator.
All devices that are compatible with PennyLane2 can be found at
`<https://pennylane.ai/devices>`_. In general, however, devices are instantiated
using :func:`~pennylane.device` as follows,

.. code-block:: python

    dev = qp.device("short.name", wires=10)

where the ``wires`` argument is the maximum number of simultaneously active
wires in the circuit and ``short.name`` is an identifier (or short name) for the
desired device. For example, the Lightning Qubit C++ simulator's short name is
``lightning.qubit``.

3. Decorate with qp.qnode, compile with qp.qjit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Together, a quantum function and a device are used to create a *quantum node*,
or QNode object, which wraps the quantum function and binds it to the device.
QNodes are the foundational computational unit of PennyLane. Simply decorate
with :func:`@qp.qnode(dev) <~pennylane.qnode>` to create a QNode.

.. note::

    A QNode *must* return a terminal measurement.

To be able to simulate or get quantum resource estimates, decorate with
:func:`@qp.qjit <~pennylane.qjit>`, which is PennyLane's bespoke quantum
just-in-time compiler.

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x, y):
        qp.RZ(x, wires=0)
        qp.CNOT(wires=[0, 1])
        qp.RY(y, wires=1)
        return qp.expval(qp.Z(wires=1))

Now, the circuit can be executed by calling it like any other standard Python
function,

>>> circuit(0.1, 0.2)
Array(0.98006658, dtype=float64)

or inspected with, say, :func:`~pennylane.analyze` for fast resource analysis.

>>> qp.analyze(circuit)(0.1, 0.2).resources
SpecsResources(counts={'CNOT': 1, 'RY': 1, 'RZ': 1}, measurement_processes={'expval(PauliZ)': 1}, num_wires=2, circuit_depth=None, total_quantum_operations=3)
