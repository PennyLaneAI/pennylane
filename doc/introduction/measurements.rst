.. role:: html(raw)
   :format: html

.. _intro_ref_meas:

Measurements
============

.. currentmodule:: pennylane.measure

A circuit in PennyLane2 can return different types of terminal measurement
results, including the expectation of an observable, its variance, samples of a
single measurement, or computational basis state probabilities.

The available measurement functions in PennyLane2 are

:html:`<div class="summary-table">`

.. autosummary::

    ~pennylane.expval
    ~pennylane.sample
    ~pennylane.counts
    ~pennylane.var
    ~pennylane.probs
    ~pennylane.state

:html:`</div>`

These different terminal measurements can be placed in the ``return`` statement
of a QNode:

.. code-block:: python

    import pennylane as qp

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def my_quantum_function(x, y):
        qp.RZ(x, wires=0)
        qp.CNOT(wires=[0, 1])
        qp.RY(y, wires=1)
        return qp.expval(qp.Z(1))

Below are example outputs from the circuit above when varying the terminal
measurement function.

.. raw:: html

    <style>
        table.measurements-table { border-collapse: collapse; }
        table.measurements-table th,
        table.measurements-table td { border: 1px solid #ccc !important; padding: 6px 10px; }
    </style>

.. list-table::
   :header-rows: 1
   :widths: 24 52 12 12
   :align: center
   :class: measurements-table

   * - Measurement
     - Example output
     - Analytic mode support
     - Finite-shots support
   * - :func:`qp.expval(qp.Z(1)) <pennylane.expval>`
     - ``0.9928``
     - ✓
     - ✓
   * - :func:`qp.var(qp.Z(1)) <pennylane.var>`
     - ``0.0143``
     - ✓
     - ✓
   * - :func:`qp.sample(qp.Z(1)) <pennylane.sample>`
     - ``[1., 1., 1., 1., 1.]``
     - ✗
     - ✓
   * - :func:`qp.counts(qp.Z(1)) <pennylane.counts>`
     - ``{-1.0: 4, 1.0: 996}``
     - ✗
     - ✓
   * - :func:`qp.probs(wires=(0, 1)) <pennylane.probs>`
     - ``[0.9964, 0.0036, 0., 0.]``
     - ✓
     - ✓
   * - :func:`qp.state() <pennylane.state>`
     - ``[0.962-0.2663j, 0.0578-0.016j, 0.+0.j, 0.+0.j]``
     - ✓
     - ✗

Analytic mode and finite-shots
------------------------------

If ``shots`` is not specified in a workflow anywhere, the workflow will run
analytically by default (analytic mode). A finite-shot simulation will be
performed if ``shots`` is set to a positive integer. The shot number can be
changed using the :func:`~.pennylane.set_shots` decorator, which can also be
directly applied to a QNode.

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=1)

    @qp.qjit(capture=True)
    @qp.set_shots(shots=10)
    @qp.qnode(dev)
    def circuit(x, y):
        qp.RX(x, wires=0)
        qp.RY(y, wires=0)
        return qp.expval(qp.PauliZ(0))

    # execute the QNode using 10 shots
    result = circuit(0.54, 0.1)

    # execute the QNode again, now using 1 shot
    result = qp.set_shots(circuit, shots=1)(0.54, 0.1)
