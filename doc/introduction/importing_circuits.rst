.. role:: html(raw)
   :format: html

.. _intro_ref_importing_circuits:

Importing Circuits
==================

PennyLane offers the :mod:`~.io` module to import quantum circuits and operations that were
constructed outside of PennyLane. This includes circuits defined using `Qiskit <https://www.ibm.com/quantum/qiskit>`__,
`OpenQASM <https://openqasm.com/>`_, and `Quil <https://docs.rigetti.com/qcs/guides/quil>`_.

.. note::

    To import a quantum circuit defined using a particular framework, you will need to install the
    corresponding PennyLane plugin for that framework. More information about PennyLane plugins is
    available on the `plugins <https://pennylane.ai/plugins.html>`_ page.

Importing Quantum Circuits in PennyLane
---------------------------------------

Importing quantum circuits can help you take advantage of PennyLane's various optimization,
visualization, and interoperability features for existing circuits. For example, you can take a
``QuantumCircuit`` from Qiskit, import it into PennyLane, and then apply a `circuit-cutting transform
<https://pennylane.ai/qml/demos/tutorial_quantum_circuit_cutting/>`_ to reduce the number of qubits
required to implement the circuit. You could also :ref:`compile <intro_ref_compile_circuits>` the
circuit using `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ or
execute the circuit remotely using a supported hardware provider.

Qiskit
~~~~~~

To import a quantum circuit from Qiskit, you must first install the `PennyLane-Qiskit
<https://docs.pennylane.ai/projects/qiskit/en/stable/>`__ plugin. This can be done by running:

.. code-block::

    pip install pennylane-qiskit

Now, suppose we define a Qiskit ``QuantumCircuit`` as follows:

.. code-block:: python

    from qiskit import QuantumCircuit

    qk_circuit = QuantumCircuit(2, 1)
    qk_circuit.h(0)
    qk_circuit.cx(0, 1)
    qk_circuit.measure_all()

We can convert the ``QuantumCircuit`` into a PennyLane quantum function using:

.. code-block:: python

    import pennylane as qml

    pl_template = qml.from_qiskit(qk_circuit)

The :func:`from_qiskit` function returns a PennyLane template which can be used inside a QNode, as
in:

.. code-block:: python

    @qml.qnode(qml.device("default.qubit"))
    def pl_circuit():
        pl_template(wires=[0, 1])
        return qml.expval(qml.Z(0) @ qml.Z(1))

The resulting PennyLane circuit can be executed directly:

>>> pl_circuit()
tensor(1., requires_grad=True)

The usual PennyLane circuit functions are also available, including the drawing tool:

>>> print(qml.draw(pl_circuit)())
0: ──H─╭●─╭||──┤↗├─┤ ╭<Z@Z>
1: ────╰X─╰||──┤↗├─┤ ╰<Z@Z>


OpenQASM
~~~~~~~~

TODO

Quil
~~~~

TODO


Importing Quantum Operations in PennyLane
-----------------------------------------



Import Functions
----------------

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.from_pyquil
    ~pennylane.from_qasm
    ~pennylane.from_qasm_file
    ~pennylane.from_qiskit
    ~pennylane.from_qiskit_op
    ~pennylane.from_quil
    ~pennylane.from_quil_file

:html:`</div>`
