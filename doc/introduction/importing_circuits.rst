.. role:: html(raw)
   :format: html

.. _intro_ref_importing_circuits:

Importing circuits
==================

PennyLane offers the :mod:`~.io` module to import quantum circuits and operations that were
constructed outside of PennyLane. This includes circuits defined using `Qiskit <https://www.ibm.com/quantum/qiskit>`__,
`OpenQASM <https://docs.quantum.ibm.com/build/interoperate-qiskit-qasm2>`_, and `Quil
<https://docs.rigetti.com/qcs/guides/quil>`_.

.. note::

    To import a quantum circuit defined using a particular framework, you will need to install the
    corresponding PennyLane plugin for that framework. More information about PennyLane plugins is
    available on the `plugins <https://pennylane.ai/plugins.html>`_ page.

Importing quantum circuits
--------------------------

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

Now, suppose we define a Qiskit `QuantumCircuit
<https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`__ as follows:

.. code-block:: python

    from qiskit import QuantumCircuit

    qk_circuit = QuantumCircuit(2, 1)
    qk_circuit.h(0)
    qk_circuit.cx(0, 1)
    qk_circuit.measure_all()

We can convert the ``QuantumCircuit`` into a PennyLane quantum function using:

.. code-block:: python

    import pennylane as qml

    pl_template_from_qk = qml.from_qiskit(qk_circuit)

Above, the :func:`~pennylane.from_qiskit` function converts a ``QuantumCircuit`` into a PennyLane
template. This template can then be called from inside a QNode to generate a PennyLane circuit:

.. code-block:: python

    @qml.qnode(qml.device("default.qubit"))
    def pl_circuit_from_qk():
        pl_template_from_qk(wires=[0, 1])
        return qml.expval(qml.Y(0)), qml.expval(qml.Z(1))

.. note::

    Alternatively, the QNode can be instantiated directly from the Qiskit circuit:

    .. code-block:: python

        measurements = [qml.expval(qml.Y(0)), qml.var(qml.Z(1))]
        pl_template_from_qk = qml.from_qiskit(qk_circuit, measurements=measurements)
        pl_circuit_from_qk = qml.QNode(pl_template_from_qk, qml.device("default.qubit"))


    Here, the ``measurements`` argument overrides the terminal measurements in the Qiskit circuit.
    See the :func:`~pennylane.from_qiskit` documentation for more details.

The resulting PennyLane circuit can be executed directly:

>>> pl_circuit_from_qk()
[tensor(0., requires_grad=True), tensor(1., requires_grad=True)]

It can also be visualized using PennyLane's :func:`~pennylane.draw` utility:

>>> print(qml.draw(pl_circuit_from_qk)())
0: ──H─╭●─╭||─┤  <Y>
1: ────╰X─╰||─┤  Var[Z]

OpenQASM
~~~~~~~~

An equivalent quantum circuit can be expressed in OpenQASM 2.0 as follows:

.. code-block:: python

    oq_circuit = (
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        h q[0];
        cx q[0], q[1];

        measure q -> c;
        """
    )

We can import this circuit into PennyLane using the PennyLane-Qiskit plugin once more:

.. code-block:: python

    import pennylane as qml

    pl_template_from_oq = qml.from_qasm(oq_circuit)

    @qml.qnode(qml.device("default.qubit"))
    def pl_circuit_from_oq():
        pl_template_from_oq(wires=[0, 1])
        return qml.expval(qml.Y(0)), qml.var(qml.Z(1))

The result is as follows:

>>> print(qml.draw(pl_circuit_from_oq)())
0: ──H─╭●──┤↗├─┤  <Y>
1: ────╰X──┤↗├─┤  Var[Z]

Quil
~~~~

PennyLane also offers convenience functions for importing circuits from `pyQuil
<https://pyquil-docs.rigetti.com/en/stable/index.html>`__ or Quil representations. Both of these
require the `PennyLane-Rigetti <https://docs.pennylane.ai/projects/rigetti/en/stable/>`__ plugin,
which can be installed using:

.. code-block::

    pip install pennylane-rigetti

We begin with a familiar pyQuil `Program
<https://pyquil-docs.rigetti.com/en/stable/apidocs/pyquil.quil.html#pyquil.quil.Program>`__:

.. code-block:: python

    import pyquil

    pq_program = pyquil.Program()
    pq_program += pyquil.gates.H(0)
    pq_program += pyquil.gates.CNOT(0, 1)

This ``Program`` can be converted into a PennyLane quantum function using the
:func:`~pennylane.from_pyquil` function:

.. code-block:: python

    import pennylane as qml

    pl_template_from_pq = qml.from_pyquil(pq_program)

    @qml.qnode(qml.device("default.qubit"))
    def pl_circuit_from_pq():
        pl_template_from_pq(wires=[0, 1])
        return qml.expval(qml.Y(0)), qml.var(qml.Z(1))

The resulting PennyLane circuit is:

>>> print(qml.draw(pl_circuit_from_pq)())
0: ──H─╭●─┤  <Y>
1: ────╰X─┤  Var[Z]

.. note::

    Quantum circuits expressed in Quil can be imported in a similar way using
    :func:`~pennylane.from_quil`.


Importing quantum operators
---------------------------

Sometimes, it is preferable to import a single operation from a framework instead of an entire
quantum circuit. This can save you some keystrokes and serve as a helpful crutch for understanding
an individual component of a circuit.

Presently, only Qiskit `SparsePauliOp
<https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp>`__ operators can be
imported into PennyLane. To see this in action, we first define a ``SparsePauliOp``:

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp

    qk_op = SparsePauliOp(["II", "XY"])

Then, we apply the :func:`~pennylane.from_qiskit_op` function to convert the ``SparsePauliOp`` into
a PennyLane :class:`Operator <pennylane.operation.Operator>`:

.. code-block:: python

    import pennylane as qml

    pl_op = qml.from_qiskit_op(qk_op)

We can inspect both operators to make sure they match:

>>> qk_op
SparsePauliOp(['II', 'XY'],
              coeffs=[1.+0.j, 1.+0.j])
>>> pl_op
I(0) + X(1) @ Y(0)


Parameterized operators
~~~~~~~~~~~~~~~~~~~~~~~

PennyLane also supports importing parameterized ``SparsePauliOp`` instances. Consider:

.. code-block:: python

    import numpy as np
    from qiskit.circuit import Parameter

    a, b, c = [Parameter(var) for var in "abc"]
    param_qk_op = SparsePauliOp(["II", "XZ", "YX"], coeffs=np.array([a, b, c]))

To import this ``SparsePauliOp``, we must specify a concrete value for each coefficient using the
``params`` argument:

.. code-block:: python

    import pennylane as qml

    param_pl_op = qml.from_qiskit_op(param_qk_op, params={a: 2, b: 3, c: 4})

The result is:

>>> param_qk_op
SparsePauliOp(['II', 'XZ', 'YX'],
              coeffs=[ParameterExpression(1.0*a), ParameterExpression(1.0*b),
 ParameterExpression(1.0*c)])
>>> param_pl_op
(
    (2+0j) * I(0)
  + (3+0j) * (X(1) @ Z(0))
  + (4+0j) * (Y(1) @ X(0))
)


Import functions
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
