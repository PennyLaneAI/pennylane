.. role:: html(raw)
   :format: html

.. _intro_ref_importing_circuits:

Importing workflows
===================

PennyLane supports importing quantum circuits and operations that were
constructed using another framework. This includes circuits defined using `Qiskit <https://www.ibm.com/quantum/qiskit>`__,
`OpenQASM <https://arxiv.org/abs/1707.03429>`_, and `Quil
<https://docs.rigetti.com/qcs/guides/quil>`_.

.. note::

    To import a quantum circuit defined using a particular framework, you will need to install the
    corresponding PennyLane plugin for that framework. More information about PennyLane plugins is
    available on the `plugins <https://pennylane.ai/plugins>`_ page.

Importing quantum circuits
--------------------------

Importing quantum circuits can help you take advantage of PennyLane's various optimization,
visualization, and interoperability features for existing circuits. For example, you can take a
`QuantumCircuit <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`__ from
Qiskit, import it into PennyLane, and then apply a `circuit-cutting transform
<https://pennylane.ai/qml/demos/tutorial_quantum_circuit_cutting/>`_ to reduce the number of qubits
required to implement the circuit. You could also :ref:`compile <intro_ref_compile_worklfows>` the
circuit using `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ or
differentiate and optimize the circuit using :ref:`quantum-specific optimizers <intro_ref_opt>`.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.from_pyquil
    ~pennylane.from_qasm
    ~pennylane.from_qiskit
    ~pennylane.from_quil
    ~pennylane.from_quil_file

:html:`</div>`

Qiskit
~~~~~~

Suppose we define a Qiskit ``QuantumCircuit`` as follows:

.. code-block:: python

    from qiskit import QuantumCircuit

    qk_circuit = QuantumCircuit(2, 1)
    qk_circuit.h(0)
    qk_circuit.cx(0, 1)

We can convert the ``QuantumCircuit`` into a PennyLane :ref:`quantum function <intro_vcirc_qfunc>`
using:

.. code-block:: python

    import pennylane as qp

    pl_qfunc_from_qk = qp.from_qiskit(qk_circuit)

.. note::

    The `PennyLane-Qiskit <https://docs.pennylane.ai/projects/qiskit/en/latest/>`__ plugin must be
    installed to use the :func:`~pennylane.from_qiskit` function.

This quantum function can then be called from inside a QNode to generate a PennyLane circuit:

.. code-block:: python

    @qp.qnode(qp.device("default.qubit"))
    def pl_circuit_from_qk():
        pl_qfunc_from_qk(wires=[0, 1])
        return qp.expval(qp.Y(0)), qp.var(qp.Z(1))

.. note::

    Alternatively, the QNode can be instantiated directly from the Qiskit circuit:

    .. code-block:: python

        measurements = [qp.expval(qp.Y(0)), qp.var(qp.Z(1))]
        pl_qfunc_from_qk = qp.from_qiskit(qk_circuit, measurements=measurements)
        pl_circuit_from_qk = qp.QNode(pl_qfunc_from_qk, qp.device("default.qubit"))


    Here, the ``measurements`` argument overrides any terminal measurements in the Qiskit circuit.
    See the :func:`~pennylane.from_qiskit` documentation for more details.

The resulting PennyLane circuit can be executed directly:

>>> pl_circuit_from_qk()
[tensor(0., requires_grad=True), tensor(1., requires_grad=True)]

It can also be visualized using PennyLane's :func:`~pennylane.draw` utility:

>>> print(qp.draw(pl_circuit_from_qk)())
0: ──H─╭●─┤  <Y>
1: ────╰X─┤  Var[Z]

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
        """
    )

We can import this circuit into PennyLane using the PennyLane-Qiskit plugin once more:

.. code-block:: python

    import pennylane as qp

    pl_qfunc_from_oq = qp.from_qasm(oq_circuit)

    @qp.qnode(qp.device("default.qubit"))
    def pl_circuit_from_oq():
        pl_qfunc_from_oq(wires=[0, 1])
        return qp.expval(qp.Y(0)), qp.var(qp.Z(1))

The result is as follows:

>>> print(qp.draw(pl_circuit_from_oq)())
0: ──H─╭●─┤  <Y>
1: ────╰X─┤  Var[Z]

Quil
~~~~

PennyLane also offers convenience functions for importing circuits from `pyQuil
<https://pyquil-docs.rigetti.com/en/stable/index.html>`__ or Quil representations. Both of these
require the `PennyLane-Rigetti <https://docs.pennylane.ai/projects/rigetti/en/stable/>`__ plugin.

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

    import pennylane as qp

    pl_qfunc_from_pq = qp.from_pyquil(pq_program)

    @qp.qnode(qp.device("default.qubit"))
    def pl_circuit_from_pq():
        pl_qfunc_from_pq(wires=[0, 1])
        return qp.expval(qp.Y(0)), qp.var(qp.Z(1))

The resulting PennyLane circuit is:

>>> print(qp.draw(pl_circuit_from_pq)())
0: ──H─╭●─┤  <Y>
1: ────╰X─┤  Var[Z]

.. note::

    Quantum circuits expressed in Quil can be imported in a similar way using
    :func:`~pennylane.from_quil`.

Importing quantum operators
---------------------------

As well as circuits, it can be useful to import operators defined in other frameworks into
PennyLane. This can be useful for workflows that involve calculating the expectation value of an
observable. By mapping to PennyLane, we can make the workflow differentiable while maintaining
access to features like grouping for hardware-efficient execution.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.from_qiskit_op

:html:`</div>`


Presently, only Qiskit `SparsePauliOp
<https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp>`__ operators can be
imported into PennyLane. To see this in action, we first define a ``SparsePauliOp``:

.. code-block:: python

    from qiskit.quantum_info import SparsePauliOp

    qk_op = SparsePauliOp(["II", "XY"])

Then, we apply the :func:`~pennylane.from_qiskit_op` function to convert the ``SparsePauliOp`` into
a PennyLane :class:`Operator <pennylane.operation.Operator>`:

.. code-block:: python

    import pennylane as qp

    pl_op = qp.from_qiskit_op(qk_op)

We can inspect both operators to make sure they match:

>>> qk_op
SparsePauliOp(['II', 'XY'],
              coeffs=[1.+0.j, 1.+0.j])
>>> pl_op
I(0) + X(1) @ Y(0)
