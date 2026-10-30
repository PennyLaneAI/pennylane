.. role:: html(raw)
   :format: html

.. _intro_ref_ops:

Elementary quantum operations
=============================

.. currentmodule:: pennylane.ops

PennyLane2 supports a wide variety of quantum operations.

Qubit operations
----------------

.. _intro_ref_ops_nonparam:

Non-parametrized gates and operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.Hadamard
    ~pennylane.S
    ~pennylane.CNOT
    ~pennylane.T
    ~pennylane.Toffoli
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.measure
    ~pennylane.PPR_2
    ~pennylane.PPR_4
    ~pennylane.PPR_8
    ~pennylane.pauli_measure
    ~pennylane.SX
    ~pennylane.CY
    ~pennylane.CZ
    ~pennylane.CH
    ~pennylane.CCZ
    ~pennylane.SWAP
    ~pennylane.ISWAP
    ~pennylane.CSWAP
    ~pennylane.SISWAP
    ~pennylane.ECR
    ~pennylane.MultiControlledX

:html:`</div>`

.. _intro_ref_ops_qparam:

Parametrized gates
^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Rot
    ~pennylane.RX
    ~pennylane.RY
    ~pennylane.RZ
    ~pennylane.MultiRZ
    ~pennylane.PauliRot
    ~pennylane.PhaseShift
    ~pennylane.ControlledPhaseShift
    ~pennylane.PCPhase
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.CRot
    ~pennylane.U1
    ~pennylane.U2
    ~pennylane.U3
    ~pennylane.IsingXX
    ~pennylane.IsingXY
    ~pennylane.IsingYY
    ~pennylane.IsingZZ
    ~pennylane.GlobalPhase

:html:`</div>`

.. _intro_ref_ops_other:

Other gates
^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.SingleExcitation
    ~pennylane.QubitUnitary
    ~pennylane.ControlledQubitUnitary
    ~pennylane.DiagonalQubitUnitary
    ~pennylane.BlockEncode
    ~pennylane.BasisState
    ~pennylane.ChangeOpBasis2

:html:`</div>`

Observables
^^^^^^^^^^^

The following operations can be the argument of certain
:doc:`terminal measurements <measurements>`, like :func:`~.expval` and
:func:`~.sample`.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.ops.LinearCombination

:html:`</div>`

.. _intro_ref_ops_funcs:

Operator functions
------------------

Various functions are available for manipulating operators. These can be broken
down into two main categories:

Gate modifiers
^^^^^^^^^^^^^^

These functions create valid quantum circuit operations from other operators.

.. autosummary::

    ~pennylane.adjoint
    ~pennylane.ctrl
    ~pennylane.cond
    ~pennylane.exp
    ~pennylane.pow
    ~pennylane.prod
    ~pennylane.evolve

For example,

.. code-block:: python

    import pennylane as qp

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit(x):
        qp.adjoint(qp.S)(wires=0)
        qp.ctrl(qp.RX, control=0)(x, wires=1)
        qp.pow(qp.T(wires=1), z=2)
        qp.evolve(qp.X(0) @ qp.X(1), x)
        return qp.expval(qp.Z(1))

>>> print(qp.analyze(circuit)(0.5))
Device: lightning.qubit
Device wires: 2
Shots: Shots(total=None)
Level: Before MLIR Passes
<BLANKLINE>
Quantum operations:
- Total: 4
  - Adjoint(S): 1
  - C(RX): 1
  - Evolution: 1
  - Pow2: 1
Measurement processes:
- expval(PauliZ): 1
Total wires: 2
Circuit Depth: Not computed

Utility functions
^^^^^^^^^^^^^^^^^

These functions are useful for back-of-the-napkin calculations, data processing,
and more.

.. autosummary::

    ~pennylane.generator
    ~pennylane.simplify
    ~pennylane.dot
    ~pennylane.map_wires
    ~pennylane.s_prod
    ~pennylane.sum
    ~pennylane.matrix
    ~pennylane.eigvals
    ~pennylane.is_commuting
    ~pennylane.is_hermitian
    ~pennylane.is_unitary
    ~pennylane.pauli_decompose

For example,

>>> qp.dot([0.5, 0.3], [qp.X(0), qp.Z(0) @ qp.Z(1)])
0.5 * X(0) + 0.3 * (Z(0) @ Z(1))
>>> qp.simplify(qp.X(0) @ qp.X(0) @ qp.Z(1))
Z(1)
>>> qp.map_wires(qp.CNOT([0, 1]), {0: "a", 1: "b"})
CNOT(wires=['a', 'b'])
>>> qp.matrix(qp.Hadamard(0))
array([[ 0.70710678,  0.70710678],
       [ 0.70710678, -0.70710678]])
>>> qp.is_commuting(qp.X(0), qp.Z(1))
True
>>> qp.pauli_decompose(qp.math.array([[1, 1], [1, -1]]))
1.0 * X(0) + 1.0 * Z(0)
