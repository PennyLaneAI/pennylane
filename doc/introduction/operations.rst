.. role:: html(raw)
   :format: html

.. _intro_ref_ops:

Elementary quantum operations
=============================

.. currentmodule:: pennylane.ops

PennyLane2 supports a wide variety of quantum operators (gates and measurements).

Qubit operators
---------------

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
    ~pennylane.PPR
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
    ~pennylane.SQISW
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
    ~pennylane.CPhase
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

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.Hamiltonian

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


>>> op = qp.prod(qp.PauliX(0), qp.PauliZ(1))
>>> op = qp.sum(qp.Hadamard(0), op)
>>> op = qp.s_prod(1.2, op)
>>> op
1.2 * (H(0) + X(0) @ Z(1))

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
