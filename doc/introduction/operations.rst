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

The following operations can be the argument of certain
:doc:`terminal measurements <measurements>`, like :func:`~.expval` and
`~.sample`.

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


TODO: update code examples

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

TODO: update code examples
