.. role:: html(raw)
   :format: html

.. _intro_ref_ops:

Quantum operators
=================

.. currentmodule:: pennylane2.ops

PennyLane supports a wide variety of quantum operators---such as gates, noisy channels, state preparations and measurements.
These operators can be used in quantum functions, like shown in the following example:

.. code-block:: python

    import pennylane2 as qp

    def my_quantum_function(x, y):
        qp.RZ(x, wires=0)
        qp.CNOT(wires=[0,1])
        qp.RY(y, wires=1)
        qp.AmplitudeDamping(0.1, wires=0)
        return qp.expval(qp.PauliZ(1))

This quantum function uses the :class:`RZ <pennylane2.RZ>`,
:class:`CNOT <pennylane2.CNOT>`,
:class:`RY <pennylane2.RY>` gates, the
:class:`AmplitudeDamping <pennylane2.AmplitudeDamping>`
noisy channel as well as the
:class:`PauliZ <pennylane2.PauliZ>` observable.

Functions applied to operators extract information (such as the matrix representation) or
transform operators (like turning a gate into a controlled gate).

PennyLane supports the following operators and operator functions:


.. _intro_ref_ops_funcs:

Operator functions
------------------

Various functions and transforms are available for manipulating operators,
and extracting information. These can be broken down into two main categories:

Operator to Operator functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~pennylane2.adjoint
    ~pennylane2.ctrl
    ~pennylane2.cond
    ~pennylane2.exp
    ~pennylane2.sum
    ~pennylane2.pow
    ~pennylane2.prod
    ~pennylane2.s_prod
    ~pennylane2.generator
    ~pennylane2.map_wires
    ~pennylane2.dot
    ~pennylane2.evolve
    ~pennylane2.simplify

These operator functions act on operators to produce new operators.

>>> op = qp.prod(qp.PauliX(0), qp.PauliZ(1))
>>> op = qp.sum(qp.Hadamard(0), op)
>>> op = qp.s_prod(1.2, op)
>>> op
1.2 * (H(0) + X(0) @ Z(1))

Operator to Other functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~pennylane2.matrix
    ~pennylane2.eigvals
    ~pennylane2.is_commuting
    ~pennylane2.is_hermitian
    ~pennylane2.is_unitary
    ~pennylane2.iterative_qpe


These operator functions act on operators and return other data types.
All operator functions can be used on instantiated operators.

>>> op = qp.RX(0.54, wires=0)
>>> qp.matrix(op)
array([[0.9637709+0.j        , 0.       -0.26673144j],
       [0.       -0.26673144j, 0.9637709+0.j        ]])

Some operator functions can also be used in a functional form:

>>> x = torch.tensor(0.6, requires_grad=True)
>>> matrix_fn = qp.matrix(qp.RX)
>>> matrix_fn(x, wires=0)
tensor([[0.9553+0.0000j, 0.0000-0.2955j],
        [0.0000-0.2955j, 0.9553+0.0000j]], grad_fn=<StackBackward0>)

In the functional form, they are usually differentiable with respect to gate arguments:

>>> loss = torch.real(torch.trace(matrix_fn(x, wires=0)))
>>> loss.backward()
>>> x.grad
tensor(-0.2955)

Some operator transforms can also act on multiple operators, by passing
quantum functions, QNodes or tapes:

>>> def circuit(theta):
...     qp.RX(theta, wires=1)
...     qp.Z(wires=0)
>>> qp.matrix(circuit, wire_order=[0,1])(np.pi / 4)
array([[ 0.92387953+0.j        ,  0.        -0.38268343j,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        -0.38268343j,  0.92387953+0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        ,  0.        +0.j        ,
        -0.92387953+0.j        ,  0.        +0.38268343j],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.38268343j, -0.92387953+0.j        ]])


Matrix to Operator functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~pennylane2.pauli_decompose
    ~pennylane2.qsvt

These functions take a matrix and return an associated native PennyLane operator.
For example:

>>> mat = np.array([[1, 1], [1, -1]])
>>> h = qp.pauli_decompose(mat)
>>> type(h)
<class 'pennylane2.ops.op_math.linear_combination.LinearCombination'>
>>> print(h)
1.0 * X(0) + 1.0 * Z(0)

.. _intro_ref_ops_qubit:

Qubit operators
---------------

.. _intro_ref_ops_nonparam:

Non-parametrized gates
^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.Identity
    ~pennylane2.Hadamard
    ~pennylane2.S
    ~pennylane2.CNOT
    ~pennylane2.T
    ~pennylane2.Toffoli
    ~pennylane2.PauliX
    ~pennylane2.PauliY
    ~pennylane2.PauliZ
    ~pennylane2.SX
    ~pennylane2.CY
    ~pennylane2.CZ
    ~pennylane2.CH
    ~pennylane2.CCZ
    ~pennylane2.SWAP
    ~pennylane2.ISWAP
    ~pennylane2.CSWAP
    ~pennylane2.SISWAP
    ~pennylane2.ECR
    ~pennylane2.SQISW
    ~pennylane2.MultiControlledX
    ~pennylane2.Barrier

:html:`</div>`

.. _intro_ref_ops_qparam:

Parametrized gates
^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.Rot
    ~pennylane2.RX
    ~pennylane2.RY
    ~pennylane2.RZ
    ~pennylane2.MultiRZ
    ~pennylane2.PauliRot
    ~pennylane2.PPR_2
    ~pennylane2.PPR_4
    ~pennylane2.PPR_8
    ~pennylane2.PhaseShift
    ~pennylane2.ControlledPhaseShift
    ~pennylane2.CPhase
    ~pennylane2.PCPhase
    ~pennylane2.CPhaseShift00
    ~pennylane2.CPhaseShift01
    ~pennylane2.CPhaseShift10
    ~pennylane2.CRX
    ~pennylane2.CRY
    ~pennylane2.CRZ
    ~pennylane2.CRot
    ~pennylane2.U1
    ~pennylane2.U2
    ~pennylane2.U3
    ~pennylane2.IsingXX
    ~pennylane2.IsingXY
    ~pennylane2.IsingYY
    ~pennylane2.IsingZZ
    ~pennylane2.PSWAP
    ~pennylane2.GlobalPhase

:html:`</div>`

.. _intro_ref_ops_qchem:

Quantum chemistry gates
^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.SingleExcitation
    ~pennylane2.SingleExcitationPlus
    ~pennylane2.SingleExcitationMinus
    ~pennylane2.DoubleExcitation
    ~pennylane2.DoubleExcitationPlus
    ~pennylane2.DoubleExcitationMinus
    ~pennylane2.OrbitalRotation
    ~pennylane2.FermionicSWAP

:html:`</div>`

Electronic Hamiltonians built independently using
`OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools can be readily converted to a
PennyLane observable using the :func:`~.pennylane2.import_operator` function.

.. _intro_ref_ops_matrix:

Gates constructed from a matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.QubitUnitary
    ~pennylane2.ControlledQubitUnitary
    ~pennylane2.DiagonalQubitUnitary
    ~pennylane2.SpecialUnitary
    ~pennylane2.BlockEncode

:html:`</div>`

.. _intro_ref_ops_arithm:

Gates performing arithmetics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.QubitCarry
    ~pennylane2.QubitSum
    ~pennylane2.IntegerComparator


:html:`</div>`

.. _intro_ref_ops_qstateprep:

State preparation
^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.BasisState
    ~pennylane2.StatePrep
    ~pennylane2.QubitDensityMatrix

:html:`</div>`


.. _intro_ref_ops_channels:

Noisy channels
^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.AmplitudeDamping
    ~pennylane2.GeneralizedAmplitudeDamping
    ~pennylane2.PhaseDamping
    ~pennylane2.DepolarizingChannel
    ~pennylane2.BitFlip
    ~pennylane2.PhaseFlip
    ~pennylane2.ResetError
    ~pennylane2.PauliError
    ~pennylane2.QubitChannel
    ~pennylane2.ThermalRelaxationError

:html:`</div>`


.. _intro_ref_ops_qobs:

Observables
^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane2.Hadamard
    ~pennylane2.Hermitian
    ~pennylane2.Identity
    ~pennylane2.PauliX
    ~pennylane2.PauliY
    ~pennylane2.PauliZ
    ~pennylane2.Projector
    ~pennylane2.Hamiltonian
    ~pennylane2.SparseHamiltonian

:html:`</div>`
