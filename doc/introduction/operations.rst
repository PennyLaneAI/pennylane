.. role:: html(raw)
   :format: html

.. _intro_ref_ops:

Quantum operators
=================

.. currentmodule:: pennylane.ops

PennyLane supports a wide variety of quantum operators---such as gates, noisy channels, state preparations and measurements.
These operators can be used in quantum functions, like shown in the following example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        qml.AmplitudeDamping(0.1, wires=0)
        return qml.expval(qml.PauliZ(1))

This quantum function uses the :class:`RZ <pennylane.RZ>`,
:class:`CNOT <pennylane.CNOT>`,
:class:`RY <pennylane.RY>` gates, the
:class:`AmplitudeDamping <pennylane.AmplitudeDamping>`
noisy channel as well as the
:class:`PauliZ <pennylane.PauliZ>` observable.

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

    ~pennylane.adjoint
    ~pennylane.ctrl
    ~pennylane.cond
    ~pennylane.exp
    ~pennylane.op_sum
    ~pennylane.prod
    ~pennylane.s_prod
    ~pennylane.generator

These operator functions act on operators to produce new operators.

>>> op = qml.prod(qml.PauliX(0), qml.PauliZ(1))
>>> op = qml.op_sum(qml.Hadamard(0), op)
>>> op = qml.s_prod(1.2, op)
>>> op
1.2*(Hadamard(wires=[0]) + (PauliX(wires=[0]) @ PauliZ(wires=[1])))

Operator to Other functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~pennylane.matrix
    ~pennylane.eigvals
    ~pennylane.is_commuting
    ~pennylane.is_hermitian
    ~pennylane.is_unitary
    ~pennylane.simplify

These operator functions act on operators and return other data types.
All operator functions can be used on instantiated operators.

>>> op = qml.RX(0.54, wires=0)
>>> qml.matrix(op)
[[0.9637709+0.j         0.       -0.26673144j]
[0.       -0.26673144j 0.9637709+0.j        ]]

Some operator functions can also be used in a functional form:

>>> x = torch.tensor(0.6, requires_grad=True)
>>> matrix_fn = qml.matrix(qml.RX)
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
...     qml.RX(theta, wires=1)
...     qml.PauliZ(wires=0)
>>> qml.matrix(circuit)(np.pi / 4)
array([[ 0.92387953+0.j,  0.+0.j ,  0.-0.38268343j,  0.+0.j],
[ 0.+0.j,  -0.92387953+0.j,  0.+0.j,  0. +0.38268343j],
[ 0. -0.38268343j,  0.+0.j,  0.92387953+0.j,  0.+0.j],
[ 0.+0.j,  0.+0.38268343j,  0.+0.j,  -0.92387953+0.j]])


.. _intro_ref_ops_qubit:

Qubit operators
---------------

.. _intro_ref_ops_nonparam:

Non-parametrized gates
^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.S
    ~pennylane.T
    ~pennylane.SX
    ~pennylane.CNOT
    ~pennylane.CZ
    ~pennylane.CY
    ~pennylane.SWAP
    ~pennylane.ISWAP
    ~pennylane.ECR
    ~pennylane.SISWAP
    ~pennylane.SQISW
    ~pennylane.CSWAP
    ~pennylane.Toffoli
    ~pennylane.MultiControlledX
    ~pennylane.Barrier
    ~pennylane.WireCut

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
    ~pennylane.CPhaseShift00
    ~pennylane.CPhaseShift01
    ~pennylane.CPhaseShift10
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
    ~pennylane.PSWAP

:html:`</div>`

.. _intro_ref_ops_qchem:

Quantum chemistry gates
^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.SingleExcitation
    ~pennylane.SingleExcitationPlus
    ~pennylane.SingleExcitationMinus
    ~pennylane.DoubleExcitation
    ~pennylane.DoubleExcitationPlus
    ~pennylane.DoubleExcitationMinus
    ~pennylane.OrbitalRotation

:html:`</div>`

Electronic Hamiltonians built independently using
`OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools can be readily converted to a
PennyLane observable using the :func:`~.pennylane.import_operator` function.

.. _intro_ref_ops_matrix:

Gates constructed from a matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.QubitUnitary
    ~pennylane.ControlledQubitUnitary
    ~pennylane.DiagonalQubitUnitary

:html:`</div>`

.. _intro_ref_ops_arithm:

Gates performing arithmetics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.QubitCarry
    ~pennylane.QubitSum

:html:`</div>`

.. _intro_ref_ops_qstateprep:

State preparation
^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.QubitStateVector
    ~pennylane.QubitDensityMatrix

:html:`</div>`


.. _intro_ref_ops_channels:

Noisy channels
^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.AmplitudeDamping
    ~pennylane.GeneralizedAmplitudeDamping
    ~pennylane.PhaseDamping
    ~pennylane.DepolarizingChannel
    ~pennylane.BitFlip
    ~pennylane.PhaseFlip
    ~pennylane.ResetError
    ~pennylane.PauliError
    ~pennylane.QubitChannel
    ~pennylane.ThermalRelaxationError

:html:`</div>`


.. _intro_ref_ops_qobs:

Observables
^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Hadamard
    ~pennylane.Hermitian
    ~pennylane.Identity
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.Projector
    ~pennylane.Hamiltonian
    ~pennylane.SparseHamiltonian

:html:`</div>`

.. _intro_ref_ops_cv:

Continuous-Variable (CV) operators
-----------------------------------

If you would like to learn more about the CV model of quantum computing, check out the
`quantum photonics <https://strawberryfields.ai/photonics/concepts/photonics.html>`_
page of the `Strawberry Fields <https://strawberryfields.ai/>`__ documentation.

.. _intro_ref_ops_cvgates:

CV gates
^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.Beamsplitter
    ~pennylane.ControlledAddition
    ~pennylane.ControlledPhase
    ~pennylane.CrossKerr
    ~pennylane.CubicPhase
    ~pennylane.Displacement
    ~pennylane.InterferometerUnitary
    ~pennylane.Kerr
    ~pennylane.QuadraticPhase
    ~pennylane.Rotation
    ~pennylane.Squeezing
    ~pennylane.TwoModeSqueezing

:html:`</div>`

.. _intro_ref_ops_cvstateprep:

CV state preparation
^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.CatState
    ~pennylane.CoherentState
    ~pennylane.DisplacedSqueezedState
    ~pennylane.FockDensityMatrix
    ~pennylane.FockState
    ~pennylane.FockStateVector
    ~pennylane.GaussianState
    ~pennylane.SqueezedState
    ~pennylane.ThermalState

:html:`</div>`

.. _intro_ref_ops_cvobs:

CV observables
^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.FockStateProjector
    ~pennylane.Identity
    ~pennylane.NumberOperator
    ~pennylane.TensorN
    ~pennylane.P
    ~pennylane.PolyXP
    ~pennylane.QuadOperator
    ~pennylane.X

:html:`</div>`

.. _intro_ref_ops_qutrit:

Qutrit operators
----------------

.. _intro_ref_ops_qutrit_nonparam:

Qutrit non-parametrized gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.TShift
    ~pennylane.TClock
    ~pennylane.TAdd
    ~pennylane.TSWAP

:html:`</div>`

.. _intro_ref_ops_qutrit_matrix:

Qutrit gates constructed from a matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.QutritUnitary

:html:`</div>`

.. _intro_ref_ops_qutrit_obs:

Qutrit Observables
^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.THermitian
    ~pennylane.GellMann

:html:`</div>`

