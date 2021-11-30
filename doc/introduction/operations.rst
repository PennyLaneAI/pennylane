.. role:: html(raw)
   :format: html

.. _intro_ref_ops:

Quantum operations
==================

.. currentmodule:: pennylane.ops

PennyLane supports a wide variety of quantum operations---such as gates, noisy channels, state preparations and measurements.
These operations can be used exclusively in quantum functions, like shown in the following example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        qml.T(wires=0).inv()
        qml.AmplitudeDamping(0.1, wires=0)
        return qml.expval(qml.PauliZ(1))

This quantum function uses the :class:`RZ <pennylane.RZ>`,
:class:`CNOT <pennylane.CNOT>`,
:class:`RY <pennylane.RY>` :ref:`gates <intro_ref_ops_qgates>`, the
:class:`AmplitudeDamping <pennylane.AmplitudeDamping>`
:ref:`noisy channel <intro_ref_ops_channels>` as well as the
:class:`PauliZ <pennylane.PauliZ>` :ref:`observable <intro_ref_ops_qobs>`.

Note that PennyLane supports inverting quantum operations via the
:meth:`Op(param, wires).inv() <.Operation.inv>` method. Additionally, PennyLane
provides a function :func:`qml.inv <.pennylane.inv>` that can be used to invert sequences
of operations and :doc:`templates`.

Below is a list of all quantum operations supported by PennyLane.

.. _intro_ref_ops_qubit:

Qubit operations
----------------

.. _intro_ref_ops_qgates:

Non-parametric Ops
^^^^^^^^^^^^^^^^^^ 


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
    ~pennylane.SISWAP
    ~pennylane.SQISW
    ~pennylane.CSWAP
    ~pennylane.Toffoli
    ~pennylane.MultiControlledX
    ~pennylane.Barrier

:html:`</div>`


Parametric Ops
^^^^^^^^^^^^^^


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
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.CRot
    ~pennylane.U1
    ~pennylane.U2
    ~pennylane.U3
    ~pennylane.IsingXX
    ~pennylane.IsingYY
    ~pennylane.IsingZZ

:html:`</div>`


Quantum Chemistry Ops
^^^^^^^^^^^^^^^^^^^^^


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


Matrix Ops
^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.QubitUnitary
    ~pennylane.ControlledQubitUnitary
    ~pennylane.DiagonalQubitUnitary

:html:`</div>`


Arithmetic Ops
^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.QubitCarry
    ~pennylane.QubitSum

:html:`</div>`


Qubit state preparation
^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.QubitStateVector

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

Qubit observables
^^^^^^^^^^^^^^^^^

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

Grouping Pauli words
^^^^^^^^^^^^^^^^^^^^

Grouping Pauli words can be used for the optimizing the measurement of qubit
Hamiltonians. Along with groups of observables, post-measurement rotations can
also be obtained using :func:`~.optimize_measurements`:

.. code-block:: python

    >>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> post_rotations, diagonalized_groupings, grouped_coeffs = optimize_measurements(obs, coeffs)
    >>> post_rotations
    [[RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
     [RX(1.5707963267948966, wires=[0])]]

The post-measurement rotations can be used to diagonalize the partitions of
observables found.

For further details on measurement optimization, grouping observables through
solving the minimum clique cover problem, and auxiliary functions, refer to the
:doc:`/code/qml_grouping` subpackage.


.. _intro_ref_ops_cv:

Continuous-variable (CV) operations
-----------------------------------

.. _intro_ref_ops_cvgates:

CV Gates
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

