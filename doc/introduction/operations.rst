.. role:: html(raw)
   :format: html

.. _intro_ref_ops:

Quantum operations
==================

.. currentmodule:: pennylane.ops

PennyLane supports a wide variety of quantum operations---such as gates, state preparations and measurements. These operations can be used exclusively in quantum functions, like shown
in the following example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        qml.T(wires=0).inv()
        return qml.expval(qml.PauliZ(1))

This quantum function uses the :class:`RZ <pennylane.RZ>`,
:class:`CNOT <pennylane.CNOT>`,
:class:`RY <pennylane.RY>` :ref:`gates <intro_ref_ops_qgates>` as well as the
:class:`PauliZ <pennylane.PauliZ>` :ref:`observable <intro_ref_ops_qobs>`.

Note that PennyLane supports inverting quantum opperations via the
:meth:`Op(param, wires).inv() <.Operation.inv>` method. Additionally, PennyLane
provides a function :func:`qml.inv <.pennylane.inv>` that can be used to invert sequences
of operations and :doc:`templates`.

Below is a list of all quantum operations supported by PennyLane.

.. _intro_ref_ops_qubit:

Qubit operations
----------------

.. _intro_ref_ops_qgates:

Qubit gates
^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.CNOT
    ~pennylane.CRot
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.CSWAP
    ~pennylane.CZ
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.PhaseShift
    ~pennylane.QubitUnitary
    ~pennylane.Rot
    ~pennylane.RX
    ~pennylane.RY
    ~pennylane.RZ
    ~pennylane.S
    ~pennylane.SWAP
    ~pennylane.T

:html:`</div>`


Qubit state preparation
^^^^^^^^^^^^^^^^^^^^^^^


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.QubitStateVector

:html:`</div>`


.. _intro_ref_ops_qobs:

Qubit observables
^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Hadamard
    ~pennylane.Hermitian
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ

:html:`</div>`

.. _intro_ref_ops_cv:

Continuous-variable (CV) operations
-----------------------------------

.. _intro_ref_ops_cvgates:

CV Gates
^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Beamsplitter
    ~pennylane.ControlledAddition
    ~pennylane.ControlledPhase
    ~pennylane.CrossKerr
    ~pennylane.CubicPhase
    ~pennylane.Displacement
    ~pennylane.Interferometer
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
    ~pennylane.NumberOperator
    ~pennylane.P
    ~pennylane.PolyXP
    ~pennylane.QuadOperator
    ~pennylane.X

:html:`</div>`

Shared operations
-----------------

The only operation shared by both qubit and continouous-variable architectures is the Identity.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity

:html:`</div>`
