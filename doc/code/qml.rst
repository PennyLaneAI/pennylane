.. _api_qml:

qml
===

.. currentmodule:: pennylane

**Module**: qml

This module allows the direct import of all basic functions and classes of PennyLane.

As a convention, pennylane is imported via:

.. code::

    import pennylane as qml

The top-level module contains the following sub-modules:

* :ref:`init <api_qml_init>` - This module contains functions that generate initial parameters,
  for example to use in templates.
* :ref:`templates <api_qml_temp>` -This module provides a growing library of
  templates of common variational circuit architectures that can be used to easily build,
  evaluate, and train quantum nodes.


About
-----

.. currentmodule:: pennylane
.. autosummary::
    about
    version

Configuration
-------------

.. currentmodule:: pennylane.configuration
.. autosummary::
    Configuration

Device
------

.. currentmodule:: pennylane
.. autosummary::
    Device
    device

Gradients
---------

.. currentmodule:: pennylane
.. autosummary::
    grad
    jacobian


Measurements
------------

.. currentmodule:: pennylane.measure
.. autosummary::
    expval
    sample
    var

Operations base class
---------------------

.. currentmodule:: pennylane.operation
.. autosummary::
    CVObservable
    CVOperation
    Observable
    Operation



Operations - CV
---------------

.. currentmodule:: pennylane.ops.cv
.. autosummary::
    Beamsplitter
    CatState
    CoherentState
    ControlledAddition
    ControlledPhase
    CrossKerr
    CubicPhase
    DisplacedSqueezedState
    Displacement
    FockDensityMatrix
    FockState
    FockStateProjector
    FockStateVector
    GaussianState
    Interferometer
    Kerr
    NumberOperator
    P
    PolyXP
    QuadOperator
    QuadraticPhase
    Rotation
    SqueezedState
    Squeezing
    ThermalState
    TwoModeSqueezing
    X



Operations - Qubits
-------------------

.. currentmodule:: pennylane.ops.qubit
.. autosummary::
    BasisState
    CNOT
    CRot
    CRX
    CRY
    CRZ
    CSWAP
    CZ
    Hadamard
    Hermitian
    PauliX
    PauliY
    PauliZ
    PhaseShift
    QubitStateVector
    QubitUnitary
    Rot
    RX
    RY
    RZ
    S
    SWAP
    T

Operations - Shared
-------------------

.. currentmodule:: pennylane.ops
.. autosummary::
    Identity

Optimizers
----------

.. _api_qml_opt:

.. currentmodule:: pennylane.optimize
.. autosummary::
    AdagradOptimizer
    AdamOptimizer
    GradientDescentOptimizer
    MomentumOptimizer
    NesterovMomentumOptimizer
    QNGOptimizer
    RMSPropOptimizer


QNode
-----

.. currentmodule:: pennylane
.. autosummary::
    QNode
    qnode


.. toctree::
    :hidden:

    pennylane.about.rst
    pennylane.configuration.Configuration.rst
    pennylane.decorator.qnode.rst
    pennylane._device.Device.rst
    pennylane.device.rst
    pennylane.grad.rst
    pennylane.jacobian.rst
    pennylane.measure.expval.rst
    pennylane.measure.sample.rst
    pennylane.measure.var.rst
    pennylane.operation.CVObservable.rst
    pennylane.operation.CVOperation.rst
    pennylane.operation.Observable.rst
    pennylane.operation.Operation.rst
    pennylane.ops.cv.CatState.rst
    pennylane.ops.cv.CoherentState.rst
    pennylane.ops.cv.ControlledAddition.rst
    pennylane.ops.cv.ControlledPhase.rst
    pennylane.ops.cv.CrossKerr.rst
    pennylane.ops.cv.CubicPhase.rst
    pennylane.ops.cv.DisplacedSqueezedState.rst
    pennylane.ops.cv.Displacement.rst
    pennylane.ops.cv.FockDensityMatrix.rst
    pennylane.ops.cv.FockStateProjector.rst
    pennylane.ops.cv.FockState.rst
    pennylane.ops.cv.FockStateVector.rst
    pennylane.ops.cv.GaussianState.rst
    pennylane.ops.cv.Interferometer.rst
    pennylane.ops.cv.Kerr.rst
    pennylane.ops.cv.NumberOperator.rst
    pennylane.ops.cv.PolyXP.rst
    pennylane.ops.cv.P.rst
    pennylane.ops.cv.QuadOperator.rst
    pennylane.ops.cv.QuadraticPhase.rst
    pennylane.ops.cv.Rotation.rst
    pennylane.ops.cv.SqueezedState.rst
    pennylane.ops.cv.Squeezing.rst
    pennylane.ops.cv.ThermalState.rst
    pennylane.ops.cv.TwoModeSqueezing.rst
    pennylane.ops.cv.X.rst
    pennylane.ops.Identity.rst
    pennylane.ops.qubit.BasisState.rst
    pennylane.ops.qubit.CNOT.rst
    pennylane.ops.qubit.CRot.rst
    pennylane.ops.qubit.CRX.rst
    pennylane.ops.qubit.CRY.rst
    pennylane.ops.qubit.CRZ.rst
    pennylane.ops.qubit.CSWAP.rst
    pennylane.ops.qubit.CZ.rst
    pennylane.ops.qubit.Hadamard.rst
    pennylane.ops.qubit.Hermitian.rst
    pennylane.ops.qubit.PauliX.rst
    pennylane.ops.qubit.PauliY.rst
    pennylane.ops.qubit.PauliZ.rst
    pennylane.ops.qubit.PhaseShift.rst
    pennylane.ops.qubit.QubitStateVector.rst
    pennylane.ops.qubit.QubitUnitary.rst
    pennylane.ops.qubit.Rot.rst
    pennylane.ops.qubit.RX.rst
    pennylane.ops.qubit.RY.rst
    pennylane.ops.qubit.RZ.rst
    pennylane.ops.qubit.S.rst
    pennylane.ops.qubit.SWAP.rst
    pennylane.ops.qubit.T.rst
    pennylane.optimize.AdagradOptimizer.rst
    pennylane.optimize.AdamOptimizer.rst
    pennylane.optimize.GradientDescentOptimizer.rst
    pennylane.optimize.MomentumOptimizer.rst
    pennylane.optimize.NesterovMomentumOptimizer.rst
    pennylane.optimize.QNGOptimizer.rst
    pennylane.optimize.RMSPropOptimizer.rst
    pennylane.qnode.QNode.rst
    pennylane.version.rst



