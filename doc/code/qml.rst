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

.. autosummary::
    :toctree: api

    about
    version

CircuitGraph
------------

.. autosummary::
    :toctree: api

    CircuitGraph

Configuration
-------------

.. autosummary::
    :toctree: api/configuration

    Configuration

Devices
-------

.. autosummary::
    :toctree: api/_device

    Device

.. autosummary::
    :toctree: api

    device

Gradients
---------

.. autosummary::
    :toctree: api

    grad
    jacobian

Measurements
------------

.. autosummary::
    :toctree: api

    expval
    sample
    var

Operator base classes
---------------------

.. currentmodule:: pennylane.operation
.. autosummary::
    :toctree: api

    CVObservable
    CVOperation
    Observable
    Operation
    Operator

Operations - CV
---------------

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

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

.. autosummary::
    :toctree: api

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

.. autosummary::
    :toctree: api

    Identity

Optimizers
----------

.. _api_qml_opt:

.. autosummary::
    :toctree: api

    AdagradOptimizer
    AdamOptimizer
    GradientDescentOptimizer
    MomentumOptimizer
    NesterovMomentumOptimizer
    QNGOptimizer
    RMSPropOptimizer

QNode
-----

.. autosummary::
    :toctree: api/decorator

    qnode

.. autosummary::
    :toctree: api

    QNode
