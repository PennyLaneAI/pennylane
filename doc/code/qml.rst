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
    :toctree: api

    about
    version

CircuitGraph
------------

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    CircuitGraph

Configuration
-------------

.. currentmodule:: pennylane.configuration
.. autosummary::
    :toctree: api

    Configuration

Devices
-------

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api/_device

    Device

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    device

Gradients
---------

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    grad
    jacobian

Measurements
------------

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    expval
    sample
    var

Operations base class
---------------------

.. currentmodule:: pennylane.operation
.. autosummary::
    :toctree: api

    CVObservable
    CVOperation
    Observable
    Operation

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

.. currentmodule:: pennylane
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

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    Identity

Optimizers
----------

.. _api_qml_opt:

.. currentmodule:: pennylane
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

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api/decorator

    qnode

.. autosummary::
    :toctree: api

    QNode
