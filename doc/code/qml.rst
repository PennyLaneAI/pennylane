.. _docs_pennylane:

qml
===

.. currentmodule:: pennylane

**Module**: qml

This top level module allows the direct import of all basic functions and classes of PennyLane.

As a convention, pennylane is imported via:

.. code::

    import pennylane as qml

Modules
-------

.. autosummary::
    init
    templates

Classes
-------

Configuration

.. currentmodule:: pennylane.configuration
.. autosummary::
    Configuration

Operations - Continuous-Variable

.. currentmodule:: pennylane.ops.cv
.. autosummary::
    Rotation
    Squeezing
    Displacement
    Beamsplitter
    TwoModeSqueezing
    QuadraticPhase
    ControlledAddition
    ControlledPhase
    Kerr
    CrossKerr
    CubicPhase
    Interferometer
    CoherentState
    SqueezedState
    DisplacedSqueezedState
    ThermalState
    GaussianState
    FockState
    FockStateVector
    FockDensityMatrix
    CatState
    NumberOperator
    X
    P
    QuadOperator
    PolyXP
    FockStateProjector

Operations - Qubits

.. currentmodule:: pennylane.ops.qubit
.. autosummary::
    Hadamard
    PauliX
    PauliY
    PauliZ
    CNOT
    CZ
    SWAP
    CSWAP
    RX
    RY
    RZ
    PhaseShift
    Rot
    CRX
    CRY
    CRZ
    CRot
    QubitUnitary
    BasisState
    QubitStateVector
    Hadamard
    PauliX
    PauliY
    PauliZ
    Hermitian

Operations - Shared

.. currentmodule:: pennylane.ops
.. autosummary::
    Identity

Optimizers

.. _docs_qml_optimize:

.. currentmodule:: pennylane.optimize
.. autosummary::
    AdagradOptimizer
    AdamOptimizer
    GradientDescentOptimizer
    MomentumOptimizer
    NesterovMomentumOptimizer
    RMSPropOptimizer
    QNGOptimizer

QNode

.. currentmodule:: pennylane.qnode
.. autosummary::
    QNode

Functions
---------

About

.. currentmodule:: pennylane
.. autosummary::
    about
    version

Device

.. currentmodule:: pennylane
.. autosummary::
    device

Gradients

.. currentmodule:: pennylane
.. autosummary::
    grad
    jacobian

Measurements

.. currentmodule:: pennylane.measure
.. autosummary::
    expval
    var
    sample

QNode Decorator

.. _docs_qml_decorator:

.. currentmodule:: pennylane.decorator
.. autosummary::
    qnode


.. toctree::
    :hidden:

    generated/pennylane.about.rst
    generated/pennylane.jacobian.rst
    generated/pennylane.grad.rst
    generated/pennylane.measure.sample.rst
    generated/pennylane.measure.var.rst
    generated/pennylane.measure.expval.rst
    generated/pennylane.ops.cv.FockStateProjector.rst
    generated/pennylane.ops.cv.CubicPhase.rst
    generated/pennylane.ops.qubit.Hermitian.rst
    generated/pennylane.ops.qubit.RZ.rst
    generated/pennylane.ops.cv.Squeezing.rst
    generated/pennylane.ops.cv.PolyXP.rst
    generated/pennylane.ops.qubit.QubitUnitary.rst
    generated/pennylane.optimize.AdagradOptimizer.rst
    generated/pennylane.ops.cv.X.rst
    generated/pennylane.ops.qubit.PauliX.rst
    generated/pennylane.ops.cv.ControlledAddition.rst
    generated/pennylane.ops.cv.Rotation.rst
    generated/pennylane.device.rst
    generated/pennylane.ops.qubit.QubitStateVector.rst
    generated/pennylane.ops.qubit.RX.rst
    generated/pennylane.ops.cv.CatState.rst
    generated/pennylane.ops.qubit.CRY.rst
    generated/pennylane.optimize.RMSPropOptimizer.rst
    generated/pennylane.ops.qubit.Rot.rst
    generated/pennylane.optimize.MomentumOptimizer.rst
    generated/pennylane.optimize.NesterovMomentumOptimizer.rst
    generated/pennylane.optimize.NesterovMomentumOptimizer.rst
    generated/pennylane.ops.cv.QuadOperator.rst
    generated/pennylane.ops.cv.P.rst

    generated/pennylane.init.random_layers_uniform.rst
    generated/pennylane.init.random_layer_uniform.rst
    generated/pennylane.init.strong_ent_layer_normal.rst
    generated/pennylane.init.strong_ent_layer_normal.rst

    generated/pennylane.templates.layers.Interferometer.rst
    generated/pennylane.templates.embeddings.DisplacementEmbedding.rst
    generated/pennylane.templates.layers.RandomLayer.rst
    generated/pennylane.ops.cv.NumberOperator.rst
    generated/pennylane.version.rst
    generated/pennylane.ops.qubit.RY.rst
    generated/pennylane.templates.embeddings.AngleEmbedding.rst
    generated/pennylane.ops.cv.TwoModeSqueezing.rst
    generated/pennylane.about.rst
    generated/pennylane.ops.cv.Beamsplitter.rst
    generated/pennylane.qnode.QNode.rst
    generated/pennylane.ops.cv.ControlledPhase.rst
    generated/pennylane.templates.embeddings.AmplitudeEmbedding.rst
    generated/pennylane.init.strong_ent_layer_uniform.rst
    generated/pennylane.ops.Identity.rst
    generated/pennylane.ops.cv.DisplacedSqueezedState.rst
    generated/pennylane.ops.qubit.PhaseShift.rst
    generated/pennylane.templates.embeddings.SqueezingEmbedding.rst
    generated/pennylane.ops.qubit.CRZ.rst
    generated/pennylane.ops.qubit.Hadamard.rst
    generated/pennylane.ops.qubit.CSWAP.rst
    generated/pennylane.templates.layers.RandomLayers.rst
    generated/pennylane.templates.layers.StronglyEntanglingLayer.rst
    generated/pennylane.ops.cv.QuadraticPhase.rst
    generated/pennylane.init.cvqnn_layer_normal.rst
    generated/pennylane.ops.cv.SqueezedState.rst
    generated/pennylane.init.random_layer_normal.rst
    generated/pennylane.templates.layers.CVNeuralNetLayer.rst
    generated/pennylane.init.interferometer_uniform.rst
    generated/pennylane.ops.qubit.SWAP.rst
    generated/pennylane.ops.cv.Interferometer.rst
    generated/pennylane.ops.qubit.BasisState.rst
    generated/pennylane.ops.cv.FockStateVector.rst
    generated/pennylane.templates.layers.StronglyEntanglingLayers.rst
    generated/pennylane.ops.qubit.PauliZ.rst
    generated/pennylane.ops.qubit.CRX.rst
    generated/pennylane.ops.cv.Kerr.rst
    generated/pennylane.templates.layers.CVNeuralNetLayers.rst
    generated/pennylane.ops.qubit.PauliY.rst
    generated/pennylane.init.interferometer_normal.rst
    generated/pennylane.init.cvqnn_layer_uniform.rst
    generated/pennylane.optimize.GradientDescentOptimizer.rst
    generated/pennylane.ops.cv.FockDensityMatrix.rst
    generated/pennylane.ops.cv.ThermalState.rst
    generated/pennylane.init.cvqnn_layers_uniform.rst
    generated/pennylane.init.random_layers_normal.rst
    generated/pennylane.ops.cv.CrossKerr.rst
    generated/pennylane.optimize.AdamOptimizer.rst
    generated/pennylane.decorator.qnode.rst
    generated/pennylane.init.strong_ent_layers_normal.rst
    generated/pennylane.ops.cv.FockState.rst
    generated/pennylane.init.cvqnn_layers_normal.rst
    generated/pennylane.ops.cv.CoherentState.rst
    generated/pennylane.templates.embeddings.BasisEmbedding.rst
    generated/pennylane.ops.qubit.CZ.rst
    generated/pennylane.ops.cv.GaussianState.rst
    generated/pennylane.ops.qubit.CRot.rst
    generated/pennylane.init.strong_ent_layers_uniform.rst
    generated/pennylane.optimize.QNGOptimizer.rst
    generated/pennylane.ops.qubit.CNOT.rst
    generated/pennylane.configuration.Configuration.rst
    generated/pennylane.ops.cv.Displacement.rst




