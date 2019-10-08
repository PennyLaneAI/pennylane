 .. role:: html(raw)
   :format: html


.. _intro_vcircuits:

Variational Circuits
====================

In the following we will see how the concept of a :ref:`variational quantum circuit <varcirc>`, the
heart piece of hybrid quantum-classical optimization, is implemented in PennyLane.
We give an overview of quantum :ref:`operations <intro_vcirc_ops>` and :ref:`measurements <intro_vcirc_measure>`
that can be used in quantum circuits, and show how a growing library of
:ref:`templates <intro_vcirc_templates>` allows
for easy use of common types of ansatze or architectures.

You can read up on the theoretical background of variational circuits and hybrid optimization
in the :ref:`Key Concepts <key_concepts>` section.

Creating quantum nodes
----------------------

.. image:: _static/qnode.png
    :align: right
    :width: 180px
    :target: javascript:void(0);


In PennyLane, variational circuits are represented as quantum nodes. A quantum node
is a combination of a :ref:`quantum functions <intro_vcirc_qfunc>` that defines the composition of the circuit,
and a :ref:`device <intro_vcirc_device>` that runs the computation. One can conveniently create quantum nodes using
the quantum node :ref:`decorator <intro_vcirc_decorator>`.

Each classical **interface** uses a different version of a quantum node, and we will here introduce the standard QNode
to use with the NumPy interface. NumPy-interfacing quantum nodes take NumPy datastructures,
such as floats and arrays, and return Numpy data structures.
They can be optimized using NumPy-based :ref:`optimization methods <optimize>`.
Quantum nodes for other PennyLane interfaces like :ref:`PyTorch <torch_interf>` and
:ref:`TensorFlow's Eager mode <tf_interf>` are introduced in the section on :ref:`Interfaces <intro_interfaces>`.

Quantum functions
^^^^^^^^^^^^^^^^^
.. _intro_vcirc_qfunc:

A quantum circuit is constructed as a special Python function, a *quantum circuit function*, or *quantum function* in short.
For example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))


Quantum functions are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the function must consist of only supported PennyLane
  :mod:`operations <pennylane.ops>` or sequences of operations called :mod:`templates <pennylane.templates>`, using one instruction per line.

* The function must always return either a single or a tuple of
  *measured observable values*, by applying a :mod:`measurement function <pennylane.measure>`
  to an :mod:`observable <pennylane.ops>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

.. note::

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operations <pennylane.operation.Operation>` require a QNode in order to perform queuing on initialization.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.


Defining a device
^^^^^^^^^^^^^^^^^
.. _intro_vcirc_device:

To run - and later optimize - a quantum circuit, one needs to first specify a *computational device*.

The device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`~device` loader.

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

PennyLane offers some basic devices such as
some basic devices such as the ``'default.qubit'`` simulator; additional devices can be installed
as plugins (see :ref:`plugins` for more details). Note that the choice of a device significantly
determines the speed of your computation.

Creating a quantum node
^^^^^^^^^^^^^^^^^^^^^^^

Together, a quantum function and a device are used to create a *quantum node* or
:class:`QNode` object, which wraps the quantum function and binds it to the device.

A `QNode` can be explicitly created as follows:

.. code-block:: python

    qnode = qml.QNode(my_quantum_function, dev)

The `QNode` can be used to compute the result of a quantum circuit as if it was a standard Python
function. It takes the same arguments as the original quantum function:

>>> qnode(np.pi/4, 0.7)
0.7648421872844883

The QNode decorator
^^^^^^^^^^^^^^^^^^^

.. _intro_vcirc_decorator:

A more convenient - and in fact the recommended - way for creating `QNodes` is the provided
quantum node decorator. This decorator converts a quantum function containing PennyLane quantum
operations to a :mod:`QNode <pennylane.qnode>` that will run on a quantum device.

.. note::
    The decorator completely replaces the Python-based quantum function with
    a :mod:`QNode <pennylane.qnode>` of the same name - as such, the original
    function is no longer accessible (but is accessible via the :attr:`~.QNode.func` attribute).

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def qfunc(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval(qml.PauliZ(0))

    result = qfunc(0.543)




Quantum operations
------------------

.. _intro_vcirc_ops:

.. currentmodule:: pennylane.ops

PennyLane supports a wide variety of quantum operations - such as gates, state preparations and measurement
observables. These operations can be used exclusively in quantum functions.

Qubit operations
^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.ops.qubit

Gates
*****

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


State preparation
*****************

.. autosummary::
    BasisState
    QubitStateVector


Observables
***********

.. autosummary::
    Hadamard
    PauliX
    PauliY
    PauliZ
    Hermitian

Continuous-variable operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.ops.cv

Gates
*****

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


State preparation
*****************

.. autosummary::
    CoherentState
    SqueezedState
    DisplacedSqueezedState
    ThermalState
    GaussianState
    FockState
    FockStateVector
    FockDensityMatrix
    CatState

Observables
***********

.. autosummary::
    NumberOperator
    X
    P
    QuadOperator
    PolyXP
    FockStateProjector



Shared operations
^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.ops

The only operation shared by both qubit and continouous-variable architectures is the Identity.

.. autosummary::
    Identity


Quantum measurements
--------------------
.. _intro_vcirc_measure:

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results:

.. autosummary::
    expval
    var
    sample


Templates
---------

.. _intro_vcirc_templates:

PennyLane provides a growing library of pre-coded templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models. In the
quantum machine learning literature, such architectures are commonly known as an
**ansatz**.

.. note::

    Templates are constructed out of **structured combinations** of the :mod:`quantum operations <pennylane.ops>`
    provided by PennyLane. This means that **template functions can only be used within a
    valid** :mod:`pennylane.qnode`.

PennyLane conceptually distinguishes two types of templates, :ref:`layer architectures <intro_vcirc_temp_layer>`
and :ref:`input embeddings <intro_vcirc_temp_emb>` .
Most templates are complemented by functions that provide an array of
random :ref:`initial parameters <intro_vcirc_temp_params>` .

Layer templates
^^^^^^^^^^^^^^^

.. _intro_vcirc_temp_layer:

.. currentmodule:: pennylane.templates.layers

Layer architectures, found in :mod:`pennylane.templates.layers`, define sequences of gates that are
repeated like the layers in a neural network. They usually contain only trainable parameters.

The following layer templates are available:

.. autosummary::
    StronglyEntanglingLayers
    RandomLayers

Embedding templates
^^^^^^^^^^^^^^^^^^^
.. _intro_vcirc_temp_emb:

.. currentmodule:: pennylane.templates.embeddings

Embeddings, found in :mod:`pennylane.templates.embeddings`, encode input features into the quantum state
of the circuit. Hence, they take a feature vector as an argument. These embeddings can also depend on
trainable parameters, in which case the embedding is learnable.

The following embedding templates are available:


.. autosummary::
    AmplitudeEmbedding
    BasisEmbedding
    AngleEmbedding
    SqueezingEmbedding
    DisplacementEmbedding

Parameter initializations
^^^^^^^^^^^^^^^^^^^^^^^^^
.. _intro_vcirc_temp_params:

.. currentmodule:: pennylane.init

Each trainable template has a dedicated function in :mod:`pennylane.init` which generates a list of
**randomly initialized** arrays for the trainable **parameters**.

Qubit architectures
~~~~~~~~~~~~~~~~~~~

Strongly entangling circuit
***************************
.. autosummary::
    strong_ent_layers_uniform
    strong_ent_layers_normal
    strong_ent_layer_uniform
    strong_ent_layer_normal

Random circuit
**************
.. autosummary::
    random_layers_uniform
    random_layers_normal
    random_layer_uniform
    random_layer_normal

Continuous-variable architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous-variable quantum neural network
******************************************
.. autosummary::
    cvqnn_layers_uniform
    cvqnn_layers_normal
    cvqnn_layer_uniform
    cvqnn_layer_normal

Interferometer
**************
.. autosummary::
    interferometer_uniform
    interferometer_normal


