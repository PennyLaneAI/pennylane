 .. role:: html(raw)
   :format: html


.. _intro_qnodes:

Variational Circuits
====================

In the following we will see how the concept of a :ref:`variational quantum circuit <varcirc>`, the
heart piece of hybrid quantum-classical optimization, is implemented in PennyLane.

New PennyLane users learn how to:

.. image:: _static/intro_qnodes.png
    :align: right
    :width: 180px
    :target: javascript:void(0);

- Construct quantum circuits via **quantum functions**
- Define **computational devices**
- Combine quantum functions and devices to **quantum nodes**
- Conveniently create quantum nodes using the quantum node **decorator**
- Find out more about **interfaces** to use with quantum nodes

The theoretical background
of variational circuits and hybrid optimization is found in the :ref:`Key Concepts <key_concepts>`.

Creating quantum nodes
----------------------

Quantum functions
^^^^^^^^^^^^^^^^^

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


Using QNodes
^^^^^^^^^^^^

Quantum nodes are typically used in :ref:`hybrid computations <hybrid_computation>`. This means
that results of `QNodes` are further processed in classical functions, and that results from
classical functions are fed into `QNodes`. The framework in which the `classical parts` of the
hybrid computation are written is the *interface* with which PennyLane is used.

In the above introduction to quantum nodes, we implicitly already used the default interface
- the :ref:`NumPy interface <numpy_interface>`.
NumPy-interfacing quantum nodes take NumPy datastructures, such as floats and arrays, and return
similar data structures. They can be optimized using NumPy-based :ref:`optimization methods <optimize>`.
Other PennyLane interfaces are :ref:`PyTorch <torch_interf>` and :ref:`TensorFlow's Eager
mode <tf_interf>`.


Supported operations
--------------------


PennyLane supports a wide variety of quantum operations - such as gates, state preparations and measurement
observables.

.. raw:: html
    <style>
    div.topic.contents > ul {
        max-height: 100px;
    }
    </style>
.. rst-class:: contents local topic
.. toctree::
    :maxdepth: 2
    ops/qubit
    ops/cv


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

The only operation shared by both qubit and continouous-variable architectures is the Identity.

.. autosummary::
    Identity


Supported measurements
----------------------

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results:

.. autosummary::
    expval
    var
    sample


Pre-coded templates
-------------------

PennyLane provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models. In the
quantum machine learning literature, such architectures are commonly known as an
**ansatz**.

.. note::

    Templates are constructed out of **structured combinations** of the :mod:`quantum operations <pennylane.ops>`
    provided by PennyLane. This means that **template functions can only be used within a
    valid** :mod:`pennylane.qnode`.

PennyLane conceptually distinguishes two types of templates, **layer architectures** and **input embeddings**.
Most templates are complemented by functions that provide an array of random **initial parameters**.

Layer templates
^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.templates.layers

Layer architectures, found in :mod:`pennylane.templates.layers`, define sequences of gates that are
repeated like the layers in a neural network. They usually contain only trainable parameters.

The following layer templates are available:

.. autosummary::
    StronglyEntanglingLayers
    RandomLayers

Embedding templates
^^^^^^^^^^^^^^^^^^^

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


