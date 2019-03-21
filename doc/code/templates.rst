.. _operations:

Templates
=========

**Module name:** :mod:`pennylane.templates`

.. currentmodule:: pennylane.templates

This module provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models. In the 
quantum machine learning literature, such architectures are commonly known as an
**ansatz**.

PennyLane conceptually distinguishes two types of templates, layer architectures and input embeddings: 

* Layer architectures in :mod:`pennylane.templates.layers` define blocks of gates that are repeated like the layers in a neural network. They usually contain only trainable parameters.
* Embeddings in :mod:`pennylane.templates.embeddings` encode input features into the quantum state of the circuit. These embeddings can also be learnable, or depend on trainable parameters.

Each template has a utilities function in :mod:`pennylane.templates.utils` which generates a randomly sampled array for the trainable parameters fed into the template.

.. note::

    The templates below are constructed out of **structured combinations**
    of the :mod:`quantum operations <pennylane.ops>` provided by PennyLane.
    As a result, you should follow all the rules of quantum operations
    when you use templates. For example **template functions can only be
    used within a valid** :mod:`pennylane.qnode`.

Examples
--------

For example, you can construct a circuit-centric quantum
classifier with the architecture from :cite:`schuld2018circuit` on an arbitrary
number of wires and with an arbitrary number of blocks by using the
template :class:`StronglyEntanglingCircuit` in the following way:

.. code-block:: python

    import pennylane as qml
    from pennylane.template.layers import StronglyEntanglingLayers
    from pennylane import numpy as np
    num_wires = 4
    num_blocks = 3
    dev = qml.device('default.qubit', wires=num_wires)
    @qml.qnode(dev)
    def circuit(weights, x=None):
        qml.BasisState(x, wires=range(num_wires))
        StronglyEntanglingLayers(weights, periodic=True, wires=range(num_wires))
        return qml.expval.PauliZ(0)
    weights = np.random.randn(num_blocks, num_wires, 3)
    print(circuit(weights, x=np.array(np.random.randint(0, 1, num_wires))))

The handy :func:`Interferometer` function can be used to construct arbitrary
interferometers in terms of elementary :class:`~.Beamsplitter` operations,
by providing lists of transmittivity and phase angles. PennyLane can
then be used to easily differentiate and optimize these
parameters:

.. code-block:: python

    import pennylane as qml
    from pennylane.template import Interferometer
    from pennylane import numpy as np
    num_wires = 4
    dev = qml.device('default.gaussian', wires=num_wires)
    num_params = int(num_wires*(num_wires-1)/2)
    # initial parameters
    r = np.random.rand(num_wires, 2)
    theta = np.random.uniform(0, 2*np.pi, num_params)
    phi = np.random.uniform(0, 2*np.pi, num_params)
    @qml.qnode(dev)
    def circuit(theta, phi):
        for w in range(num_wires):
            qml.Squeezing(r[w][0], r[w][1], wires=w)
        Interferometer(theta=theta, phi=phi, wires=range(num_wires))
        return [qml.expval.MeanPhoton(wires=w) for w in range(num_wires)]
    print(qml.jacobian(circuit, 0)(theta, phi))

The function :func:`CVNeuralNet` implements the continuous-variable neural network architecture
from :cite:`killoran2018continuous`. Provided with a suitable array of weights, such neural
networks can be easily constructed and trained with PennyLane.



.. automodule:: pennylane.templates
   :members:
   :private-members:
   :inherited-members:
