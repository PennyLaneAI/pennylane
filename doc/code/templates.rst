.. _templates:

Templates
=========

**Module name:** :mod:`pennylane.templates`

.. currentmodule:: pennylane.templates

This module provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models. In the
quantum machine learning literature, such architectures are commonly known as an
**ansatz**.

PennyLane conceptually distinguishes two types of templates, **layer architectures** and **input embeddings**:

* Layer architectures, found in :mod:`pennylane.templates.layers`, define blocks of gates that are repeated like the layers in a neural network. They usually contain only trainable parameters.

* Embeddings, found in :mod:`pennylane.templates.embeddings`, encode input features into the quantum state of the circuit. These embeddings can also depend on trainable parameters, in which case the embedding is learnable.

Each trainable template has a dedicated function in :mod:`pennylane.templates.parameters` which generates a list of **randomly initialized** arrays for the trainable **parameters**. The entries of the list contain valid positional arguments for the template, allowing for the syntax ``MyTemplate(*par_list)``.

.. note::

    Templates are constructed out of **structured combinations**
    of the :mod:`quantum operations <pennylane.ops>` provided by PennyLane.
    As a result, you should follow all the rules of quantum operations
    when you use templates. For example **template functions can only be
    used within a valid** :mod:`pennylane.qnode`.


Summary
-------

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 3

    templates/layers
    templates/embeddings
    templates/parameters

Examples
--------

You can construct a circuit-centric quantum
classifier with the architecture from :cite:`schuld2018circuit` on an arbitrary
number of wires and with an arbitrary number of blocks by using the
template :func:`~.StronglyEntanglingLayers` in the following way:

.. code-block:: python

	import pennylane as qml
	from pennylane.templates.layers import StronglyEntanglingLayers
	from pennylane.templates.parameters import parameters_stronglyentangling_layers

	from pennylane import numpy as np

	num_wires = 4
	num_blocks = 3

	dev = qml.device('default.qubit', wires=num_wires)


	@qml.qnode(dev)
	def circuit(pars, x=None):
	    qml.BasisState(x, wires=range(num_wires))
	    StronglyEntanglingLayers(*pars, periodic=True, wires=range(num_wires))
	    return qml.expval.PauliZ(0)


	pars = parameters_stronglyentangling_layers(n_layers= 2, n_wires=4)
	print(circuit(pars, x=np.array(np.random.randint(0, 1, num_wires))))

.. note::

	``weights`` is a list of parameter arrays. In the case of the strongly entangling template, the list contains exactly one such parameter array of shape ``(num_blocks, num_wires, 3)``. One could alternatively create this list of an array by hand, replacing second-to-last line with ``weights = [np.random.randn(num_blocks, num_wires, 3)]``.

Templates can contain each other. An example is the handy :class:`~.Interferometer` function. It constructs arbitrary interferometers in terms of elementary :class:`~.Beamsplitter` operations, by providing lists of transmittivity and phase angles. A :func:`~.CVNeuralNetLayers` - implementing the continuous-variable neural network architecture from :cite:`killoran2018continuous` - contains two such interferometers. But it can also be used (and optimized) independently:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates.layers import Interferometer
    from pennylane import numpy as np
    
    num_wires = 4
    num_params = int(num_wires * (num_wires - 1) / 2)
    
    dev = qml.device('default.gaussian', wires=num_wires)
    
    # initial parameters
    r = np.random.rand(num_wires, 2)
    theta = np.random.uniform(0, 2 * np.pi, num_params)
    phi = np.random.uniform(0, 2 * np.pi, num_params)
    varphi = np.random.uniform(0, 2 * np.pi, num_wires)
    
    
    @qml.qnode(dev)
    def circuit(theta, phi, varphi):
        for w in range(num_wires):
            qml.Squeezing(r[w][0], r[w][1], wires=w)
        Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(num_wires))
        return [qml.expval.MeanPhoton(wires=w) for w in range(num_wires)]
    
    j = qml.jacobian(circuit, 0)
    print(j(theta, phi, varphi))

Instead of generating the arrays for ``theta``, ``phi`` and ``varphi`` by hand, one can use the :func:`pennylane.templates.parameters.parameters_interferometer()` function. 


.. code-block:: python
    
    from pennylane.templates.parameters import parameters_interferometer

    ...
    
    # initial parameters
    r = np.random.rand(num_wires, 2)
    pars = parameters_interferometer(num_wires)

    ...

    j = qml.jacobian(circuit, 0)
    print(j(*pars))

By growing this library of templates, PennyLane allows easy access to variational models discussed in the quantum machine learning literature. 

.. automodule:: pennylane.templates
   :members:
   :private-members:
   :inherited-members:
