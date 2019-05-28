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

Templates are used exactly as one would use a quantum gate, only that they invoke a sequence of quantum gates instead.

.. note::

    Templates are constructed out of **structured combinations** of the :mod:`quantum operations <pennylane.ops>` provided by PennyLane. This means that **template functions can only be used within a valid** :mod:`pennylane.qnode`.

PennyLane conceptually distinguishes two types of templates, **layer architectures** and **input embeddings**:

* Layer architectures, found in :mod:`pennylane.templates.layers`, define sequences of gates that are repeated
  like the layers in a neural network. They usually contain only trainable parameters.

* Embeddings, found in :mod:`pennylane.templates.embeddings`, encode input features into the quantum state of the
  circuit. Hence, they take a feature vector as an argument. These embeddings can also depend on trainable parameters, in which case the embedding is learnable.

The following templates of each type are available:

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 3

    templates/layers
    templates/embeddings


Creating initial parameters
---------------------------

Each trainable template has a dedicated function in :mod:`pennylane.init` which generates a list of
**randomly initialized** arrays for the trainable **parameters**. To illustrate how these can be used, let us use a hypothetical parameter initialization function ``my_init_fun()`` and its corresponding hypothetical template ``MyTemplate()``, which takes three parameter arrays. The two can be combined in the following two ways: 

1. Dereference the list when feeding it into the template:

.. code-block:: python

   par_list = my_init_fun(...)
   ...
   MyTemplate(*par_list, ...)


2. Unpack the list as the init function is called:

.. code-block:: python

   a, b, c = my_init_fun(...)
   ...
   MyTemplate(a, b, c, ...)

The following parameter initialization methods are available:


.. toctree::
    :maxdepth: 3

    templates/init_parameters

Examples
--------

You can construct a circuit-centric quantum
classifier with the architecture from :cite:`schuld2018circuit` on an arbitrary
number of wires and with an arbitrary number of layers by using the
template :func:`~.StronglyEntanglingLayers` in the following way:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates.layers import StronglyEntanglingLayers
    from pennylane.init import strong_ent_layers_normal

    from pennylane import numpy as np

    n_wires = 4
    n_layers = 3

    dev = qml.device('default.qubit', wires=n_wires)


    @qml.qnode(dev)
    def circuit(pars, x=None):
        qml.BasisState(x, wires=range(n_wires))
        StronglyEntanglingLayers(*pars, wires=range(n_wires))
        return qml.expval.PauliZ(0)


    pars = strong_ent_layers_normal(n_layers=n_layers, n_wires=n_wires, mean=0, std=0.1)
    print(circuit(pars, x=np.array(np.random.randint(0, 1, n_wires))))

.. note::

    ``pars`` is a list of parameter arrays. In the case of the strongly entangling template, the list contains
    exactly one such parameter array of shape ``(n_layers, n_wires, 3)``. One could alternatively create this
    list of arrays by hand, replacing second-to-last line with

    .. code-block:: python

        pars = [np.random.normal(loc=0, scale=0.1, size=(n_layers, n_wires, 3))]

.. note::

    Most parameter generating methods have a 'normal' and a 'uniform' version, sampling the angle parameters
    of rotation gates either from a normal or uniform distribution.

Templates can contain each other. An example is the handy :class:`~.Interferometer` template. It constructs
arbitrary interferometers in terms of elementary :class:`~.Beamsplitter` operations, by providing lists of
transmittivity and phase angles. A :func:`~.CVNeuralNetLayer` template — implementing the continuous-variable neural
network architecture from :cite:`killoran2018continuous` — contains two such interferometers. But it can also
be used (and optimized) independently:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates.layers import Interferometer
    from pennylane import numpy as np

    n_wires = 4
    n_params = int(n_wires * (n_wires - 1) / 2)

    dev = qml.device('default.gaussian', wires=n_wires)

    # initial parameters
    r = np.random.rand(n_wires, 2)
    theta = np.random.uniform(0, 2 * np.pi, n_params)
    phi = np.random.uniform(0, 2 * np.pi, n_params)
    varphi = np.random.uniform(0, 2 * np.pi, n_wires)


    @qml.qnode(dev)
    def circuit(theta, phi, varphi):
        for w in range(n_wires):
            qml.Squeezing(r[w][0], r[w][1], wires=w)
        Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(n_wires))
        return [qml.expval.MeanPhoton(wires=w) for w in range(n_wires)]


    j = qml.jacobian(circuit, 0)
    print(j(theta, phi, varphi))

Once more, instead of generating the arrays for ``theta``, ``phi`` and ``varphi`` by hand, one can use
the :func:`~.interferometer_uniform` function.


.. code-block:: python

    from pennylane.init import interferometer_uniform

    import pennylane as qml
    from pennylane.templates.layers import Interferometer
    from pennylane import numpy as np

    n_wires = 4
    n_params = int(n_wires * (n_wires - 1) / 2)

    dev = qml.device('default.gaussian', wires=n_wires)

    # initial parameters
    r = np.random.rand(n_wires, 2)
    pars = interferometer_uniform(n_wires)


    @qml.qnode(dev)
    def circuit(theta, phi, varphi):
        for w in range(n_wires):
            qml.Squeezing(r[w][0], r[w][1], wires=w)
        Interferometer(theta=theta, phi=phi, varphi=varphi, wires=range(n_wires))
        return [qml.expval.MeanPhoton(wires=w) for w in range(n_wires)]


    j = qml.jacobian(circuit, 0)
    print(j(*pars))


By growing this library of templates, PennyLane allows easy access to variational models discussed
in the quantum machine learning literature.

.. automodule:: pennylane.templates
   :members:
   :private-members:
   :inherited-members:
