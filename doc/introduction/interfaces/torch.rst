.. _torch_interf:

PyTorch interface
==================

In order to use PennyLane in combination with PyTorch, we have to generate PyTorch-compatible
quantum nodes. Such a QNode can be created explicitly using the ``interface='torch'`` keyword in
the QNode decorator or QNode class constructor.

.. note::

    To use the PyTorch interface in PennyLane, you must first
    `install PyTorch <https://pytorch.org/get-started/locally/#start-locally>`_
    and import it together with PennyLane via:

    .. code::

        import pennylane as qml
        import torch

Using the PyTorch interface is easy in PennyLane --- let's consider a few ways
it can be done.


.. _torch_interf_keyword:

Construction via keyword
------------------------

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating
:class:`QNode <pennylane.QNode>` objects in PennyLane. The only change required to construct a PyTorch-capable
QNode is to specify the ``interface='torch'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit()`` is now a PyTorch-capable QNode, accepting ``torch.tensor`` objects as
input, and returning ``torch.tensor`` objects. Subclassing from ``torch.autograd.Function``,
it can now be used like any other PyTorch function:

>>> phi = torch.tensor([0.5, 0.1])
>>> theta = torch.tensor(0.2)
>>> circuit(phi, theta)
tensor([0.8776, 0.6880], dtype=torch.float64)

PyTorch-capable QNodes can also be created using the
:ref:`QNode class constructor <intro_vcirc_qnode>`:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('default.mixed', wires=2)

    def circuit1(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

    qnode1 = qml.QNode(circuit1, dev1)
    qnode2 = qml.QNode(circuit1, dev2, interface='torch')

``qnode1()`` is a default NumPy-interfacing QNode, while ``qnode2()`` is a PyTorch-capable
QNode:

>>> qnode2(phi, theta)
tensor([0.8776, 0.6880], dtype=torch.float64)

.. _pytorch_qgrad:

Quantum gradients using PyTorch
-------------------------------

Since a PyTorch-interfacing QNode acts like any other ``torch.autograd.Function``,
the standard method used to calculate gradients with PyTorch can be used.

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit3(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = torch.tensor([0.5, 0.1], requires_grad=True)
    theta = torch.tensor(0.2, requires_grad=True)
    result = circuit3(phi, theta)

Now, performing the backpropagation and accumulating the gradients:

>>> result.backward()
>>> phi.grad
tensor([-0.4794,  0.0000])
>>> theta.grad
tensor(-5.5511e-17)

To include non-differentiable data arguments, simply set ``requires_grad=False``:

.. code-block:: python

    @qml.qnode(dev, interface='torch')
    def circuit3(weights, data):
        qml.AmplitudeEmbedding(data, normalize=True, wires=[0, 1])
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(weights[2], wires=0)
        return qml.expval(qml.PauliZ(0))

Here, ``data`` is non-trainable embedded data, so should be marked as non-differentiable:

>>> weights = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
>>> data = torch.tensor(np.random.random([4]), requires_grad=False)
>>> result = circuit3(weights, data)
>>> result.backward()
>>> data.grad is None
True
>>> weights.grad
tensor([3.6317e-02, 0.0000e+00, 5.5511e-17])


.. _pytorch_optimize:

Optimization using PyTorch
--------------------------

To optimize your hybrid classical-quantum model using the Torch interface,
you **must** make use of the `PyTorch provided optimizers <https://pytorch.org/docs/stable/optim.html>`_,
or your own custom PyTorch optimizer. **The** :ref:`PennyLane optimizers <intro_ref_opt>`
**cannot be used with the Torch interface**.

For example, to optimize a Torch-interfacing QNode (below) such that the weights ``x``
result in an expectation value of 0.5 we can do the following:

.. code-block:: python

    import torch
    import pennylane as qml

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit4(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RZ(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    def cost(phi, theta):
        return torch.abs(circuit4(phi, theta) - 0.5)**2

    phi = torch.tensor([0.011, 0.012], requires_grad=True)
    theta = torch.tensor(0.05, requires_grad=True)

    opt = torch.optim.Adam([phi, theta], lr = 0.1)

    steps = 200

    def closure():
        opt.zero_grad()
        loss = cost(phi, theta)
        loss.backward()
        return loss

    for i in range(steps):
        opt.step(closure)

The final weights and circuit value are:

>>> phi_final, theta_final = opt.param_groups[0]['params']
>>> phi_final
tensor([7.3449e-01, 3.1554e-04], requires_grad=True)
>>> theta_final
tensor(0.8316, requires_grad=True)
>>> circuit4(phi_final, theta_final)
tensor(0.5000, dtype=torch.float64, grad_fn=<SqueezeBackward0>)

.. note::

    For more advanced PyTorch models, Torch-interfacing QNodes can be used to construct
    layers in custom PyTorch modules (``torch.nn.Module``).

    See https://pytorch.org/docs/stable/notes/extending.html#adding-a-module for more details.

GPU and CUDA support
--------------------

This section only applies to users who have installed torch with CUDA support.
If you are not sure if you have CUDA support, you can check with the following function:

>>> torch.cuda.is_available()
True

If at least one input parameter is on a CUDA device and you are using backpropogation,
the execution will occur on the CUDA device. For systems with a high number of wires, CUDA
execution can be much faster. For lower wire count, the overhead of moving everything to
the GPU will dominate performance; for less than 15 wires, the GPU will probably be slower.

.. code-block:: python

    n_wires = 20
    n_layers = 10

    dev = qml.device('default.qubit', wires=n_wires)

    params_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    params = torch.rand(params_shape)

    @qml.qnode(dev, interface='torch', diff_method="backprop")
    def circuit_cuda(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

>>> import timeit
>>> timeit.timeit("circuit_cuda(params)", globals=globals(), number=5))
10.110647433029953
>>> params = params.to(device=torch.device('cuda'))
>>> timeit.timeit("circuit_cuda(params)", globals=globals(), number=5)
2.297812332981266

Torch.nn integration
--------------------

Once you have a Torch-compaible QNode, it is easy to convert this into a ``torch.nn`` layer. To help
automate this process, PennyLane also provides a :class:`~.qnn.TorchLayer` class to easily
convert a QNode to a ``torch.nn`` layer. Please see the corresponding :class:`~.qnn.TorchLayer`
documentation for more details and examples.
