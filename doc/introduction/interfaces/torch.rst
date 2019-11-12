.. _torch_interf:

PyTorch interface
=================

In order to use PennyLane in combination with PyTorch, we have to generate PyTorch-compatible
quantum nodes. A basic :class:`QNode <pennylane.qnode.QNode>` can be translated into a quantum node that interfaces
with PyTorch, either by using the ``interface='torch'`` flag in the QNode Decorator, or
by calling the :meth:`QNode.to_torch <pennylane.QNode.to_torch>` method. Internally, the translation is executed by
the :func:`TorchQNode <pennylane.interfaces.torch.TorchQNode>` function that returns the new quantum node object.

.. note::

    To use the PyTorch interface in PennyLane, you must first
    `install PyTorch <https://pytorch.org/get-started/locally/#start-locally>`_
    and import it together with PennyLane via:

    .. code::

        import pennylane as qml
        import torch


Construction via the decorator
------------------------------

The :ref:`QNode decorator <intro_vcirc_decorator>` is the recommended way for creating
a PyTorch-capable QNode in PennyLane. Simply specify the ``interface='torch'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit1(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

The QNode ``circuit1()`` is now a PyTorch-capable QNode, accepting ``torch.tensor`` objects
as input, and returning ``torch.tensor`` objects. Subclassing from ``torch.autograd.Function``,
it can now be used like any other PyTorch function:

>>> phi = torch.tensor([0.5, 0.1])
>>> theta = torch.tensor(0.2)
>>> circuit1(phi, theta)
tensor([0.8776, 0.6880], dtype=torch.float64)

Construction from a NumPy QNode
-------------------------------

Sometimes, it is more convenient to instantiate a :class:`~.QNode` object directly, for example,
if you would like to reuse the same quantum function across multiple devices, or even
use different classical interfaces:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('forest.wavefunction', wires=2)

    def circuit2(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.Hadamard(1))

    qnode1 = qml.QNode(circuit2, dev1)
    qnode2 = qml.QNode(circuit2, dev2)

We can convert the default NumPy-interfacing QNode to a PyTorch-interfacing QNode by
using the :meth:`~.QNode.to_torch` method:

>>> qnode1_torch = qnode1.to_torch()
>>> qnode1_torch
<QNode: device='default.qubit', func=circuit, wires=2, interface=PyTorch>

Internally, the :meth:`QNode.to_torch <qnode.QNode.to_torch>` method uses the
:func:`TorchQNode <interfaces.torch.TorchQNode>` function to do the conversion.

Quantum gradients using PyTorch
-------------------------------

Since a PyTorch-interfacing QNode acts like any other ``torch.autograd.Function``,
the standard method used to calculate gradients with PyTorch can be used.

For example:

.. code-block:: python

    from torch.autograd import Variable

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit3(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    phi = Variable(torch.tensor([0.5, 0.1]), requires_grad=True)
    theta = Variable(torch.tensor(0.2), requires_grad=True)
    result = circuit3(phi, theta)

Now, performing the backpropagation and accumulating the gradients:

>>> result.backward()
>>> phi.grad
tensor([-0.4794,  0.0000])
>>> theta.grad
tensor(-5.5511e-17)

.. _pytorch_optimize:

Optimization using PyTorch
--------------------------

To optimize your hybrid classical-quantum model using the Torch interface,
you **must** make use of the `PyTorch provided optimizers <https://pytorch.org/docs/stable/optim.html>`_,
or your own custom PyTorch optimizer. **The** :ref:`PennyLane optimizers <intro_ref_opt>`
**cannot be used with the Torch interface**.

For example, to optimize a Torch-interfacing QNode (below) such that the weights ``x``
result in an expectation value of 0.5, with the classical nodes processed on a GPU,
we can do the following:

.. code-block:: python

    import torch
    from torch.autograd import Variable
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

    phi = Variable(torch.tensor([0.011, 0.012]), requires_grad=True)
    theta = Variable(torch.tensor(0.05), requires_grad=True)

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
>>> phi_final, theta_final
(tensor([0.7345, 0.0120], device='cuda:0', requires_grad=True), tensor(0.8316, device='cuda:0', requires_grad=True))
>>> circuit(phi_final, theta_final)
tensor(0.5000, device='cuda:0', dtype=torch.float64, grad_fn=<_TorchQNodeBackward>)

.. note::

    For more advanced PyTorch models, Torch-interfacing QNodes can be used to construct
    layers in custom PyTorch modules (``torch.nn.Module``).

    See https://pytorch.org/docs/stable/notes/extending.html#adding-a-module for more details.



