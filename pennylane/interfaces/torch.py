# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
.. _torch_qnode:

PyTorch interface
*****************

**Module name:** :mod:`pennylane.interfaces.torch`

.. currentmodule:: pennylane.interfaces.torch

.. warning:: This interface is **experimental**


Using the PyTorch interface
---------------------------

.. note::

    To use the PyTorch interface in PennyLane, you must first
    `install PyTorch <https://pytorch.org/get-started/locally/#start-locally>`_.

Using the PyTorch interface is easy in PennyLane --- let's consider a few ways
it can be done.

Via the QNode decorator
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`QNode decorator <qnode_decorator>` is the recommended way for creating QNodes
in PennyLane. The only change required to construct a PyTorch-capable QNode is to
specify the ``interface='torch'`` keyword argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval.PauliZ(0), qml.expval.Hadamard(1)

The QNode ``circuit()`` is now a PyTorch-capable QNode, accepting ``torch.tensor`` objects
as input, and returning ``torch.tensor`` objects. Subclassing from ``torch.autograd.Function``,
it can now be used like any other PyTorch function:

>>> phi = torch.tensor([0.5, 0.1])
>>> theta = torch.tensor(0.2)
>>> circuit(phi, theta)
tensor([0.8776, 0.6880], dtype=torch.float64)

Via the QNode class
^^^^^^^^^^^^^^^^^^^

Sometimes, it is more convenient to instantiate a :class:`~.QNode` object directly, for example,
if you would like to reuse the same quantum function across multiple devices, or even
using different classical interfaces:

.. code-block:: python

    dev1 = qml.device('default.qubit', wires=2)
    dev2 = qml.device('forest.wavefunction', wires=2)

    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval.PauliZ(0), qml.expval.Hadamard(1)

    qnode1 = qml.QNode(circuit, dev1)
    qnode2 = qml.QNode(circuit, dev2, interface='torch')

As with the QNode decorator, we simply pass the ``interface`` keyword argument
to set the classical interface.

We can also convert a NumPy-interfacing QNode to a PyTorch-interfacing QNode by
using the :meth:`~.QNode.to_torch` method:

>>> qnode1 = qnode1.to_torch()
>>> qnode1
<function TorchQNode.<locals>._TorchQNode.apply>

Internally, the :meth:`~.QNode.to_torch` method uses the :func:`~.TorchQNode` function
to do the conversion.


Quantum gradients using PyTorch
-------------------------------

Since a PyTorch-interfacing QNode acts like any other ``torch.autograd.Function``,
the standard method used to calculate gradients with PyTorch can be used.

For example:

.. code-block:: python

    import pennylane as qml
    import torch
    from torch.autograd import Variable

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RY(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PhaseShift(theta, wires=0)
        return qml.expval.PauliZ(0)

    phi = Variable(torch.tensor([0.5, 0.1]), requires_grad=True)
    theta = Variable(torch.tensor(0.2), requires_grad=True)
    result = circuit(phi, theta)

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
or your own custom PyTorch optimizer. **The** :ref:`PennyLane optimizers <optimization_methods>`
**cannot be used with the Torch interface, only the** :ref:`numpy_qnode`.

For example, to optimize a Torch-interfacing QNode (below) such that the weights ``x``
result in an expectation value of 0.5:

.. code-block:: python

    import torch
    from torch.autograd import Variable
    import pennylane as qml

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit(phi, theta):
        qml.RX(phi[0], wires=0)
        qml.RZ(phi[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(theta, wires=0)
        return qml.expval.PauliZ(0)

    def cost(phi, theta):
        return torch.abs(circuit(phi, theta) - 0.5)**2

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

The final weights and circuit value:

>>> phi_final, theta_final = opt.param_groups[0]['params']
>>> phi_final, theta_final
(tensor([0.7345, 0.0120], requires_grad=True), tensor(0.8316, requires_grad=True))
>>> circuit(phi_final, theta_final)
tensor(0.5000, dtype=torch.float64, grad_fn=<_TorchQNodeBackward>)

.. note::

    For more advanced PyTorch models, Torch-interfacing QNodes can be used to construct
    layers in custom PyTorch modules (``torch.nn.Module``).

    See https://pytorch.org/docs/stable/notes/extending.html#adding-a-module for more details.


Code details
^^^^^^^^^^^^
"""
# pylint: disable=redefined-outer-name
import numpy as np
import torch

from pennylane.utils import unflatten


def TorchQNode(qnode):
    """Function that accepts a :class:`~.QNode`, and returns a PyTorch-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode

    Returns:
        torch.autograd.Function: the QNode as a PyTorch autograd function
    """

    class _TorchQNode(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *input_):
            # detach all input tensors, convert to NumPy array
            args = [i.detach().numpy() for i in input_]
            # if NumPy array is scalar, convert to a Python float
            args = [i.tolist() if not i.shape else i for i in args]

            # evaluate the QNode
            res = qnode(*args)

            if not isinstance(res, np.ndarray):
                # scalar result, cast to NumPy scalar
                res = np.array(res)

            ctx.save_for_backward(*input_)
            return torch.from_numpy(res)

        @staticmethod
        def backward(ctx, grad_output):
            # detach all saved input tensors, convert to NumPy array
            args = [i.detach().numpy() for i in ctx.saved_tensors]
            # evaluate the Jacobian matrix of the QNode
            jacobian = qnode.jacobian(args)

            grad_output_np = grad_output.detach().numpy()

            # perform the vector-Jacobian product
            if not grad_output_np.shape:
                temp = grad_output_np * jacobian
            else:
                temp = grad_output_np.T @ jacobian

            # restore the nested structure of the input args
            temp = unflatten(temp.flat, args)
            # convert the result to torch tensors, matching
            # the type of the input tensors
            grad_input = [torch.as_tensor(torch.from_numpy(i), dtype=j.dtype) for i, j in zip(temp, ctx.saved_tensors)]
            return tuple(grad_input)

    return _TorchQNode.apply
