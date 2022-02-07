# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes and functions for integrating QNodes with the Torch Module
API."""
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Optional

try:
    import torch
    from torch.nn import Module

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the TorchLayer.
    from unittest.mock import Mock

    Module = Mock
    TORCH_IMPORTED = False


class TorchLayer(Module):
    r"""Converts a :func:`~.QNode` to a Torch layer.

    The result can be used within the ``torch.nn``
    `Sequential <https://pytorch.org/docs/stable/nn.html#sequential>`__ or
    `Module <https://pytorch.org/docs/stable/nn.html#module>`__ classes for
    creating quantum and hybrid models.

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Torch layer
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding shapes
        init_method (callable): a `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__
            function for initializing the QNode weights. If not specified, weights are randomly
            initialized using the uniform distribution over :math:`[0, 2 \pi]`.

    **Example**

    First let's define the QNode that we want to convert into a Torch layer:

    .. code-block:: python

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights_0, weight_1):
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)
            qml.Rot(*weights_0, wires=0)
            qml.RY(weight_1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    The signature of the QNode **must** contain an ``inputs`` named argument for input data,
    with all other arguments to be treated as internal weights. We can then convert to a Torch
    layer with:

    >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
    >>> qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    The internal weights of the QNode are automatically initialized within the
    :class:`~.TorchLayer` and must have their shapes specified in a ``weight_shapes`` dictionary.
    It is then easy to combine with other neural network layers from the
    `torch.nn <https://pytorch.org/docs/stable/nn.html>`__ module and create a hybrid:

    >>> clayer = torch.nn.Linear(2, 2)
    >>> model = torch.nn.Sequential(qlayer, clayer)

    .. UsageDetails::

        **QNode signature**

        The QNode must have a signature that satisfies the following conditions:

        - Contain an ``inputs`` named argument for input data.
        - All other arguments must accept an array or tensor and are treated as internal
          weights of the QNode.
        - All other arguments must have no default value.
        - The ``inputs`` argument is permitted to have a default value provided the gradient with
          respect to ``inputs`` is not required.
        - There cannot be a variable number of positional or keyword arguments, e.g., no ``*args``
          or ``**kwargs`` present in the signature.

        **Initializing weights**

        The optional ``init_method`` argument of :class:`~.TorchLayer` allows for the initialization
        method of the QNode weights to be specified. The function passed to the argument must be
        from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ module. For
        example, weights can be randomly initialized from the normal distribution by passing:

        .. code-block::

            init_method = torch.nn.init.normal_

        If ``init_method`` is not specified, weights are randomly initialized from the uniform
        distribution on the interval :math:`[0, 2 \pi]`.

        **Full code example**

        The code block below shows how a circuit composed of templates from the
        :doc:`/introduction/templates` module can be combined with classical
        `Linear <https://pytorch.org/docs/stable/nn.html#linear>`__ layers to learn
        the two-dimensional `moons <https://scikit-learn.org/stable/modules/generated/sklearn
        .datasets.make_moons.html>`__ dataset.

        .. code-block:: python

            import numpy as np
            import pennylane as qml
            import torch
            import sklearn.datasets

            n_qubits = 2
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def qnode(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

            weight_shapes = {"weights": (3, n_qubits, 3)}

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
            clayer1 = torch.nn.Linear(2, 2)
            clayer2 = torch.nn.Linear(2, 2)
            softmax = torch.nn.Softmax(dim=1)
            model = torch.nn.Sequential(clayer1, qlayer, clayer2, softmax)

            samples = 100
            x, y = sklearn.datasets.make_moons(samples)
            y_hot = np.zeros((samples, 2))
            y_hot[np.arange(samples), y] = 1

            X = torch.tensor(x).float()
            Y = torch.tensor(y_hot).float()

            opt = torch.optim.SGD(model.parameters(), lr=0.5)
            loss = torch.nn.L1Loss()

        The model can be trained using:

        .. code-block:: python

            epochs = 8
            batch_size = 5
            batches = samples // batch_size

            data_loader = torch.utils.data.DataLoader(list(zip(X, Y)), batch_size=batch_size,
                                                      shuffle=True, drop_last=True)

            for epoch in range(epochs):

                running_loss = 0

                for x, y in data_loader:
                    opt.zero_grad()

                    loss_evaluated = loss(model(x), y)
                    loss_evaluated.backward()

                    opt.step()

                    running_loss += loss_evaluated

                avg_loss = running_loss / batches
                print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

        An example output is shown below:

        .. code-block:: rst

            Average loss over epoch 1: 0.5089
            Average loss over epoch 2: 0.4765
            Average loss over epoch 3: 0.2710
            Average loss over epoch 4: 0.1865
            Average loss over epoch 5: 0.1670
            Average loss over epoch 6: 0.1635
            Average loss over epoch 7: 0.1528
            Average loss over epoch 8: 0.1528
    """

    def __init__(self, qnode, weight_shapes: dict, init_method: Optional[Callable] = None):
        if not TORCH_IMPORTED:
            raise ImportError(
                "TorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()

        weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        # validate the QNode signature, and convert to a Torch QNode.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        self._signature_validation(qnode, weight_shapes)
        self.qnode = qnode

        try:
            # TODO: remove when the old QNode is removed
            self.qnode.to_torch()
        except AttributeError:
            self.qnode.interface = "torch"

        if not init_method:
            init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)

        self.qnode_weights = {}

        for name, size in weight_shapes.items():
            if len(size) == 0:
                self.qnode_weights[name] = torch.nn.Parameter(init_method(torch.Tensor(1))[0])
            else:
                self.qnode_weights[name] = torch.nn.Parameter(init_method(torch.Tensor(*size)))

            self.register_parameter(name, self.qnode_weights[name])

    def _signature_validation(self, qnode, weight_shapes):
        sig = inspect.signature(qnode.func).parameters

        if self.input_arg not in sig:
            raise TypeError(
                f"QNode must include an argument with name {self.input_arg} for inputting data"
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                f"{self.input_arg} argument should not have its dimension specified in "
                f"weight_shapes"
            )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds:
            if set(weight_shapes.keys()) | {self.input_arg} != set(sig.keys()):
                raise ValueError("Must specify a shape for every non-input parameter in the QNode")

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """

        if len(inputs.shape) > 1:
            # If the input size is not 1-dimensional, unstack the input along its first dimension, recursively call
            # the forward pass on each of the yielded tensors, and then stack the outputs back into the correct shape
            reconstructor = []
            for x in torch.unbind(inputs):
                reconstructor.append(self.forward(x))
            return torch.stack(reconstructor)

        # If the input is 1-dimensional, calculate the forward pass as usual
        return self._evaluate_qnode(inputs)

    def _evaluate_qnode(self, x):
        """Evaluates the QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        return self.qnode(**kwargs).type(x.dtype)

    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _input_arg = "inputs"

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg
