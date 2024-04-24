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

import contextlib
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Dict, Union, Any

from pennylane import QNode

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
    r"""Converts a :class:`~.QNode` to a Torch layer.

    The result can be used within the ``torch.nn``
    `Sequential <https://pytorch.org/docs/stable/nn.html#sequential>`__ or
    `Module <https://pytorch.org/docs/stable/nn.html#module>`__ classes for
    creating quantum and hybrid models.

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Torch layer
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding shapes
        init_method (Union[Callable, Dict[str, Union[Callable, torch.Tensor]], None]): Either a
            `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ function for
            initializing all QNode weights or a dictionary specifying the callable/value used for
            each weight. If not specified, weights are randomly initialized using the uniform
            distribution over :math:`[0, 2 \pi]`.

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
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

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

    .. details::
        :title: Usage Details

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

        **Output shape**

        If the QNode returns a single measurement, then the output of the ``KerasLayer`` will have
        shape ``(batch_dim, *measurement_shape)``, where ``measurement_shape`` is the output shape
        of the measurement:

        .. code-block::

            def print_output_shape(measurements):
                n_qubits = 2
                dev = qml.device("default.qubit", wires=n_qubits, shots=100)

                @qml.qnode(dev)
                def qnode(inputs, weights):
                    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    if len(measurements) == 1:
                        return qml.apply(measurements[0])
                    return [qml.apply(m) for m in measurements]

                weight_shapes = {"weights": (3, n_qubits, 3)}
                qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

                batch_dim = 5
                x = torch.zeros((batch_dim, n_qubits))
                return qlayer(x).shape

        >>> print_output_shape([qml.expval(qml.Z(0))])
        torch.Size([5])
        >>> print_output_shape([qml.probs(wires=[0, 1])])
        torch.Size([5, 4])
        >>> print_output_shape([qml.sample(wires=[0, 1])])
        torch.Size([5, 100, 2])

        If the QNode returns multiple measurements, then the measurement results will be flattened
        and concatenated, resulting in an output of shape ``(batch_dim, total_flattened_dim)``:

        >>> print_output_shape([qml.expval(qml.Z(0)), qml.probs(wires=[0, 1])])
        torch.Size([5, 5])
        >>> print_output_shape([qml.probs([0, 1]), qml.sample(wires=[0, 1])])
        torch.Size([5, 204])

        **Initializing weights**

        If ``init_method`` is not specified, weights are randomly initialized from the uniform
        distribution on the interval :math:`[0, 2 \pi]`.

        Alternative a): The optional ``init_method`` argument of :class:`~.TorchLayer` allows for the initialization
        method of the QNode weights to be specified. The function passed to the argument must be
        from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ module. For
        example, weights can be randomly initialized from the normal distribution by passing:

        .. code-block::

            init_method = torch.nn.init.normal_

        Alternative b): Two dictionaries ``weight_shapes`` and ``init_method`` are passed, whose ``keys`` match the ``args`` of the qnode.

        .. code-block::

            @qml.qnode(dev)
            def qnode(inputs, weights_0, weights_1, weights_2, weight_3, weight_4):
                qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.templates.StronglyEntanglingLayers(weights_0, wires=range(n_qubits))
                qml.templates.BasicEntanglerLayers(weights_1, wires=range(n_qubits))
                qml.Rot(*weights_2, wires=0)
                qml.RY(weight_3, wires=1)
                qml.RZ(weight_4, wires=1)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))


            weight_shapes = {
                "weights_0": (3, n_qubits, 3),
                "weights_1": (3, n_qubits),
                "weights_2": 3,
                "weight_3": 1,
                "weight_4": (1,),
            }

            init_method = {
                "weights_0": torch.nn.init.normal_,
                "weights_1": torch.nn.init.uniform,
                "weights_2": torch.tensor([1., 2., 3.]),
                "weight_3": torch.tensor(1.),  # scalar when shape is not an iterable and is <= 1
                "weight_4": torch.tensor([1.]),
            }

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes, init_method=init_method)

        **Model saving**

        Instances of ``TorchLayer`` can be saved using the usual ``torch.save()`` utility:

        .. code-block::

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes)
            torch.save(qlayer.state_dict(), SAVE_PATH)

        To load the layer again, an instance of the class must be created first before calling ``torch.load()``,
        as required by PyTorch:

        .. code-block::

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes)
            qlayer.load_state_dict(torch.load(SAVE_PATH))
            qlayer.eval()

        .. note::

            Currently ``TorchLayer`` objects cannot be saved using the ``torch.save(qlayer, SAVE_PATH)``
            syntax. In order to save a ``TorchLayer`` object, the object's ``state_dict`` should be
            saved instead.

        PyTorch modules that contain ``TorchLayer`` objects can also be saved and loaded.

        Saving:

        .. code-block::

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes)
            clayer = torch.nn.Linear(2, 2)
            model = torch.nn.Sequential(qlayer, clayer)
            torch.save(model.state_dict(), SAVE_PATH)

        Loading:

        .. code-block::

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes)
            clayer = torch.nn.Linear(2, 2)
            model = torch.nn.Sequential(qlayer, clayer)
            model.load_state_dict(torch.load(SAVE_PATH))
            model.eval()

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
                return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

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

    def __init__(
        self,
        qnode: QNode,
        weight_shapes: dict,
        init_method: Union[Callable, Dict[str, Union[Callable, Any]]] = None,
        # FIXME: Cannot change type `Any` to `torch.Tensor` in init_method because it crashes the
        # tests that don't use torch module.
    ):
        if not TORCH_IMPORTED:
            raise ImportError(
                "TorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()

        weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else () if size == 1 else (size,))
            for weight, size in weight_shapes.items()
        }

        # validate the QNode signature, and convert to a Torch QNode.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        self._signature_validation(qnode, weight_shapes)
        self.qnode = qnode
        self.qnode.interface = "torch"

        self.qnode_weights: Dict[str, torch.nn.Parameter] = {}

        self._init_weights(init_method=init_method, weight_shapes=weight_shapes)
        self._initialized = True

    def _signature_validation(self, qnode: QNode, weight_shapes: dict):
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

        if inspect.Parameter.VAR_KEYWORD not in param_kinds and set(weight_shapes.keys()) | {
            self.input_arg
        } != set(sig.keys()):
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = inputs.shape[:-1]
            inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        results = self._evaluate_qnode(inputs)

        if isinstance(results, tuple):
            if has_batch_dim:
                results = [torch.reshape(r, (*batch_dims, *r.shape[1:])) for r in results]
            return torch.stack(results, dim=0)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results

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
        res = self.qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        def _combine_dimensions(_res):
            if len(x.shape) > 1:
                _res = [torch.reshape(r, (x.shape[0], -1)) for r in _res]
            return torch.hstack(_res).type(x.dtype)

        if isinstance(res, tuple) and len(res) > 1:
            return tuple(_combine_dimensions(r) for r in res)

        return _combine_dimensions(res)

    def construct(self, args, kwargs):
        """Constructs the wrapped QNode on input data using the initialized weights.

        This method was added to match the QNode interface. The provided args
        must contain a single item, which is the input to the layer. The provided
        kwargs is unused.

        Args:
            args (tuple): A tuple containing one entry that is the input to this layer
            kwargs (dict): Unused
        """
        x = args[0]
        kwargs = {
            self.input_arg: x,
            **{arg: weight.data.to(x) for arg, weight in self.qnode_weights.items()},
        }
        self.qnode.construct((), kwargs)

    def __getattr__(self, item):
        """If the qnode is initialized, first check to see if the attribute is on the qnode."""
        if self._initialized:
            with contextlib.suppress(AttributeError):
                return getattr(self.qnode, item)

        return super().__getattr__(item)

    def __setattr__(self, item, val):
        """If the qnode is initialized and item is already a qnode property, update it on the qnode, else
        just update the torch layer itself."""
        if self._initialized and item in self.qnode.__dict__:
            setattr(self.qnode, item, val)
        else:
            super().__setattr__(item, val)

    def _init_weights(
        self,
        weight_shapes: Dict[str, tuple],
        init_method: Union[Callable, Dict[str, Union[Callable, Any]], None],
    ):
        r"""Initialize and register the weights with the given init_method. If init_method is not
        specified, weights are randomly initialized from the uniform distribution on the interval
        [0, 2π].

        Args:
            weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
                their corresponding shapes
            init_method (Union[Callable, Dict[str, Union[Callable, torch.Tensor]], None]): Either a
                `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ function for
                initializing the QNode weights or a dictionary specifying the callable/value used for
                each weight. If not specified, weights are randomly initialized using the uniform
                distribution over :math:`[0, 2 \pi]`.
        """

        def init_weight(weight_name: str, weight_size: tuple) -> torch.Tensor:
            """Initialize weights.

            Args:
                weight_name (str): weight name
                weight_size (tuple): size of the weight

            Returns:
                torch.Tensor: tensor containing the weights
            """
            if init_method is None:
                init = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
            elif callable(init_method):
                init = init_method
            elif isinstance(init_method, dict):
                init = init_method[weight_name]
                if isinstance(init, torch.Tensor):
                    if tuple(init.shape) != weight_size:
                        raise ValueError(
                            f"The Tensor specified for weight '{weight_name}' doesn't have the "
                            + "appropiate shape."
                        )
                    return init
            return init(torch.Tensor(*weight_size)) if weight_size else init(torch.Tensor(1))[0]

        for name, size in weight_shapes.items():
            self.qnode_weights[name] = torch.nn.Parameter(
                init_weight(weight_name=name, weight_size=size)
            )

            self.register_parameter(name, self.qnode_weights[name])

    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _input_arg = "inputs"
    _initialized = False

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg

    @staticmethod
    def set_input_argument(input_name: str = "inputs") -> None:
        """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
        TorchLayer._input_arg = input_name
