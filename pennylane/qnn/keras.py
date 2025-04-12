# Copyright 2018-2024 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the classes and functions for integrating QNodes with the Keras Layer
API."""
import inspect
import warnings
from collections.abc import Iterable
from typing import Optional, Text

from packaging.version import Version

from pennylane import PennyLaneDeprecationWarning
from keras.layers import Layer
from keras import ops
import keras 

current_backend = keras.config.backend()
CORRECT_KERAS_VERSION = True

if current_backend == "tensorflow":
    try:
        import tensorflow as tf
        CORRECT_BACKEND_VERSION = Version(tf.__version__) >= Version("2.0.0")

    except ImportError:
        # The following allows this module to be imported even if TensorFlow is not installed. Users
        # will instead see an ImportError when instantiating the KerasLayer.
        from abc import ABC

        Layer = ABC
        CORRECT_BACKEND_VERSION = False

elif current_backend == "torch":
    try:
        import torch
        from torch.nn import Module

        CORRECT_BACKEND_VERSION = True
    except ImportError:
        # The following allows this module to be imported even if PyTorch is not installed. Users
        # will instead see an ImportError when instantiating the TorchLayer.
        from unittest.mock import Mock

        Module = Mock
        CORRECT_BACKEND_VERSION = False


class KerasLayer(Layer):
    """Converts a :class:`~.QNode` to a Keras
    `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__.

    The result can be used within the Keras
    `Sequential <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`__ or
    `Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ classes for
    creating quantum and hybrid models.

    Args:
        qnode (qml.QNode): the PennyLane QNode to be converted into a Keras Layer_
        weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
            their corresponding shapes
        output_dim (int): the output dimension of the QNode
        weight_specs (dict[str, dict]): An optional dictionary for users to provide additional
            specifications for weights used in the QNode, such as the method of parameter
            initialization. This specification is provided as a dictionary with keys given by the
            arguments of the `add_weight()
            <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight>`__
            method and values being the corresponding specification.
        **kwargs: additional keyword arguments passed to the Layer_ base class
    """

    def __init__(
        self,
        qnode,
        weight_shapes: dict,
        output_dim,
        weight_specs: Optional[dict] = None,
        **kwargs,
    ):
        warnings.warn(
            "The 'KerasLayer' class is deprecated and will be removed in v0.42. ",
            PennyLaneDeprecationWarning,
        )
        # pylint: disable=too-many-arguments
        if not CORRECT_BACKEND_VERSION:
            raise ImportError(
                "KerasLayer requires TensorFlow version <=2 or PyTorch "
            )
        ## Loads backend version
        self.current_backend = keras.config.backend()
        
        self.weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        self._signature_validation(qnode, weight_shapes)

        self.qnode = qnode
        if self.current_backend == "torch":
            if self.qnode.interface not in ("auto", "torch", "pytorch"):
                raise ValueError(f"Invalid interface '{self.qnode.interface}' for KerasLayer with Torch Backend")
        elif self.current_backend == "tensorflow":
            if self.qnode.interface not in (
                "auto",
                "tf",
                "tensorflow",
                "tensorflow-autograph",
                "tf-autograph",
            ):
                raise ValueError(f"Invalid interface '{self.qnode.interface}' for KerasLayer with TensorFlow Backend")

        # Allows output_dim to be specified as an int or as a tuple, e.g, 5, (5,), (5, 2), [5, 2]
        # Note: Single digit values will be considered an int and multiple as a tuple, e.g [5,] or (5,)
        # are passed as integer 5 and [5, 2] will be passes as tuple (5, 2)
        if isinstance(output_dim, Iterable) and len(output_dim) > 1:
            self.output_dim = tuple(output_dim)
        else:
            self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        self.weight_specs = weight_specs if weight_specs is not None else {}

        self.qnode_weights = {}

        if CORRECT_KERAS_VERSION:
            super().__init__(dynamic=True, **kwargs)
        else:  # pragma: no cover
            super().__init__(**kwargs)

        # no point in delaying the initialization of weights, since we already know their shapes
        self.build(None)
        self._initialized = True

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

    def build(self, input_shape):
        """Initializes the QNode weights.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data; this is unused since
                the weight shapes are already known in the __init__ method.
        """
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(name=weight, shape=size, **spec)

        super().build(input_shape)

    def call(self, inputs):
        """Evaluates the QNode on input data using the initialized weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = ops.shape(inputs)[:-1]
            inputs = ops.reshape(inputs, (-1, inputs.shape[-1]))

        # calculate the forward pass as usual
        results = self._evaluate_qnode(inputs)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
            new_shape = ops.concat([batch_dims, ops.shape(results)[1:]], axis=0)
            results = ops.reshape(results, new_shape)

        return results

    def _evaluate_qnode(self, x):
        """Evaluates a QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{k: 1.0 * w for k, w in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, (list, tuple)):
            if len(x.shape) > 1:
                # multi-return and batch dim case
                res = [ops.reshape(r, (ops.shape(x)[0], ops.prod(r.shape[1:]))) for r in res]

            # multi-return and no batch dim
            return ops.hstack(res)

        return res

    def construct(self, args, kwargs):
        """Constructs the wrapped QNode on input data using the initialized weights.

        This method was added to match the QNode interface. The provided args
        must contain a single item, which is the input to the layer. The provided
        kwargs is unused.

        Args:
            args (tuple): A tuple containing one entry that is the input to this layer
            kwargs (dict): Unused
        """
        # GradientTape required to ensure that the weights show up as trainable on the qtape
        if self.current_backend == "tensorflow":
            with tf.GradientTape() as tape:
                tape.watch(list(self.qnode_weights.values()))

                inputs = args[0]
                kwargs = {self.input_arg: inputs, **{k: 1.0 * w for k, w in self.qnode_weights.items()}}
                self.qnode.construct((), kwargs)
        elif self.current_backend == "torch":
            x = args[0]
            kwargs = {
                self.input_arg: x,
                **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
            }
            self.qnode.construct((), kwargs)

    def __getattr__(self, item):
        """If the given attribute does not exist in the class, look for it in the wrapped QNode."""
        if self._initialized and hasattr(self.qnode, item):
            return getattr(self.qnode, item)

        return super().__getattr__(item)

    def __setattr__(self, item, val):
        """If the given attribute does not exist in the class, try to set it in the wrapped QNode."""
        if self._initialized and hasattr(self.qnode, item):
            setattr(self.qnode, item, val)
        else:
            super().__setattr__(item, val)

    def compute_output_shape(self, input_shape):
        """Computes the output shape after passing data of shape ``input_shape`` through the
        QNode.

        Args:
            input_shape (tuple): shape of input data

        Returns:
            tf.tuple: shape of output data
        """
        return tuple([input_shape[0]] + [self.output_dim])

    def __str__(self):
        detail = "<Quantum Keras Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _input_arg = "inputs"
    _initialized = False

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Keras
        `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__. Set to
        ``"inputs"``."""
        return self._input_arg

    @staticmethod
    def set_input_argument(input_name: Text = "inputs") -> None:
        """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
        KerasLayer._input_arg = input_name

    def get_config(self) -> dict:
        """
        Get serialized layer configuration

        Returns:
            dict: layer configuration
        """
        config = super().get_config()

        config.update(
            {
                "output_dim": self.output_dim,
                "weight_specs": self.weight_specs,
                "weight_shapes": self.weight_shapes,
            }
        )
        return config
