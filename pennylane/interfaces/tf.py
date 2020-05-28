# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the :func:`to_tf` function to convert Numpy-interfacing quantum nodes to TensorFlow
compatible quantum nodes.
"""
# pylint: disable=redefined-outer-name
import numbers
from collections import Iterable
from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow import Variable  # pylint: disable=unused-import,ungrouped-imports

try:
    from tensorflow.python.eager.tape import should_record_backprop
except ImportError:
    from tensorflow.python.eager.tape import should_record as should_record_backprop

from pennylane.utils import _flatten


def unflatten_tf(flat, model):
    """Restores an arbitrary nested structure to a flattened TF tensor.

    See also :func:`~.unflatten`.

    Args:
        flat (tf.Tensor): 1D tensor of items
        model (array, Iterable, Number): model nested structure

    Returns:
        Union[tf.Tensor, list], array: first elements of flat arranged into the nested
        structure of model, unused elements of flat

    Raises:
        TypeError: if ``model`` contains an object of unsupported type
    """
    if isinstance(model, (numbers.Number, str)):
        return flat[0], flat[1:]

    if isinstance(model, (tf.Tensor, tf.Variable)):
        idx = tf.size(model)
        res = tf.reshape(flat[:idx], model.shape)
        return res, flat[idx:]

    if isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = unflatten_tf(flat, x)
            res.append(val)
        return res, flat

    raise TypeError("Unsupported type in the model: {}".format(type(model)))


def to_tf(qnode, dtype=None):
    """Function that accepts a :class:`~.QNode`, and returns a TensorFlow eager-execution-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode
        dtype (tf.DType): target output type of QNode; uses the TensorFlow equivalent of the
            QNode output type if ``dtype`` is not specified

    Returns:
        function: the QNode as a TensorFlow function
    """

    class qnode_str(partial):
        """TensorFlow QNode"""

        # pylint: disable=too-few-public-methods

        @property
        def interface(self):
            """String representing the QNode interface"""
            return "tf"

        def __str__(self):
            """String representation"""
            detail = "<QNode: device='{}', func={}, wires={}, interface={}>"
            return detail.format(
                qnode.device.short_name, qnode.func.__name__, qnode.num_wires, self.interface
            )

        def __repr__(self):
            """REPL representation"""
            return self.__str__()

        print_applied = qnode.print_applied
        jacobian = qnode.jacobian
        metric_tensor = qnode.metric_tensor
        draw = qnode.draw
        func = qnode.func
        num_variables = property(lambda self: qnode.num_variables)
        arg_vars = property(lambda self: qnode.arg_vars)

    @qnode_str
    @tf.custom_gradient
    def _TFQNode(*input_, **input_kwargs):
        # Determine which input tensors/Variables are being recorded for backpropagation.
        # The function should_record_backprop, documented here:
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/eager/tape.py#L163
        # accepts lists of *tensors* (not Variables), returning True if all are being watched by one or more
        # existing gradient tape, False if not.
        requires_grad = [
            should_record_backprop([tf.convert_to_tensor(i)])
            if isinstance(i, (Variable, tf.Tensor))
            else False
            for i in input_
        ]

        # detach all input Tensors, convert to NumPy array
        args = [i.numpy() if isinstance(i, (Variable, tf.Tensor)) else i for i in input_]
        kwargs = {
            k: v.numpy() if isinstance(v, (Variable, tf.Tensor)) else v
            for k, v in input_kwargs.items()
        }

        # if NumPy array is scalar, convert to a Python float
        args = [i.tolist() if (isinstance(i, np.ndarray) and not i.shape) else i for i in args]
        kwargs = {
            k: v.tolist() if (isinstance(v, np.ndarray) and not v.shape) else v
            for k, v in kwargs.items()
        }

        # evaluate the QNode
        res = qnode(*args, **kwargs)

        if not isinstance(res, np.ndarray):
            # scalar result, cast to NumPy scalar
            res = np.array(res)

        def grad(grad_output, **tfkwargs):
            """Returns the vector-Jacobian product"""
            diff_indices = None
            non_diff_indices = set()

            # determine the QNode variables which should be differentiated
            for differentiable, arg_variable in zip(requires_grad, qnode.arg_vars):
                if not differentiable:
                    indices = [i.idx for i in _flatten(arg_variable)]
                    non_diff_indices.update(indices)

            if non_diff_indices:
                diff_indices = set(range(qnode.num_variables)) - non_diff_indices

            # evaluate the Jacobian matrix of the QNode
            variables = tfkwargs.get("variables", None)
            jacobian = qnode.jacobian(args, kwargs, wrt=diff_indices)
            jacobian = tf.constant(jacobian, dtype=dtype)

            # Reshape gradient output array as a 2D row-vector.
            grad_output_row = tf.transpose(tf.reshape(grad_output, [-1, 1]))

            # Calculate the vector-Jacobian matrix product, and flatten the output.
            grad_input = tf.matmul(grad_output_row, jacobian)
            grad_input = tf.reshape(grad_input, [-1])

            if non_diff_indices:
                # TensorFlow requires we return a gradient of size (num_variables,)
                res = np.zeros([qnode.num_variables])
                indices = np.fromiter(diff_indices, dtype=np.int64)
                res[indices] = grad_input
                grad_input = tf.constant(res, dtype=dtype)

            grad_input_unflattened = unflatten_tf(grad_input, input_)[0]

            if variables is not None:
                return grad_input_unflattened, variables

            return grad_input_unflattened

        return tf.convert_to_tensor(res, dtype=dtype), grad

    return _TFQNode
