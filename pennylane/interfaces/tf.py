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
This module contains the :func:`TFQNode` function to convert Numpy-interfacing quantum nodes to TensorFlow
compatible quantum nodes.
"""
# pylint: disable=redefined-outer-name
from functools import partial

import numpy as np
import tensorflow as tf

from pennylane.utils import unflatten


if tf.__version__[0] == "1":
    import tensorflow.contrib.eager as tfe # pylint: disable=unused-import,ungrouped-imports
    Variable = tfe.Variable
else:
    from tensorflow import Variable # pylint: disable=unused-import,ungrouped-imports


def TFQNode(qnode):
    """Function that accepts a :class:`~.QNode`, and returns a TensorFlow eager-execution-compatible QNode.

    Args:
        qnode (~pennylane.qnode.QNode): a PennyLane QNode

    Returns:
        function: the QNode as a TensorFlow function
    """
    class qnode_str(partial):
        """TensorFlow QNode"""
        # pylint: disable=too-few-public-methods

        def __str__(self):
            """String representation"""
            detail = "<QNode: device='{}', func={}, wires={}, interface=TensorFlow>"
            return detail.format(qnode.device.short_name, qnode.func.__name__, qnode.num_wires)

        def __repr__(self):
            """REPL representation"""
            return self.__str__()

    @qnode_str
    @tf.custom_gradient
    def _TFQNode(*input_, **input_kwargs):
        # detach all input Tensors, convert to NumPy array
        args = [i.numpy() if isinstance(i, (Variable, tf.Tensor)) else i for i in input_]
        kwargs = {k:v.numpy() if isinstance(v, (Variable, tf.Tensor)) else v for k, v in input_kwargs.items()}

        # if NumPy array is scalar, convert to a Python float
        args = [i.tolist() if (isinstance(i, np.ndarray) and not i.shape) else i for i in args]
        kwargs = {k:v.tolist() if (isinstance(v, np.ndarray) and not v.shape) else v for k, v in kwargs.items()}

        # evaluate the QNode
        res = qnode(*args, **kwargs)

        if not isinstance(res, np.ndarray):
            # scalar result, cast to NumPy scalar
            res = np.array(res)

        def grad(grad_output):
            """Returns the vector-Jacobian product"""
            # evaluate the Jacobian matrix of the QNode
            jacobian = qnode.jacobian(args, **kwargs)

            grad_output_np = grad_output.numpy()

            # perform the vector-Jacobian product
            if not grad_output_np.shape:
                temp = grad_output_np * jacobian
            else:
                temp = grad_output_np.T @ jacobian

            # restore the nested structure of the input args
            grad_input = unflatten(temp.flat, args)
            return tuple(grad_input)

        return res, grad

    return _TFQNode
