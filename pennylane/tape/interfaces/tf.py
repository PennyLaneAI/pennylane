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
This module contains the mixin interface class for creating differentiable quantum tapes with
TensorFlow.
"""
# pylint: disable=protected-access, attribute-defined-outside-init
import numpy as np
import tensorflow as tf

try:
    from tensorflow.python.eager.tape import should_record_backprop
except ImportError:
    from tensorflow.python.eager.tape import should_record as should_record_backprop


from pennylane.tape.queuing import AnnotatedQueue


class TFInterface(AnnotatedQueue):
    """Mixin class for applying an TensorFlow interface to a :class:`~.JacobianTape`.

    TensorFlow-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyTFQuantumTape(TFInterface, JacobianTape):

    Alternatively, the TensorFlow interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the TensorFlow interface can be used to perform quantum-classical
    differentiable programming.

    .. note::

        If using a device that supports native TensorFlow computation and backpropagation, such as
        :class:`~.DefaultQubitTF`, the TensorFlow interface **does not need to be applied**. It is
        only applied to tapes executed on non-TensorFlow compatible devices.

    **Example**

    Once a TensorFlow quantum tape has been created, it can be differentiated using the gradient tape:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        with tf.GradientTape() as tape:
            with TFInterface.apply(JacobianTape()) as qtape:
                qml.Rot(p[0], p[1] ** 2 + p[0] * p[2], p[1] * tf.sin(p[2]), wires=0)
                expval(qml.PauliX(0))

            result = qtape.execute(dev)

    >>> print(result)
    tf.Tensor([0.06982072], shape=(1,), dtype=float64)
    >>> grad = tape.gradient(result, p)
    >>> print(grad)
    tf.Tensor([0.29874274 0.39710271 0.09958091], shape=(3,), dtype=float64)

    The TensorFlow interface defaults to ``tf.float64`` output. This can be modified by
    providing the ``dtype`` argument when applying the interface:

    >>> p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float32)
    >>> with tf.GradientTape() as tape:
    ...     TFInterface.apply(qtape, dtype=tf.float32)  # reusing the previous qtape
    ...     result = qtape.execute(dev)
    >>> print(result)
    tf.Tensor([0.06982072], shape=(1,), dtype=float32)
    >>> grad = tape.gradient(result, p)
    >>> print(grad)
    tf.Tensor([0.2895088  0.38464668 0.09645163], shape=(3,), dtype=float32)
    """

    dtype = tf.float64

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "tf"

    def _update_trainable_params(self):
        params = self.get_parameters(trainable_only=False)

        trainable_params = set()

        for idx, p in enumerate(params):
            # Determine which input tensors/Variables are being recorded for backpropagation.
            # The function should_record_backprop, documented here:
            # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/eager/tape.py#L167
            # accepts lists of *Tensors* (not Variables), returning True if all are being watched by one or more
            # existing gradient tapes, False if not.

            if isinstance(p, (tf.Variable, tf.Tensor)) and should_record_backprop(
                # we need to convert any Variable objects to Tensors here, otherwise
                # should_record_backprop will raise an error
                [tf.convert_to_tensor(p)]
            ):
                trainable_params.add(idx)

        self.trainable_params = trainable_params

    @staticmethod
    def convert_to_numpy(tensors):
        """Converts any TensorFlow tensors in a sequence to NumPy arrays.

        Args:
            tensors (Sequence[Any, tf.Variable, tf.Tensor]): input sequence

        Returns:
            list[Any, array]: list with all tensors converted to NumPy arrays
        """
        return [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in tensors]

    @tf.custom_gradient
    def _execute(self, params, **input_kwargs):
        # unwrap free parameters
        args = self.convert_to_numpy(params)

        # unwrap constant parameters
        all_params = self.get_parameters(trainable_only=False)
        all_params_unwrapped = self.convert_to_numpy(all_params)

        self.set_parameters(all_params_unwrapped, trainable_only=False)
        res = self.execute_device(args, input_kwargs["device"])
        self.set_parameters(all_params, trainable_only=False)

        def _evaluate_grad_matrix(grad_matrix_fn):
            self.set_parameters(all_params_unwrapped, trainable_only=False)
            grad_matrix = grad_matrix_fn(
                input_kwargs["device"], params=args, **self.jacobian_options
            )
            self.set_parameters(all_params, trainable_only=False)
            return tf.constant(grad_matrix, dtype=self.dtype)

        def _evaluate_vector_grad_product(dy, grad_matrix):
            # Reshape gradient output array as a 2D row-vector.
            grad_matrix = tf.cond(
                tf.rank(grad_matrix) > 2,
                lambda: tf.reshape(grad_matrix, [-1, dy.shape[0]]),
                lambda: grad_matrix,
            )

            dy_row = tf.reshape(dy, [1, -1])
            # Calculate the vector-Gradient matrix product, and unstack the output.
            vgp = tf.matmul(dy_row, grad_matrix)
            return tf.unstack(tf.reshape(vgp, [-1]), num=dy.shape[1])

        def jacobian_product(dy, **tfkwargs):
            variables = tfkwargs.get("variables", None)

            @tf.custom_gradient
            def jacobian(p):
                def hessian_product(ddy, **tfkwargs):
                    variables = tfkwargs.get("variables", None)
                    hessian = _evaluate_grad_matrix(self.hessian)
                    vhp = _evaluate_vector_grad_product(ddy, hessian)
                    return (vhp, variables) if variables is not None else vhp

                return _evaluate_grad_matrix(self.jacobian), hessian_product

            vjp = _evaluate_vector_grad_product(dy, jacobian(params))
            return (vjp, variables) if variables is not None else vjp

        if self.is_sampled:
            return res, grad

        if res.dtype == np.dtype("object"):
            res = np.hstack(res)

        return tf.convert_to_tensor(res, dtype=self.dtype), jacobian_product

    @classmethod
    def apply(cls, tape, dtype=tf.float64):
        """Apply the TensorFlow interface to an existing tape in-place.

        Args:
            tape (.JacobianTape): a quantum tape to apply the TF interface to
            dtype (tf.dtype): the dtype that the returned quantum tape should
                output

        **Example**

        >>> with JacobianTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> TFInterface.apply(tape)
        >>> tape
        <TFQuantumTape: wires=<Wires = [0]>, params=1>
        """
        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("TFQuantumTape", (cls, tape_class), {"dtype": dtype})
        tape._update_trainable_params()
        return tape
