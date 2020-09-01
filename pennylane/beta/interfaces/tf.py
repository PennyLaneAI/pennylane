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


from pennylane.beta.queuing import AnnotatedQueue


class TFInterface(AnnotatedQueue):
    """Mixin class for applying an TensorFlow interface to a :class:`~.QuantumTape`.

    TensorFlow-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyAutogradQuantumTape(AutogradInterface, QuantumTape):

    Alternatively, the autograd interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the autograd interface can be used to perform quantum-classical
    differentiable programming.

    .. note::

        If using a device that supports native autograd computation and backpropagation, such as
        :class:`~.DefaultQubitTF`, the TensorFlow interface **does not need to be applied**. It is
        only applied to tapes executed on non-TensorFlow compatible devices.

    **Example**

    One a TensorFlow quantum tape has been created, it can be differentiated using the gradient tape:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        with tf.GradientTape() as tape:
            with TFInterface.apply(QuantumTape()) as qtape:
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
        params = [o.data for o in self.operations + self.observables]
        params = [item for sublist in params for item in sublist]

        trainable_params = set()

        for idx, p in enumerate(params):
            # Determine which input tensors/Variables are being recorded for backpropagation.
            # The function should_record_backprop, documented here:
            # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/eager/tape.py#L163
            # accepts lists of *tensors* (not Variables), returning True if all are being watched by one or more
            # existing gradient tape, False if not.
            if isinstance(p, (tf.Variable, tf.Tensor)) and should_record_backprop(
                [tf.convert_to_tensor(p)]
            ):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        return params

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
        all_params = self.get_parameters(free_only=False)
        all_params_unwrapped = self.convert_to_numpy(all_params)

        self.set_parameters(all_params_unwrapped, free_only=False)
        res = self.execute_device(args, input_kwargs["device"])
        self.set_parameters(all_params, free_only=False)

        def grad(grad_output, **tfkwargs):
            variables = tfkwargs.get("variables", None)

            self.set_parameters(all_params_unwrapped, free_only=False)
            jacobian = self.jacobian(input_kwargs["device"], params=args)
            self.set_parameters(all_params, free_only=False)

            jacobian = tf.constant(jacobian, dtype=self.dtype)

            # Reshape gradient output array as a 2D row-vector.
            # grad_output = tf.cast(grad_output, dtype=self.dtype)
            grad_output_row = tf.transpose(tf.reshape(grad_output, [-1, 1]))

            # Calculate the vector-Jacobian matrix product, and unstack the output.
            grad_input = tf.matmul(grad_output_row, jacobian)
            grad_input = tf.unstack(tf.reshape(grad_input, [-1]))

            if variables is not None:
                return grad_input, variables

            return grad_input

        if res.dtype == np.dtype("object"):
            res = np.hstack(res)

        return tf.convert_to_tensor(res, dtype=self.dtype), grad

    @classmethod
    def apply(cls, tape, dtype=tf.float64):
        """Apply the TensorFlow interface to an existing tape in-place.

        Args:
            tape (.QuantumTape): a quantum tape to apply the TF interface to
            dtype (tf.dtype): the dtype that the returned quantum tape should
                output

        **Example**

        >>> with QuantumTape() as tape:
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
