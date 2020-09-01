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
Differentiable quantum tapes with TF interface.
"""
# pylint: disable=protected-access
import tensorflow as tf

try:
    from tensorflow.python.eager.tape import should_record_backprop
except ImportError:
    from tensorflow.python.eager.tape import should_record as should_record_backprop


from pennylane.beta.queuing import AnnotatedQueue


class TFInterface(AnnotatedQueue):
    name = "tf"
    # cast = staticmethod(tf.stack)
    dtype = tf.float64

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the quantum tap (if any)"""
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
        return [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in tensors]


    @tf.custom_gradient
    def _execute(self, params, **input_kwargs):
        """TensorFlow execution interface"""

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

        return tf.convert_to_tensor(res, dtype=self.dtype), grad

    @classmethod
    def apply(cls, tape, dtype=None):
        tape.__class__ = type("TFQuantumTape", (cls, tape.__class__), {"dtype": dtype})
        tape._update_trainable_params()
        return tape
