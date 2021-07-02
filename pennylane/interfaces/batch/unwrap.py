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
"""
This module contains a context manager for unwrapping tapes
"""


class UnwrapTape:
    """A context manager that unwraps a tape with TensorFlow parameters
    to NumPy arrays.

    Args:
        tape (.QuantumTape): the quantum tape to unwrap

    Returns:
        .QuantumTape: the unwrapped quantum tape

    **Example**

    >>> with tf.GradientTape():
    ...     with qml.tape.QuantumTape() as tape:
    ...         qml.RX(tf.Variable(0.1), wires=0)
    ...         qml.RY(tf.constant(0.2), wires=0)
    ...         qml.RZ(tf.Variable(0.3), wires=0)
    ...     with UnwrapTape(tape) as unwrapped_tape:
    ...         print("Trainable params:", unwrapped_tape.trainable_params)
    ...         print("Unwrapped params:", unwrapped_tape.get_parameters())
    Trainable params: {0, 2}
    Unwrapped params: [0.1, 0.3]
    >>> print("Original parameters:", tape.get_parameters())
    Original parameters: [<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.1>,
      <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.3>]
    """

    def __init__(self, tape, unwrap_fn, trainable_fn):
        self.tape = tape
        self.unwrap_fn = unwrap_fn
        self.trainable_fn = trainable_fn

        self._original_params = None
        self._unwrapped_params = None

    def __enter__(self):
        self.tape.trainable_params, self._original_params = self.trainable_fn(self.tape)
        self._unwrapped_params = self.unwrap_fn(self._original_params)
        self.tape.set_parameters(self._unwrapped_params, trainable_only=False)
        return self.tape

    def __exit__(self, exception_type, exception_value, traceback):
        self.tape.set_parameters(self._original_params, trainable_only=False)
