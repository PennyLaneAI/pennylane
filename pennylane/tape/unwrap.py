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
import contextlib
import pennylane as qml


class Unwrap:
    """A context manager that unwraps multiple tapes with tensor-like parameters
    to NumPy arrays. In addition, this context manager also correctly infers
    the trainable parameters of the tapes.

    Args:
        *tapes (.QuantumTape): a sequence of quantum tapes to unwrap
        params (List[List[tensor_like or float]] or None): Nested list of parameters
            for each tape in the input sequence. If provided, these parameter
            values will be applied to each tape within the context.

    Returns:
        Sequence[.QuantumTape]: a sequence of unwrapped quantum tapes

    **Example**

    Consider the following two tapes:

    .. code-block:: python

        x = torch.tensor([0.1, 0.2, 0.3], requires_grad=True, dtype=torch.float64)
        y = torch.tensor([0.5, 0.6], dtype=torch.float64)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(x[0], wires=0)
            qml.RY(y[1], wires=0)
            qml.RZ(x[2], wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(x[1], wires=0)
            qml.RY(x[1], wires=0)
            qml.RZ(y[0], wires=0)

    We can use the ``Unwrap`` context manager to simultaneously unwrap the
    parameters of both tapes:

    >>> with Unwrap(tape1, tape2):
    ...     print("Tape 1 trainable:", tape1.trainable_params)
    ...     print("Tape 1 params:", tape1.get_parameters())
    ...     print("Tape 2 trainable:", tape2.trainable_params)
    ...     print("Tape 2 params:", tape2.get_parameters())
    Tape 1 trainable: {0, 2}
    Tape 1 params: [0.1, 0.3]
    Tape 2 trainable: {0, 1}
    Tape 2 params: [0.2, 0.2]

    Outside of the context, the original parameter types remain:

    >>> print("Original parameters:", tape1.get_parameters())
    Original parameters: [tensor(0.1000, dtype=torch.float64, grad_fn=<SelectBackward>),
      tensor(0.3000, dtype=torch.float64, grad_fn=<SelectBackward>)]
    """

    def __init__(self, *tapes, params=None):
        self.tapes = tapes
        self.stack = None
        self.params = params

    def __enter__(self):
        with contextlib.ExitStack() as stack:
            for i, tape in enumerate(self.tapes):
                stack.enter_context(
                    UnwrapTape(tape, params=self.params[i] if self.params is not None else None)
                )

            self.stack = stack.pop_all()

        return self.tapes

    def __exit__(self, exception_type, exception_value, traceback):
        self.stack.__exit__(exception_type, exception_value, traceback)
        self.stack = None


class UnwrapTape:
    """A context manager that unwraps a single tape with tensor-like parameters
    to NumPy arrays. In addition, this context manager also correctly infers
    the trainable parameters of the tape.

    Args:
        tape (.QuantumTape): the quantum tape to unwrap
        params (List[tensor_like or float] or None): List of parameters.
            If provided, these parameters will be applied to the tape within the context.

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

    def __init__(self, tape, params=None):
        self.tape = tape
        self._original_params = None
        self._unwrapped_params = params or None

    def __enter__(self):
        self._original_params = self.tape.get_parameters(trainable_only=False)
        self._unwrapped_params = self._unwrapped_params or qml.math.unwrap(self._original_params)
        self.tape.set_parameters(self._unwrapped_params, trainable_only=False)

        return self.tape

    def __exit__(self, exception_type, exception_value, traceback):
        self.tape.set_parameters(self._original_params, trainable_only=False)
