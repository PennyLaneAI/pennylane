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
import pennylane as qml
import numpy as np


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


def _vector_jacobian_product(dy, jac):
    num_params = len(jac)
    dy_row = qml.math.reshape(dy, [-1])
    jac = qml.math.transpose(qml.math.stack(jac))
    jac = qml.math.reshape(jac, [-1, num_params])
    return qml.math.tensordot(jac, dy_row, [[0], [0]])


def batch_vjp(dy, tapes, execute_fn, gradient_fn, reduction="append", **kwargs):
    reshape_info = []
    gradient_tapes = []
    processing_fns = []
    classical_jacs = []

    unsupported_op = lambda op: op.grad_recipe is None
    supported_op = lambda op: op.grad_recipe is not None
    trainable_op = lambda op: any(qml.math.requires_grad(p) for p in op.parameters)

    for t in tapes:

        if any(unsupported_op(op) and trainable_op(op) for op in t.operations):

            def classical_gate_processing(*x):
                t.set_parameters(x)
                t._expanded_tape = t.expand(
                    depth=10,
                    stop_at=lambda obj: not isinstance(obj, qml.measure.MeasurementProcess)
                    and ((supported_op(obj) and trainable_op(obj)) or not trainable_op(obj))
                )
                return qml.math.stack(t._expanded_tape.get_parameters())

            params = t.get_parameters()
            c_jac = [qml.jacobian(classical_gate_processing, argnum=i)(*params) for i in range(len(params))][0]
            t._expanded_tape.set_parameters([qml.math.toarray(t) for t in t._expanded_tape.get_parameters()])
            t = t._expanded_tape

        else:
            c_jac = None

        g_tapes, fns = gradient_fn(t)
        reshape_info.extend([len(t) for t in g_tapes])

        g_tapes = [item for sublist in g_tapes for item in sublist]

        processing_fns.append(fns)
        gradient_tapes.extend(g_tapes)
        classical_jacs.append(c_jac)

    results = execute_fn(gradient_tapes, gradient_fn=gradient_fn, **kwargs)
    vjps = []
    start = 0

    for t, d in zip(range(len(tapes)), dy):
        num_params = len(tapes[t].trainable_params)
        jac = []

        if num_params == 0:
            vjps.append(None)
            continue

        for fn, res_len in zip(processing_fns[t], reshape_info):
            # extract the correct results from the flat list
            res = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the gradient
            jac.append(fn(res))

        getattr(vjps, reduction)(_vector_jacobian_product(d, jac))

        if classical_jacs[t] is not None:
            vjps[-1] = qml.math.tensordot(vjps[-1], classical_jacs[t], axes=[[0], [0]])

    return vjps
