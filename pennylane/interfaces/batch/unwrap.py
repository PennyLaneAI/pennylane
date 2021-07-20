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

    _n = kwargs.pop("_n")

    for t, d in zip(tapes, dy):
        t._par_info = {}
        t._update()

        if qml.math.allclose(d, 0):
            g_tapes = []
            fns = []
            reshape_info.append([0])
        else:
            g_tapes, fns = gradient_fn(t, _n=_n)
            reshape_info.append([len(t) for t in g_tapes])
            g_tapes = [item for sublist in g_tapes for item in sublist]

        processing_fns.append(fns)
        gradient_tapes.extend(g_tapes)

    results = execute_fn(gradient_tapes, gradient_fn=gradient_fn, _n=_n+1, **kwargs)
    vjps = []
    start = 0

    for t, d in zip(range(len(tapes)), dy):
        num_params = len(tapes[t].trainable_params)
        jac = []

        if num_params == 0:
            vjps.append(None)
            continue

        if qml.math.allclose(d, 0):
            vjp = qml.math.convert_like(np.zeros([num_params]), d)
            getattr(vjps, reduction)(vjp)
            continue

        for fn, res_len in zip(processing_fns[t], reshape_info[t]):
            # extract the correct results from the flat list
            res = results[start : start + res_len]
            start += res_len

            # postprocess results to compute the gradient
            jac.append(fn(res))

        getattr(vjps, reduction)(_vector_jacobian_product(d, jac))

    return vjps
