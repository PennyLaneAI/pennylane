# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classical shadow transforms"""

from itertools import product
from functools import reduce

import pennylane as qml
import pennylane.numpy as np


@qml.batch_transform
def __replace_obs(tape, obs, *args, **kwargs):
    # construct a new tape with everything except the measurement process
    with qml.tape.QuantumTape() as new_tape:
        for op in tape.operations:
            qml.apply(op)

        # queue the new observable
        obs(*args, **kwargs)

    def processing_fn(res):
        return qml.math.squeeze(qml.math.stack(res))

    return [new_tape], processing_fn


def shadow_expval(H, k=1):
    """TODO: docs"""

    def decorator(qnode):
        def wrapper(*args, **kwargs):
            new_qnode = __replace_obs(qnode, qml.shadow_expval, H, k=k)
            return new_qnode(*args, **kwargs)

        return wrapper

    return decorator


def shadow_state(wires):
    """TODO: docs"""

    # all pauli observables
    observables = []
    for obs in product(
        *[[qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ] for _ in range(len(wires))]
    ):
        observables.append(reduce(lambda a, b: a @ b, [ob(wire) for ob, wire in zip(obs, wires)]))

    def decorator(qnode):
        new_qnode = __replace_obs(qnode, qml.shadow_expval, observables)

        def wrapper(*args, **kwargs):
            results = new_qnode(*args, **kwargs)

            # cast to complex
            results = qml.math.cast(results, np.complex64)

            # reconstruct the state given the observables and the expectations of
            # those observables
            state = 0
            for res, obs in zip(results, observables):
                state = state + res * qml.math.cast_like(
                    qml.math.convert_like(qml.matrix(obs), res), res
                )
            state = state / 2 ** len(wires)

            return state

        return wrapper

    return decorator
