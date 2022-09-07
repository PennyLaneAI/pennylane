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
from functools import reduce, wraps
import warnings

import pennylane as qml
import pennylane.numpy as np


@qml.batch_transform
def __replace_obs(tape, obs, *args, **kwargs):
    """
    Tape transform to replace the measurement processes with the given one
    """
    # check if the measurement process of the tape is qml.classical_shadow
    for o in tape.observables:
        if o.return_type is not qml.measurements.Shadow:
            raise ValueError(
                f"Tape measurement must be {qml.measurements.Shadow!r}, got {o.return_type!r}"
            )

    with qml.tape.QuantumTape() as new_tape:
        # queue everything from the old tape except the measurement processes
        for op in tape.operations:
            qml.apply(op)

        # queue the new observable
        obs(*args, **kwargs)

    def processing_fn(res):
        return qml.math.squeeze(qml.math.stack(res))

    return [new_tape], processing_fn


def shadow_expval(H, k=1):
    """Transform a QNode returning a classical shadow into one that returns
    the approximate expectation values in a differentiable manner.

    See :func:`~.pennylane.shadow_expval` for more usage details.

    Args:
        H (:class:`~.pennylane.Observable` or list[:class:`~.pennylane.Observable`]): Observables
            for which to compute the expectation values
        k (int): k (int): Number of equal parts to split the shadow's measurements to compute
            the median of means. ``k=1`` corresponds to simply taking the mean over all measurements.

    Returns:
        tensor-like[float]: 1-D tensor containing the expectation value estimates for each observable

    **Example**

    .. code-block:: python3

        H = qml.PauliZ(0) @ qml.PauliZ(1)
        dev = qml.device("default.qubit", wires=2, shots=10000)

        @qml.shadows.expval(H, k=1)
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.classical_shadow(wires=[0, 1])

    >>> x = np.array(1.2)
    >>> circuit(x)
    tensor(0.3069, requires_grad=True)
    >>> qml.grad(circuit)(x)
    -0.9323999999999998
    """

    def decorator(qnode):
        return wraps(qnode)(__replace_obs(qnode, qml.shadow_expval, H, k=k))

    return decorator


def _shadow_state_diffable(wires):
    """Differentiable version of the shadow state transform"""
    wires_list = [wires] if not isinstance(wires[0], list) else wires

    if any(len(w) >= 8 for w in wires_list):
        warnings.warn(
            "Differentiable state reconstruction for more than 8 qubits is not recommended",
            UserWarning,
        )

    # all pauli observables
    all_observables = []
    for w in wires_list:
        observables = []
        # Create all combinations of possible Pauli products P_i P_j P_k.... for w wires
        for obs in product(
            *[[qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ] for _ in range(len(w))]
        ):
            # Perform tensor product (((P_i @ P_j) @ P_k ) @ ....)
            observables.append(reduce(lambda a, b: a @ b, [ob(wire) for ob, wire in zip(obs, w)]))
        all_observables.extend(observables)

    def decorator(qnode):
        new_qnode = __replace_obs(qnode, qml.shadow_expval, all_observables)

        @wraps(qnode)
        def wrapper(*args, **kwargs):
            # pylint: disable=not-callable
            results = new_qnode(*args, **kwargs)

            # cast to complex
            results = qml.math.cast(results, np.complex64)

            states = []
            start = 0
            for w in wires_list:
                # reconstruct the state given the observables and the expectations of
                # those observables

                obs_matrices = qml.math.stack(
                    [
                        qml.math.cast_like(qml.math.convert_like(qml.matrix(obs), results), results)
                        for obs in all_observables[start : start + 4 ** len(w)]
                    ]
                )

                s = qml.math.einsum(
                    "a,abc->bc", results[start : start + 4 ** len(w)], obs_matrices
                ) / 2 ** len(w)
                states.append(s)

                start += 4 ** len(w)

            return states[0] if not isinstance(wires[0], list) else states

        return wrapper

    return decorator


def _shadow_state_undiffable(wires):
    """Non-differentiable version of the shadow state transform"""
    wires_list = [wires] if not isinstance(wires[0], list) else wires

    def decorator(qnode):
        @wraps(qnode)
        def wrapper(*args, **kwargs):
            bits, recipes = qnode(*args, **kwargs)
            shadow = qml.shadows.ClassicalShadow(bits, recipes)

            states = [qml.math.mean(shadow.global_snapshots(wires=w), 0) for w in wires_list]
            return states[0] if not isinstance(wires[0], list) else states

        return wrapper

    return decorator


def shadow_state(wires, diffable=False):
    """Transform a QNode returning a classical shadow into one that returns
    the reconstructed state in a differentiable manner.

    Args:
        wires (list[int] or list[list[int]]): If a list of ints, this represents
            the wires over which to reconstruct the state. If a list of list of ints,
            a state is reconstructed for every element of the outer list, saving
            qfunc evaluations.
        diffable (bool): If True, reconstruct the state in a differentiable
            fashion, where the gradient of the reconstructed state approaches
            the gradient of the true state in expectation. This comes at a performance
            cost.

    Returns:
        list[tensor-like[complex]]: The reconstructed states

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2, shots=10000)

        @qml.shadows.state(wires=[0, 1], diffable=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.classical_shadow(wires=[0, 1])

    >>> x = np.array(1.2)
    >>> circuit(x)
    tensor([[ 0.33714998+0.j        ,  0.007875  +0.228825j  ,
             -0.010575  +0.22642499j,  0.33705002+0.01125j   ],
            [ 0.007875  -0.228825j  ,  0.16104999+0.j        ,
              0.17055   -0.0126j    ,  0.011025  -0.232575j  ],
            [-0.010575  -0.22642499j,  0.17055   +0.0126j    ,
              0.16704999+0.j        , -0.006075  -0.225225j  ],
            [ 0.33705002-0.01125j   ,  0.011025  +0.232575j  ,
             -0.006075  +0.225225j  ,  0.33475   +0.j        ]],
           dtype=complex64, requires_grad=True)
    >>> qml.jacobian(circuit)(x)
    array([[-0.245025, -0.005325,  0.004275, -0.2358  ],
           [-0.005325,  0.235275,  0.2358  , -0.004275],
           [ 0.004275,  0.2358  ,  0.244875, -0.002175],
           [-0.2358  , -0.004275, -0.002175, -0.235125]])
    """
    return _shadow_state_diffable(wires) if diffable else _shadow_state_undiffable(wires)
