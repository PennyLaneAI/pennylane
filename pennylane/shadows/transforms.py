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

import warnings
from functools import reduce, partial
from itertools import product
from typing import Sequence, Callable

import pennylane as qml
import pennylane.numpy as np
from pennylane.measurements import ClassicalShadowMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.core import transform


@transform
def _replace_obs(tape: QuantumTape, obs, *args, **kwargs) -> (Sequence[QuantumTape], Callable):
    """
    Tape transform to replace the measurement processes with the given one
    """
    # check if the measurement process of the tape is qml.classical_shadow
    for m in tape.measurements:
        if not isinstance(m, ClassicalShadowMP):
            raise ValueError(
                f"Tape measurement must be ClassicalShadowMP, got {m.__class__.__name__!r}"
            )

    with qml.queuing.AnnotatedQueue() as q:
        # queue everything from the old tape except the measurement processes
        for op in tape.operations:
            qml.apply(op)

        # queue the new observable
        obs(*args, **kwargs)
    qscript = QuantumScript.from_queue(q, shots=tape.shots)

    def processing_fn(res):
        return res[0]

    return [qscript], processing_fn


@partial(transform, final_transform=True)
def shadow_expval(tape: QuantumTape, H, k=1) -> (Sequence[QuantumTape], Callable):
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

        @qml.shadows.shadow_expval(H, k=1)
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
    tapes, _ = _replace_obs(tape, qml.shadow_expval, H, k=k)

    def post_processing_fn(res):
        return res

    return tapes, post_processing_fn


def _shadow_state_diffable(tape, wires):
    """Differentiable version of the shadow state transform"""
    wires_list = wires if isinstance(wires[0], list) else [wires]

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

    tapes, _ = _replace_obs(tape, qml.shadow_expval, all_observables)

    def post_processing_fn(results):
        """Post process the classical shadows."""
        results = results[0]
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

        return states if isinstance(wires[0], list) else states[0]

    return tapes, post_processing_fn


def _shadow_state_undiffable(tape, wires):
    """Non-differentiable version of the shadow state transform"""
    wires_list = wires if isinstance(wires[0], list) else [wires]

    def post_processing(results):
        bits, recipes = results[0]
        shadow = qml.shadows.ClassicalShadow(bits, recipes)

        states = [qml.math.mean(shadow.global_snapshots(wires=w), 0) for w in wires_list]
        return states if isinstance(wires[0], list) else states[0]

    return [tape], post_processing


@partial(transform, final_transform=True)
def shadow_state(tape: QuantumTape, wires, diffable=False) -> (Sequence[QuantumTape], Callable):
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

        @qml.shadows.shadow_state(wires=[0, 1], diffable=True)
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
    >>> qml.jacobian(lambda x: np.real(circuit(x)))(x)
    array([[-0.245025, -0.005325,  0.004275, -0.2358  ],
           [-0.005325,  0.235275,  0.2358  , -0.004275],
           [ 0.004275,  0.2358  ,  0.244875, -0.002175],
           [-0.2358  , -0.004275, -0.002175, -0.235125]])
    """
    tapes, fn = (
        _shadow_state_diffable(tape, wires) if diffable else _shadow_state_undiffable(tape, wires)
    )
    return tapes, fn
