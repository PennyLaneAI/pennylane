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
from functools import partial
from itertools import product

import numpy as np

import pennylane.ops as qops
from pennylane import math
from pennylane.core.qscript import QuantumScript, QuantumScriptBatch
from pennylane.measurements import shadow_expval
from pennylane.queuing import AnnotatedQueue, apply
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn

from .classical_shadow import ClassicalShadow


@transform
def _replace_obs(
    tape: QuantumScript, obs, *args, **kwargs
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """
    Tape transform to replace the measurement processes with the given one
    """
    with AnnotatedQueue() as q:
        # queue everything from the old tape except the measurement processes
        for op in tape.operations:
            apply(op)

        # queue the new observable
        obs(*args, **kwargs)
    qscript = QuantumScript.from_queue(q, shots=tape.shots)

    def processing_fn(res):
        return res[0]

    return [qscript], processing_fn


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
        for obs in product([qops.Identity, qops.X, qops.Y, qops.Z], repeat=len(w)):
            # Perform tensor product (((P_i @ P_j) @ P_k ) @ ....)
            observables.append(qops.prod(*(ob(wire) for ob, wire in zip(obs, w, strict=True))))
        all_observables.extend(observables)

    tapes, _ = _replace_obs(tape, shadow_expval, all_observables)

    def post_processing_fn(results):
        """Post process the classical shadows."""
        results = results[0]
        # cast to complex
        results = math.cast(results, np.complex64)

        states = []
        start = 0
        for w in wires_list:
            # reconstruct the state given the observables and the expectations of
            # those observables

            obs_matrices = math.stack(
                [
                    math.cast_like(math.convert_like(qops.functions.matrix(obs), results), results)
                    for obs in all_observables[start : start + 4 ** len(w)]
                ]
            )

            s = math.einsum(
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
        shadow = ClassicalShadow(bits, recipes)

        states = [math.mean(shadow.global_snapshots(wires=w), 0) for w in wires_list]
        return states if isinstance(wires[0], list) else states[0]

    return [tape], post_processing


@partial(transform, final_transform=True)
def shadow_state(
    tape: QuantumScript, wires, diffable=False
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transform a circuit returning a classical shadow into one that returns
    the reconstructed state in a differentiable manner.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        wires (list[int] or list[list[int]]): If a list of ints, this represents
            the wires over which to reconstruct the state. If a list of list of ints,
            a state is reconstructed for every element of the outer list, saving
            qfunc evaluations.
        diffable (bool): If True, reconstruct the state in a differentiable
            fashion, where the gradient of the reconstructed state approaches
            the gradient of the true state in expectation. This comes at a performance
            cost.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qp.transform <pennylane.transform>`. Executing this circuit
        will provide the reconstructed state in the form of a tensor.

    **Example**

    .. code-block:: python

        import numpy as onp
        from pennylane import numpy as np

        dev = qp.device("default.qubit", wires=2, seed=42)

        @qp.set_shots(shots=10000)
        @qp.shadows.shadow_state(wires=[0, 1], diffable=True)
        @qp.qnode(dev)
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=0)
            return qp.classical_shadow(wires=[0, 1], seed=99)

    >>> x = np.array(1.2, requires_grad=True)
    >>> onp.random.seed(123)
    >>> circuit(x)
    array([[ 3.2582501e-01+0.j      , -8.2500000e-03+0.238425j,
             2.9999996e-04+0.230925j,  3.3142501e-01+0.009j   ],
           [-8.2500000e-03-0.238425j,  1.6337499e-01+0.j      ,
             1.5997499e-01-0.0099j  , -5.5499999e-03-0.251475j],
           [ 2.9999996e-04-0.230925j,  1.5997499e-01+0.0099j  ,
             1.5797499e-01+0.j      ,  2.9999996e-04-0.241275j],
           [ 3.3142501e-01-0.009j   , -5.5499999e-03+0.251475j,
             2.9999996e-04+0.241275j,  3.5282502e-01+0.j      ]],
          dtype=complex64)
    >>> onp.random.seed(123)
    >>> qp.jacobian(lambda x: np.real(circuit(x)))(x)
    array([[-2.30550e-01,  5.62500e-03,  2.25000e-04, -2.29725e-01],
           [ 5.62500e-03,  2.30550e-01,  2.29725e-01, -1.57500e-03],
           [ 2.25000e-04,  2.29725e-01,  2.37900e-01,  2.47500e-03],
           [-2.29725e-01, -1.57500e-03,  2.47500e-03, -2.37900e-01]])
    """
    tapes, fn = (
        _shadow_state_diffable(tape, wires) if diffable else _shadow_state_undiffable(tape, wires)
    )
    return tapes, fn
