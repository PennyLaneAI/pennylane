# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the qp.iterative_qpe function.
"""

import numpy as np

from pennylane import capture
from pennylane import ops as pl_ops
from pennylane.core.operator.operator2 import pop_op_eqns  # tach-ignore


def _iterative_qpe(base, aux_wire, iters):
    """The rounds of iterative QPE.

    Notes regarding implementation,

    * Static Argument: 'iters' must be a concrete value known at trace time,
                        as it dictates the shape of the returned measurements
    ^ Python Loops: Standard for loops are used instead of 'qp.for_loop'
                    as 'Pow2' expects a static, concrete, compile-time constant.

    """

    measurements = []

    for i in range(iters):
        pl_ops.Hadamard(aux_wire)
        pl_ops.ctrl(pl_ops.pow(base, z=2 ** (iters - i - 1)), control=aux_wire)

        # Apply phase corrections based on previous bit measurements
        for j in range(i):
            meas = measurements[j]

            def cond_func(j=j):
                pl_ops.PhaseShift(-2.0 * np.pi / (2 ** (j + 2)), wires=aux_wire)

            pl_ops.cond(meas, cond_func)()

        pl_ops.Hadamard(aux_wire)
        # Measure and reset auxiliary wire to reuse for next iteration
        measurements.insert(0, pl_ops.measure(wires=aux_wire, reset=True))

    return measurements


_iterative_qpe_subroutine = capture.subroutine(_iterative_qpe, static_argnames="iters")


def iterative_qpe(base, aux_wire, iters):
    r"""Performs the `iterative quantum phase estimation <https://arxiv.org/pdf/quant-ph/0610214.pdf>`_ circuit.

    Given a unitary :math:`U`, this function applies the circuit for iterative quantum phase
    estimation and returns a list of mid-circuit measurements with qubit reset.

    Args:
        base (Operator): the phase estimation unitary, specified as an :class:`~.Operator`
        aux_wire (Union[Wires, int, str]): the wire to be used for the estimation
        iters (int): the number of measurements to be performed

    Returns:
        list[MeasurementValue]: the abstract results of the mid-circuit measurements

    .. seealso:: :class:`~.QuantumPhaseEstimation`, :func:`~.measure`

    **Example**

    .. code-block:: python

        dev = qp.device("default.qubit", seed=42)

        @qp.set_shots(5)
        @qp.qnode(dev)
        def circuit():

            # Initial state
            qp.X(0)

            # Iterative QPE
            measurements = qp.iterative_qpe(qp.RZ(2.0, wires=[0]), aux_wire=1, iters=3)

            return qp.sample(measurements)

    >>> result = circuit()
    >>> assert result.shape == (5, 3)
    >>> print(result)
    [[0 0 1]
     [0 0 1]
     [0 0 1]
     [0 0 1]
     [0 0 1]]

    The output is an array of size ``(number of shots, number of iterations)``.

    >>> print(qp.draw(circuit, max_length=150)())
    0: ──X─╭RZ(2.00)⁴─────────────────╭RZ(2.00)²────────────────────────────╭RZ(2.00)¹────────────────────────────────────┤
    1: ──H─╰●──────────H──┤↗│  │0⟩──H─╰●──────────Rϕ(-1.57)──H──┤↗│  │0⟩──H─╰●──────────Rϕ(-1.57)──Rϕ(-0.79)──H──┤↗│  │0⟩─┤
                           ╚══════════════════════╩══════════════║══════════════════════║══════════╩══════════════║═══════╡ ╭Sample[MCM]
                                                                 ╚══════════════════════╩═════════════════════════║═══════╡ ├Sample[MCM]
                                                                                                                  ╚═══════╡ ╰Sample[MCM]
    """

    if not capture.enabled():
        return _iterative_qpe(base, aux_wire, iters)

    # NOTE: Guard so that operator1 instances still work here
    if getattr(base, "tracer", None) is not None:
        pop_op_eqns((base,))

    return _iterative_qpe_subroutine(base, aux_wire, iters)
