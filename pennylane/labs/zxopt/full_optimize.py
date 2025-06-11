# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimization pass ``full_optimize`` from pyzx using ZX calculus."""
import warnings

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .util import _tape2pyzx

has_zx = True
try:
    import pyzx as zx

except ImportError:
    has_zx = False


def null_postprocessing(results):
    """A postprocesing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def full_optimize(
    tape: QuantumScript, clifford_t_args: dict = None
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""

    Full optimization pipeline applying `TODD <https://arxiv.org/abs/1712.01557>`__ and ZX-based T gate reduction to a PennyLane `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuit.

    This function applies `zx.full_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.full_optimize>`__ and is basically a combination of :func:`~todd` and :func:`~full_reduce`.

    When there are continuous rotation gates such as :class:`~RZ`, we suggest to use :func:`~full_reduce`. Otherwise, :func:`~clifford_t_decomposition` is used to decompose the circuit to the (Clifford + T) gate set.

    Args:
        tape (QNode or QuantumTape or Callable): Input PennyLane circuit. This circuit has to be in the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ gate set.
        clifford_t_args (dict): Optional arguments to be passed to :func:`~clifford_t_decomposition` when a circuit with continuous gates is passed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: T gate optimized PennyLane circuit. See :func:`qml.transform <pennylane.transform>` for the different output formats depending on the input type.

    .. seealso:: :func:`~full_reduce`, :func:`~todd`

    **Example**

    Let us optimize a circuit with :class:`~T` gates.

    .. code-block:: python

        from pennylane.labs.zxopt import full_optimize

        circ = qml.tape.QuantumScript([
            qml.CNOT((0, 1)),
            qml.T(0),
            qml.CNOT((3, 2)),
            qml.T(1),
            qml.CNOT((1, 2)),
            qml.T(2),
            qml.CNOT((1, 2)),
            qml.T(1),
            qml.CNOT((3, 2)),
            qml.T(0),
            qml.CNOT((0, 1)),
        ], [])

        print(f"Circuit before:")
        print(qml.drawer.tape_text(circ, wire_order=range(4)))

        (new_circ,), _ = full_optimize(circ)
        print(f"Circuit after full_optimize:")
        print(qml.drawer.tape_text(new_circ, wire_order=range(4)))

    .. code-block::

        Circuit before:
        0: ─╭●──T──T──────────╭●─┤
        1: ─╰X──T─╭●────╭●──T─╰X─┤
        2: ─╭X────╰X──T─╰X─╭X────┤
        3: ─╰●─────────────╰●────┤

        Circuit after full_optimize:
        0: ──S─╭X──S†─╭X─╭X──Z──T─╭X─╭X─╭X─┤
        1: ────╰●─────│──│────────╰●─│──│──┤
        2: ──Z────────╰●─│───────────╰●─│──┤
        3: ──Z───────────╰●─────────────╰●─┤

    The original five T gates are reduced to just one.

    .. details::
        :title: Usage Details

        There is the option to pass circuits that are not in the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ gate set. Those circuits will be first decomposed using :func:`~clifford_t_decomposition`.
        We can pass optional keyword arguments to it via the ``clifford_t_args`` argument in the following way.

        .. code-block:: python

            circ = qml.tape.QuantumScript(
                [
                    qml.CNOT((0, 1)),
                    qml.T(0),
                    qml.RZ(0.5, 0),
                    qml.Hadamard(0),
                ],
                [],
            )

            new_circ = full_optimize(circ, clifford_t_args = {"epsilon": 0.1})

    """
    if not has_zx:  # pragma: no cover
        raise ImportError(
            "full_optimize requires the package pyzx. "
            "You can install it with pip install pyzx"
        )  # pragma: no cover

    try:
        pyzx_circ = _tape2pyzx(tape)

        pyzx_circ = zx.full_optimize(pyzx_circ)

    except TypeError:

        if clifford_t_args is None:
            warnings.warn(
                "Input circuit is not in the (Clifford + T) basis, will attempt to decompose using qml.clifford_t_decomposition."
            )
            clifford_t_args = {}

        (tape,), _ = qml.clifford_t_decomposition(tape, **clifford_t_args)
        pyzx_circ = _tape2pyzx(tape)

        pyzx_circ = zx.full_optimize(pyzx_circ)

    pl_circ = qml.transforms.from_zx(pyzx_circ.to_graph())

    return [pl_circ], null_postprocessing
