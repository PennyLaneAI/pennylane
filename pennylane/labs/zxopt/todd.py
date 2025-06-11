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
"""Third order duplicate and destroy (TODD) optimization method from pyzx, using ZX calculus."""

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
def todd(
    tape: QuantumScript, pre_optimize: bool = True, post_optimize: bool = True
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""

    Apply Third Order Duplicate and Destroy (TODD) by means of `zx.phase_block_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.phase_block_optimize>`__
    to a PennyLane `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuit.

    After running `TODD <https://arxiv.org/abs/1712.01557>`__, this pipeline uses `parity synthesis <https://arxiv.org/abs/1712.01859>`__ to synthesize the optimized phase polynomial.
    A final :func:`~basic_optimization` is applied to the final circuit (can be optionally turned off via the ``post_optimize`` argument).

    When there are continuous rotation gates such as :class:`~RZ`, we suggest to use :func:`~full_reduce`.

    Args:
        tape (QNode or QuantumTape or Callable): Input PennyLane circuit. This circuit has to be in the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ basis.
        pre_optimize (bool): Whether or not to call :func:`~basic_optimization` first. Default is ``True``.
        pre_optimize (bool): Whether or not to call :func:`~basic_optimization` after TODD. Default is ``True``.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: Improved PennyLane circuit. See :func:`qml.transform <pennylane.transform>` for the different output formats depending on the input type.

    .. seealso:: :func:`~full_reduce`, :func:`~full_optimize`, :func:`~basic_optimization`

    **Example**

    Let us optimize a circuit with :class:`~T` gates.

    .. code-block:: python

        from pennylane.labs.zxopt import todd

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

        (new_circ,), _ = todd(circ)
        print(f"Circuit after phase_block_optimize:")
        print(qml.drawer.tape_text(new_circ, wire_order=range(4)))

    .. code-block::

        Circuit before:
        0: ─╭●──T──T──────────╭●─┤
        1: ─╰X──T─╭●────╭●──T─╰X─┤
        2: ─╭X────╰X──T─╰X─╭X────┤
        3: ─╰●─────────────╰●────┤

        Circuit after phase_block_optimize:
        0: ──S─╭X──S†─╭X─╭X──Z──T─╭X─╭X─╭X─┤
        1: ────╰●─────│──│────────╰●─│──│──┤
        2: ──Z────────╰●─│───────────╰●─│──┤
        3: ──Z───────────╰●─────────────╰●─┤

    The original five T gates are reduced to just one.

    """
    if not has_zx:  # pragma: no cover
        raise ImportError(
            "The package pyzx is required by todd. You can install it with pip install pyzx"
        )  # pragma: no cover

    pyzx_circ = _tape2pyzx(tape)

    pyzx_circ = zx.phase_block_optimize(pyzx_circ, pre_optimize=pre_optimize)

    if post_optimize:
        pyzx_circ = zx.basic_optimization(pyzx_circ)

    pl_circ = qml.transforms.from_zx(pyzx_circ.to_graph())

    return [pl_circ], null_postprocessing
