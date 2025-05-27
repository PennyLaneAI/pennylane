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

import pyzx as zx

import pennylane as qml

from .zx_conversion import _tape2pyzx


def todd(tape, pre_optimize=True, verbose=False):
    r"""

    Apply Third Order Duplicate and Destroy (TODD) by means of `zx.phase_block_optimize <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.phase_block_optimize>`__
     to a PennyLane `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuit.

    After `TODD <https://arxiv.org/abs/1712.01557>`__, this pipeline uses `parity synthesis <https://arxiv.org/abs/1712.01859>`__ to synthesize the optimized phase polynomial.

    When there are continuous rotation gates such as :class:`~RZ`, we suggest to use :func:`~full_reduce`.

    Args:
        tape (qml.tape.QuantumScript): Input PennyLane circuit. This circuit has to be in the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ basis.
        pre_optimize (bool): Whether or not to call :func:`~basic_optimization` first. Default is True.
        verbose (bool): Whether or not to print the new T gate and two-qubit gate count, as well as draw the diagram before and after the optimization. Default is `False`.

    Returns:
        qml.tape.QuantumScript: T-gate optimized PennyLane circuit.

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

        new_circ = todd(circ)
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

    pyzx_circ = _tape2pyzx(tape)

    pyzx_circ = zx.phase_block_optimize(pyzx_circ, pre_optimize=pre_optimize, quiet=not verbose)

    pyzx_circ = zx.basic_optimization(pyzx_circ)
    pl_circ = qml.transforms.from_zx(pyzx_circ.to_graph())

    return pl_circ
