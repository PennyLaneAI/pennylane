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
"""Optimization pass ``basic_optimization`` from pyzx using ZX calculus."""

import pyzx as zx

import pennylane as qml

from .zx_conversion import _tape2pyzx


def basic_optimization(tape, verbose=False):
    r"""
    Apply `zx.basic_optimization <https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization>`__ to a PennyLane `phase polynomial <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__ circuit with :class:`~Hadamard` gates.
    This step can help improve phase polynomial based optimization schemes like :func:`~todd` or :func:`~full_optimize` by moving :class:`~Hadamard` to create big and few phase polynomial blocks.

    Args:
        tape (qml.tape.QuantumScript): Input PennyLane circuit.
        verbose (bool): whether or not to print new T gate and two-qubit gate count, as well as draw the diagram before and after the optimization. Default is `False`.

    Returns:
        qml.tape.QuantumScript: T-gate optimized PennyLane circuit.

    .. seealso:: :func:`~full_reduce` (arbitrary circuits), :func:`~full_optimize` (`(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuits)

    **Example**

    This pass tried to push :class:`~Hadamard` gates as far as possible to the side to allow better phase polynomial optimization via, e.g., :func:`~todd`.

    .. code-block:: python

        from pennylane.labs.zxopt import basic_optimization
        circ = qml.tape.QuantumScript([
            qml.CNOT((0, 1)),
            qml.T(0),
            qml.CNOT((3, 2)),
            qml.Hadamard(0),
            qml.T(1),
            qml.Hadamard(1),
            qml.CNOT((1, 2)),
            qml.Hadamard(2),
            qml.T(2),
            qml.Hadamard(2),
            qml.RZ(0.5, 1),
            qml.CNOT((1, 2)),
            qml.T(1),
            qml.CNOT((3, 2)),
            qml.Hadamard(1),
            qml.T(0),
            qml.CNOT((0, 1)),
        ], [])

        print(f"Circuit before:")
        print(qml.drawer.tape_text(circ, wire_order=range(4)))

        new_circ = basic_optimization(circ)
        print(f"Circuit after basic_optimization:")
        print(qml.drawer.tape_text(new_circ, wire_order=range(4)))

    .. code-block::

        Circuit before:
        0: ─╭●──T──H──T────────────────────╭●─┤
        1: ─╰X──T──H─╭●──RZ───────╭●──T──H─╰X─┤
        2: ─╭X───────╰X──H───T──H─╰X─╭X───────┤
        3: ─╰●───────────────────────╰●───────┤

        Circuit after basic_optimization:
        0: ──T─╭●───────────╭X──H──T─┤
        1: ────╰X──T──H──RZ─╰●──H────┤
        2: ─╭●───────╭Z──────────────┤
        3: ─╰X──H──T─╰●──H───────────┤

    """
    pyzx_circ = _tape2pyzx(tape)

    pyzx_circ = pyzx_circ.to_basic_gates()

    pyzx_circ = zx.basic_optimization(pyzx_circ, quiet=not verbose)

    pl_circ = qml.transforms.from_zx(pyzx_circ.to_graph())

    return pl_circ
