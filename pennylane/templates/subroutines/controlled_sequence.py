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
r"""
Contains the ControlledSequence template.
"""

from collections import Counter

from pennylane import capture, compiler, math
from pennylane.capture.op_eqns import pop_op_eqns
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_name, register_resources
from pennylane.ops.op_math import SymbolicOp2, ctrl, pow
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Wire
from pennylane.wires import Wires, validate_no_wire_overlaps


class ControlledSequence(SymbolicOp2):
    r"""Creates a sequence of controlled gates raised to decreasing powers of 2. Can be used as
    a sub-block in building a `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__
    circuit.

    Given an :class:`~.Operator` and a list of control wires, this template creates a sequence of
    controlled gates, one for each control wire, with the base :class:`~.Operator` raised to
    decreasing powers of 2:

    .. figure:: ../../_static/templates/subroutines/big_ctrl.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    Args:
        base (Operator): the phase estimation unitary, specified as an :class:`~.Operator`
        control (Union[Wires, Sequence[int], or int]): the wires to be used for control

    Raises:
        ValueError: if the wires in ``control`` and wires on the ``base`` operator share a common
            element

    .. seealso:: :class:`~.QuantumPhaseEstimation`

    **Examples**

    .. code-block:: python

        dev = qp.device("default.qubit")

        @qp.transforms._tape_decompose(max_expansion=1)
        @qp.qnode(dev)
        def circuit():
            qp.ControlledSequence(qp.RX(0.25, wires = 3), control = [0, 1, 2])
            return qp.state()

    >>> print(qp.draw(circuit, wire_order=[0,1,2,3])())
    0: ─╭●────────────────────────────┤ ╭State
    1: ─│─────────╭●──────────────────┤ ├State
    2: ─│─────────│─────────╭●────────┤ ├State
    3: ─╰RX(1.00)─╰RX(0.50)─╰RX(0.25)─┤ ╰State


    .. code-block:: python

        dev = qp.device("default.qubit", wires = 4)

        @qp.qnode(dev)
        def circuit():

            for i in range(3):
                qp.Hadamard(wires = i)

            qp.ControlledSequence(qp.RX(0.25, wires = 3), control = [0, 1, 2])

            qp.adjoint(qp.QFT)(wires = range(3))

            return qp.probs(wires = range(3))

    >>> print(circuit()) # doctest: +SKIP
    [0.9206 0.0264 0.0073 0.0042 0.0036 0.0042 0.0073 0.0264]

    """

    wire_argnames = ("control",)
    hybrid_argnames = ("base",)
    grad_method = None

    def __init__(self, base, control):
        control_wires = Wires(control)
        base_wires = getattr(base, "wires", ())

        wire_args = {
            "base.wires": base_wires,
            "control_wires": control_wires,
        }

        validate_no_wire_overlaps(wire_args)

        self._name = "ControlledSequence"

        super().__init__(base, control)

    def __hash__(self):
        return hash(
            (
                str(self.name),
                self.control,
                hash(self.base),
            )
        )

    @property
    def data(self):
        return self.base.data

    @property
    def ndim_params(self):
        return self.base.ndim_params

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def has_sparse_matrix(self):
        return False

    @property
    def has_matrix(self):
        return False

    def __repr__(self):
        return f"ControlledSequence({self.base}, control={list(self.control)})"


def _ctrl_seq_standard_resources(base, control) -> dict:
    control = Wires(control)
    base_rep = abstractify(base)

    if not control:
        return {}
    controlled_base = _ctrl_abstract(base_rep, Wire[1])
    return dict(
        Counter(abstractify(pow(controlled_base, 2**i, lazy=False)) for i in range(len(control)))
    )


@register_name("standard")
@register_resources(_ctrl_seq_standard_resources)
def _ctrl_seq_standard(*_, base=None, control=None, **__):
    if compiler.active() or capture.enabled():
        control = math.array(control, like="jax")

    num_controls = len(control)

    for i in range(num_controls - 1, -1, -1):
        j = num_controls - 1 - i
        ctrl_wire = control[j]

        controlled_base = ctrl(base, control=ctrl_wire)
        if capture.enabled():
            pop_op_eqns((controlled_base,))
        pow(controlled_base, 2**i, lazy=False)


add_decomps(ControlledSequence, _ctrl_seq_standard)
