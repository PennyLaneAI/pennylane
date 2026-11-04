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

from copy import copy

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math import SymbolicOp2, ctrl
from pennylane.ops.op_math.controlled2 import _ctrl_abstract
from pennylane.typing import Wire
from pennylane.wires import Wires


class ControlledSequence(SymbolicOp2, Operator2):
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

    **Example**

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

        if (
            isinstance(base.wires, Wires)
            and isinstance(control_wires, Wires)
            and len(Wires.shared_wires([base.wires, control_wires])) != 0
        ):
            raise ValueError("The control wires must be different from the base operation wires.")

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
    def wires(self):
        return self.control + self.base.wires

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

    def map_wires(self, wire_map: dict):
        new_op = ControlledSequence(
            self.base.map_wires(wire_map=wire_map),
            Wires([wire_map.get(wire, wire) for wire in self.control]),
        )
        return new_op


def _ctrl_seq_decomposition_resources(base, control) -> dict:
    control = Wires(control)
    base_rep = abstractify(base)

    if not control:
        return {}
    return {_ctrl_abstract(base_rep, Wire[1]): 2 ** len(control) - 1}


# pylint: disable=no-value-for-parameter
@register_resources(_ctrl_seq_decomposition_resources)
def _ctrl_seq_decomposition(*_, base=None, control=None, **__):
    control = Wires(control)

    if compiler.active() or capture.enabled():
        control = math.array(control, like="jax")

    num_controls = len(control)

    @for_loop(num_controls - 1, -1, -1)
    def _powers_loop(i):
        j = num_controls - 1 - i
        ctrl_wire = control[j]

        # Note that we cannot use both Pow here and a for_loop on line 144
        # since the arg i which becomes z will be traced, and Pow expects a static
        # z. Opted for an inner for_loop since it is still compact, rather than
        # an unrolled outer python loop and Pows.
        @for_loop(2**i)
        def _repeat_loop(_):
            ctrl(base, control=ctrl_wire)

        _repeat_loop()

    _powers_loop()


add_decomps(ControlledSequence, _ctrl_seq_decomposition)
