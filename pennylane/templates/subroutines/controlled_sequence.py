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

from pennylane.ops.op_math import SymbolicOp2
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

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.ControlledSequence
            :stacked:
            :symbols: :math:`U=` ``base`` and :math:`n` control wires
            :example: :math:`n=3`

            .. decomposition-rule:: controlled_powers
                :image: /_static/decomposition_thumbnails/controlled_sequence_powers.svg
                :where: Powers are simplified eagerly when the base operator supports it.

                :math:`U^{2^k}` with 1 control | :math:`n` (one per :math:`k < n`) | 3

            .. decomposition-rule:: looped_controlled_repetitions
                :image: /_static/decomposition_thumbnails/controlled_sequence_repetitions.svg
                :where: Nested loops keep the circuit representation compact. The image shows
                    the operations executed for three control wires.

                :math:`U` with 1 control | :math:`2^n - 1` | 7

    **Examples**

    .. code-block:: python

        dev = qp.device("default.qubit")

        @qp.transforms._tape_decompose(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, max_expansion=1)
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
