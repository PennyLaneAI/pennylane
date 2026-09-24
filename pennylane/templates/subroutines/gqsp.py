# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the GQSP template.
"""

from pennylane import capture, compiler, math, ops
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.controlled2 import _ctrl_abstract, _validate_work_wire_type
from pennylane.typing import Float, Wire
from pennylane.wires import Wires


class GQSP(Operator2):
    r"""
    Implements the generalized quantum signal processing (GQSP) circuit.

    This operation encodes a polynomial transformation of an input unitary operator following
    the algorithm described in `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`__ as:

    .. math::
        U
        \xrightarrow{GQSP}
        \begin{pmatrix}
        \text{poly}(U) & * \\
        * & * \\
        \end{pmatrix}

    The implementation requires one control qubit.

    Args:

        unitary (Operator): the operator to be encoded by the GQSP circuit
        angles (tensor[float]): array of angles defining the polynomial transformation. The
            shape of the array must be `(3, d+1)`, where `d` is the degree of the polynomial.
        control (Union[Wires, int, str]): control qubit used to encode the polynomial
            transformation

    .. note::

        The :func:`~.poly_to_angles` function can be used to calculate the angles for a
        given polynomial.

    Example:

    .. code-block:: python

        # P(x) = 0.1 + 0.2j x + 0.3 x^2
        poly = [0.1, 0.2j, 0.3]

        angles = qp.poly_to_angles(poly, "GQSP")

        @qp.prod # transforms the qfunc into an Operator
        def unitary(wires):
            qp.RX(0.3, wires)

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(angles):
            qp.GQSP(unitary(wires = 1), angles, control = 0)
            return qp.state()

        matrix = qp.matrix(circuit, wire_order=[0, 1])(angles)

    .. code-block:: pycon

        >>> print(np.round(matrix,3)[:2, :2])
        [[0.387+0.198j 0.03 -0.089j]
        [0.03 -0.089j 0.387+0.198j]]
    """

    dynamic_argnames = ("angles",)
    static_argnames = ("work_wire_type",)
    hybrid_argnames = ("unitary",)
    wire_argnames = ("control", "work_wires")

    arg_specs = {"angles": Float[3, -1], "control": Wire[1], "work_wires": Wire[-1]}

    def __init__(self, unitary, angles, control, work_wires=None, work_wire_type="borrowed"):
        # pylint: disable=too-many-arguments
        work_wires = Wires(()) if work_wires is None else work_wires
        _validate_work_wire_type(work_wire_type)
        if isinstance(angles, (list, tuple)):
            angles = math.stack(angles)
        super().__init__(unitary, angles, control, work_wires, work_wire_type)


def _GQSP_resources(
    unitary, angles, control, work_wires, work_wire_type
):  # pylint: disable=unused-argument
    num_iters = angles.shape[1]
    return {
        ops.X: 2 + 2 * (num_iters - 1),
        ops.U3: num_iters,
        ops.Z: num_iters,
        _ctrl_abstract(
            abstractify(unitary),
            Wire[1],
            num_zero_control_values=1,
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): num_iters
        - 1,
    }


@register_resources(_GQSP_resources)
def _GQSP_decomposition(unitary, angles, control, work_wires, work_wire_type):
    if compiler.active() or capture.enabled():
        angles = math.array(angles, like="jax")

    thetas, phis, lambdas = angles[0], angles[1], angles[2]

    # These four gates adapt PennyLane's ops.U3 to the chosen U3 format in the GQSP paper.
    ops.X(control)
    ops.U3(2 * thetas[0], phis[0], lambdas[0], wires=control)
    ops.X(control)
    ops.Z(control)

    @for_loop(1, len(thetas))
    def gqsp_loop(i):
        ops.ctrl(
            unitary,
            control=control,
            control_values=[0],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        ops.X(control)
        ops.U3(2 * thetas[i], phis[i], lambdas[i], wires=control)
        ops.X(control)
        ops.Z(control)

    gqsp_loop()  # pylint:disable= no-value-for-parameter


add_decomps(GQSP, _GQSP_decomposition)
