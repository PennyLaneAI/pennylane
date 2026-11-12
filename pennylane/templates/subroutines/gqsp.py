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

import numpy as np

from pennylane import math
from pennylane.core.operator import Operator2
from pennylane.ops.op_math.controlled2 import _validate_work_wire_type
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
        U^{s}\,\text{poly}(U) & * \\
        * & * \\
        \end{pmatrix},

    where :math:`s` is an optional integer ``shift``. A negative ``shift`` allows encoding
    Laurent polynomials, i.e. polynomials that contain negative powers of :math:`U`.

    The implementation requires one control qubit.

    Args:

        unitary (Operator): the operator to be encoded by the GQSP circuit
        angles (tensor[float]): array of angles defining the polynomial transformation. The
            shape of the array must be `(3, d+1)`, where `d` is the degree of the polynomial.
        control (Union[Wires, int, str]): control qubit used to encode the polynomial
            transformation
        work_wires (WiresLike): auxiliary qubits that are passed to the controlled ``unitary``.
        work_wire_type (str): the type of the ``work_wires``. Must be ``"borrowed"`` (default)
            or ``"zeroed"``.
        shift (int): number of powers of :math:`U` by which the polynomial is shifted, such that
            the encoded operator is :math:`U^{s}\,\text{poly}(U)`. Default is ``0``.

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

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.GQSP
            :stacked:
            :symbols: :math:`U=` ``unitary``, polynomial degree :math:`d` (``angles`` of shape
                ``(3, d+1)``) and shift :math:`s`
            :example: :math:`d=1`, :math:`s=0`

            .. decomposition-rule:: _GQSP_decomposition
                :image: /_static/decomposition_thumbnails/gqsp_signal_processing.svg
                :where: :math:`a = \min(\max(-s, 0), d)` is the number of controlled unitaries
                    absorbing a negative shift, and :math:`[\cdot]` is 1 if the condition holds
                    and 0 otherwise

                :class:`~.X` | :math:`2(d+1)` | 4
                :class:`~.U3` | :math:`d+1` | 2
                :class:`~.Z` | :math:`d+1` | 2
                :math:`U` with 1 control on :math:`|0\rangle` | :math:`d-a` | 1
                :math:`U^\dagger` with 1 control | :math:`a` | 0
                :math:`U^{s}` | :math:`[s > 0]` | 0
                :math:`(U^\dagger)^{-s-d}` | :math:`[s < -d]` | 0

    .. details::
        :title: Usage Details

        **Shifting the polynomial**

        The ``shift`` argument multiplies the encoded polynomial by :math:`U^{s}`. Following
        Theorem 6 of `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`__, a negative shift
        is implemented at no extra cost: each of the first :math:`\min(|s|, d)` operators
        :math:`|0\rangle\langle 0| \otimes U + |1\rangle\langle 1| \otimes I` is replaced by
        :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U^{\dagger}`,
        which equals the original operator multiplied by :math:`I \otimes U^{\dagger}`.
        If :math:`|s| > d`, the remaining :math:`|s| - d` powers are applied as
        ``qp.pow(qp.adjoint(U), abs(s) - d)``. A positive shift is applied as ``qp.pow(U, s)``.

        This can be used to encode a Laurent polynomial
        :math:`P(x) = \sum_{k=-m}^{n} c_k x^k`: compute the angles of the polynomial
        :math:`x^{m} P(x)` and use ``shift=-m``.

        .. code-block:: python

            # P(x) = 0.1 x^-1 + 0.2j + 0.3 x
            poly = [0.1, 0.2j, 0.3]

            angles = qp.poly_to_angles(poly, "GQSP")

            @qp.qnode(qp.device("default.qubit"))
            def circuit(angles):
                qp.GQSP(qp.RX(0.3, wires=1), angles, control=0, shift=-1)
                return qp.state()

            matrix = qp.matrix(circuit, wire_order=[0, 1])(angles)

        .. code-block:: pycon

            >>> print(np.round(matrix, 3)[:2, :2])
            [[0.396+0.2j  0.   -0.03j]
             [0.   -0.03j 0.396+0.2j ]]
    """

    dynamic_argnames = ("angles",)
    static_argnames = ("work_wire_type", "shift")
    hybrid_argnames = ("unitary",)
    wire_argnames = ("control", "work_wires")

    arg_specs = {"angles": Float[3, -1], "control": Wire[1], "work_wires": Wire[-1]}

    def __init__(
        self, unitary, angles, control, work_wires=None, work_wire_type="borrowed", shift=0
    ):
        # pylint: disable=too-many-arguments
        work_wires = Wires(()) if work_wires is None else work_wires
        _validate_work_wire_type(work_wire_type)
        if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
            raise TypeError(f"shift must be an integer. Got {shift} of type {type(shift)}.")
        shift = int(shift)
        if isinstance(angles, (list, tuple)):
            angles = math.stack(angles)
        super().__init__(unitary, angles, control, work_wires, work_wire_type, shift)
