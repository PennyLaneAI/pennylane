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
Contains the Adder template.
"""

from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


class Adder(Operator2):
    r"""Performs the in-place modular addition operation.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    computational basis:

    .. math::

        \text{Adder}(k, mod) |x \rangle = | x+k \; \text{mod} \; mod \rangle.

    Two decomposition rules are registered for this operator. The first is based on the quantum
    Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. The second is a QFT-free arithmetic
    rule built from carry-ripple incrementers, which avoids the arbitrarily precise rotations of the
    QFT approach and whose multi-controlled gates reuse the remaining register wires as borrowed
    (dirty) work wires for a cheaper decomposition. Its modular-reduction structure follows the
    modular adder of Beauregard, 2.2
    (`arXiv:quant-ph/0205095 <https://arxiv.org/abs/quant-ph/0205095>`_), with the Fourier additions
    replaced by computational-basis incrementers. When the graph-based decomposition system is
    enabled (via :func:`~pennylane.decomposition.enable_graph`), the cheaper rule is selected
    automatically.

    .. note::

        To obtain the correct result, :math:`x` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.OutAdder`.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
            for encoding `x` in the computational basis. The number of wires also limits the
            maximum value for `mod`.
        mod (int): the modulo for performing the addition. If not provided, it will be set to its
            maximum value, :math:`2^{\text{len(x_wires)}}`.
        work_wires (Sequence[int]): the auxiliary wires to use for the addition. The
            work wires are not needed if :math:`mod=2^{\text{len(x_wires)}}`, otherwise at least
            two work wires should be provided. Defaults to empty tuple.

    **Example**

    This example computes the sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`.

    .. code-block:: python

        x = 8
        k = 5
        mod = 15

        x_wires =[0,1,2,3]
        work_wires=[4,5]

        dev = qp.device("default.qubit")
        @qp.qnode(dev, shots=1)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(x_wires))
            qp.BasisEmbedding(x_bin, wires=x_wires)
            qp.Adder(k, x_wires, mod, work_wires)
            return qp.sample(wires=x_wires)

    >>> print(circuit())
    [[1 1 0 1]]

    The result, :math:`[[1 1 0 1]]`, is the binary representation of
    :math:`8 + 5  \; \text{modulo} \; 15 = 13`.

    .. details::
        :title: Usage Details

        This template takes as input two different sets of wires.

        The first one is ``x_wires``, used to encode the integer :math:`x < \text{mod}` in the Fourier basis.
        To represent :math:`x`, ``x_wires`` must include at least :math:`\lceil \log_2(x) \rceil` wires.
        After the modular addition, the result can be as large as :math:`\text{mod} - 1`,
        requiring at least :math:`\lceil \log_2(\text{mod}) \rceil` wires. Since :math:`x < \text{mod}`,
        :math:`\lceil \log_2(\text{mod}) \rceil` is a sufficient length for ``x_wires`` to cover all possible inputs and outputs.

        The second set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular addition operation.

        - If :math:`mod = 2^{\text{len(x_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=()``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(x_wires)}}`, at least two ``work_wires`` have to be provided.

        Note that the ``Adder`` template allows us to perform modular addition in the computational basis. However if one just wants to perform standard addition (with no modulo), that would be equivalent to setting
        the modulo :math:`mod` to a large enough value to ensure that :math:`x+k < mod`.
    """

    compilable_argnames = ("k", "mod")
    wire_argnames = ("x_wires", "work_wires")
    arg_specs = {"x_wires": Wire[-1], "work_wires": Wire[-1]}

    def __init__(self, k, x_wires: WiresLike, mod=None, work_wires: WiresLike = ()):
        x_wires = Wires(x_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        num_works_wires = len(work_wires)

        if mod is None:
            mod = 2 ** len(x_wires)
        elif mod != 2 ** len(x_wires) and num_works_wires < 2:
            raise ValueError(
                f"If mod is not 2^{len(x_wires)}, at least two work wires should be provided"
            )
        if not isinstance(k, int) or not isinstance(mod, int):
            raise ValueError("Both k and mod must be integers")
        if mod > 2 ** len(x_wires):
            raise ValueError(
                "Adder must have enough x_wires to represent mod. The maximum mod "
                f"with len(x_wires)={len(x_wires)} is {2 ** len(x_wires)}, but received {mod}."
            )
        validate_no_wire_overlaps({"x_wires": x_wires, "work_wires": work_wires})

        super().__init__(k, x_wires, mod, work_wires)

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.x_wires + self.work_wires
