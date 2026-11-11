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
Contains the PhaseAdder template.
"""

from pennylane.core.operator import Operator2
from pennylane.typing import AbstractArray, Int, Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


class PhaseAdder(Operator2):
    r"""Performs the in-place modular phase addition operation.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    Fourier basis:

    .. math::

        \text{PhaseAdder}(k,mod) |\phi (x) \rangle = |\phi (x+k \; \text{mod} \; mod) \rangle,

    where :math:`|\phi (x) \rangle` represents the :math:`| x \rangle` state in the Fourier basis,

    .. math::

        \text{QFT} |x \rangle = |\phi (x) \rangle.

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x` must be smaller than :math:`mod`. Also, when
        :math:`mod \neq 2^{\text{len(x_wires)}}`, :math:`x` must satisfy :math:`x < 2^{\text{len(x_wires)}-1}`,
        which means that one extra wire in ``x_wires`` is required.

    .. seealso:: :class:`~.QFT` and :class:`~.Adder`.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
            for a binary representation of the value being targeted, :math:`x`. In some cases an additional
            wire is needed, see usage details below. The number of wires also limits the maximum
            value for ``mod``.
        mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
        work_wire (Sequence[int] or int): the auxiliary wire to use for the addition. Optional
            when ``mod`` is :math:`2^{\text{len(x_wires)}}`. Defaults to empty tuple. The work wire must
            be in the :math:`|0\rangle` state.

    **Example**

    This example computes the sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`.

    .. code-block:: python

        x = 8
        k = 5
        mod = 15

        x_wires =[0,1,2,3]
        work_wire=[5]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(x_wires))
            qp.BasisEmbedding(x_bin, wires=x_wires)
            qp.QFT(wires=x_wires)
            qp.PhaseAdder(k, x_wires, mod, work_wire)
            qp.adjoint(qp.QFT)(wires=x_wires)
            return qp.sample(wires=x_wires)

    >>> print(circuit())
    [[1 1 0 1]]

    The result, :math:`[[1 1 0 1]]`, is the binary representation of
    :math:`8 + 5  \; \text{modulo} \; 15 = 13`.

    .. details::
        :title: Usage Details

        This template takes as input two different sets of wires.

        The first one is ``x_wires``, used to encode the integer :math:`x < \text{mod}` in the Fourier basis.
        To represent :math:`x`, at least :math:`\lceil \log_2(x) \rceil` wires are needed.
        After the modular addition, the result can be as large as :math:`\text{mod} - 1`,
        requiring at least :math:`\lceil \log_2(\text{mod}) \rceil` wires. Since :math:`x < \text{mod}`, a length of
        :math:`\lceil \log_2(\text{mod}) \rceil` is sufficient for ``x_wires`` to cover all possible inputs and
        outputs when :math:`mod = 2^{\text{len(x_wires)}}`.
        An exception occurs when :math:`mod \neq 2^{\text{len(x_wires)}}`. In that case one extra wire in ``x_wires`` will be needed to correctly perform the phase
        addition operation.

        The second set of wires is ``work_wire`` which consist of the auxiliary qubit used to perform the modular phase addition operation.

        - If :math:`mod = 2^{\text{len(x_wires)}}`, there will be no need for ``work_wire``, hence ``work_wire=()``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(x_wires)}}`, one ``work_wire`` has to be provided.

        Note that the ``PhaseAdder`` template allows us to perform modular addition in the Fourier basis. However if one just wants to perform standard addition (with no modulo),
        that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x+k < mod`.
    """

    grad_method = None

    dynamic_argnames = ("k",)
    compilable_argnames = ("mod",)
    wire_argnames = ("x_wires", "work_wire")
    arg_specs = {"k": Int, "x_wires": Wire[-1], "work_wire": Wire[-1]}

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, k, x_wires: WiresLike, mod=None, work_wire: WiresLike = ()):

        work_wire = Wires(() if work_wire is None else work_wire)
        x_wires = Wires(x_wires)

        num_x_wires = len(x_wires)
        max_mod = 2**num_x_wires

        if mod is None:
            mod = max_mod
        elif mod != max_mod and len(work_wire) != 1:
            raise ValueError(f"If mod is not 2^{num_x_wires}, one work wire should be provided.")

        if not isinstance(mod, int):
            raise ValueError("Both k and mod must be integers")

        if mod > max_mod:
            raise ValueError(
                "PhaseAdder must have enough x_wires to represent mod. The maximum mod "
                f"with len(x_wires)={num_x_wires} is {max_mod}, but received {mod}."
            )
        validate_no_wire_overlaps({"x_wires": x_wires, "work_wire": work_wire})
        super().__init__(k=k, x_wires=x_wires, mod=mod, work_wire=work_wire)
        if not isinstance(self.k, AbstractArray):
            self.arguments["k"] = self.k % mod
