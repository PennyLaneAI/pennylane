# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
Contains the OutMultiplier template.
"""

from pennylane.core.operator import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires, WiresLike, validate_no_wire_overlaps


def _resolve_mod_and_num_work_wires(num_output_wires, mod, num_work_wires):
    """Resolve default ``mod`` and truncated work wire count."""
    max_mod = 2**num_output_wires
    if mod is None:
        mod = max_mod
    elif mod != max_mod:
        # After the ≥2 work-wire guard in __init__, the truncated count is always 2
        num_work_wires = 2
    return mod, num_work_wires


class OutMultiplier(Operator2):
    r"""Performs the out-place modular multiplication operation.

    This operator performs the modular multiplication of integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::
        \text{OutMultiplier}(mod) |x \rangle |y \rangle |z \rangle = |x \rangle |y \rangle |z + x \cdot y \; \text{mod} \; mod \rangle,

    There are five implementations available, which differ in the auxiliary wires
    and in the gate counts they require, and in whether or not they support arbitrary values for
    the modulus ``mod``. See the usage details for more information.

    .. note::

        To obtain the correct result, :math:`x`, :math:`y` and :math:`z` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.Multiplier`, :class:`~.SemiAdder`, and :class:`~.PhaseAdder`.

    Args:
        x_wires (Sequence[int]): wires that store the integer :math:`x`
        y_wires (Sequence[int]): wires that store the integer :math:`y`
        output_wires (Sequence[int]): wires that store the multiplication result. If the
            register is in a non-zero state :math:`z`, the solution will be added to this value
        mod (int): the modulo for performing the multiplication. If not provided, it will be set
            to its maximum value, :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): auxiliary wires to use for the multiplication. The needed
            number of work wires depends on the decomposition, the register sizes and
            ``output_wires_zeroed``. Defaults to an empty tuple, i.e., no work wires.
        output_wires_zeroed (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially. Setting this argument to ``True`` reduces the cost of
            the operation.

    **Example**

    This example performs the multiplication of two integers :math:`x=2` and :math:`y=7` modulo
    :math:`mod=12`. We'll let :math:`z=0`. See Usage Details for :math:`z \neq 0`.

    .. code-block:: python

        x = 2
        y = 7
        mod = 12

        x_wires = [0, 1]
        y_wires = [2, 3, 4]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(x_wires))
            y_bin = qp.math.int_to_binary(y, len(y_wires))
            qp.BasisState(x_bin, wires=x_wires)
            qp.BasisState(y_bin, wires=y_wires)
            qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qp.sample(wires=output_wires)

    >>> print(circuit())
    [[0 0 1 0]]

    The result :math:`[[0 0 1 0]]`, is the binary representation of
    :math:`2 \cdot 7 \; \text{modulo} \; 12 = 2`.

    .. details::
        :title: Available decompositions

        .. decomposition-table:: qp.OutMultiplier
            :symbols: :math:`n` ``x_wires``, :math:`m` ``y_wires``, :math:`k \geq n + m`
                ``output_wires``, :math:`w` ``work_wires``, ``mod`` :math:`=2^k` and
                ``output_wires_zeroed=False``
            :example: :math:`n=m=1`, :math:`k=2`

            .. decomposition-rule:: _out_multiplier_with_qft
                :image: /_static/decomposition_thumbnails/out_multiplier_qft.svg
                :requires: program capture to be disabled

                :class:`~.QFT` | :math:`1` | 1
                :class:`~.ControlledSequence` (nested, of :class:`~.PhaseAdder`) | :math:`1` | 1
                :class:`Adjoint(QFT) <.QFT>` | :math:`1` | 1

            .. decomposition-rule:: _out_multiplier_with_adder
                :image: /_static/decomposition_thumbnails/out_multiplier_adder.svg
                :requires: :math:`w \geq k` work wires

                :class:`~.SemiAdder` with 1 control | :math:`n` | 1

            .. decomposition-rule:: _out_multiplier_with_caddsub
                :image: /_static/decomposition_thumbnails/out_multiplier_caddsub.svg
                :requires: :math:`w \geq k + 1` work wires
                :where: :math:`[m > 1]` equal to 1 if :math:`m > 1` and 0 otherwise

                :class:`~.TemporaryAND` | :math:`(n + 1)k - m - \binom{n}{2}` | 3
                :class:`Adjoint(TemporaryAND) <.TemporaryAND>` | :math:`(n + 1)k - m - \binom{n}{2}` | 3
                :class:`~.CNOT` | :math:`(n + 1)k + 5nm + n - m - 4 - \binom{n}{2}` | 5
                :class:`~.X` | :math:`2k + 8` | 12
                :class:`~.X` with 1 control on :math:`|0\rangle` | :math:`4n` | 4
                :class:`~.MultiX` with 1 control on :math:`|0\rangle` | :math:`2n\,[m > 1]` | 0
                :class:`~.SemiAdder` | :math:`2` | 2
                :class:`~.Incrementer` | :math:`1` | 1

            .. decomposition-rule:: _out_multiplier_with_cache
                :image: /_static/decomposition_thumbnails/out_multiplier_cache.svg
                :requires: :math:`w \geq 2k - 1` work wires

                :class:`~.OutMultiplier` with ``output_wires_zeroed=True`` | :math:`1` | 1
                :class:`~.SemiAdder` | :math:`1` | 1
                :class:`Adjoint(OutMultiplier) <.OutMultiplier>` | :math:`1` | 1

            .. decomposition-rule:: _out_multiplier_swapped
                :image: /_static/decomposition_thumbnails/out_multiplier_swapped.svg
                :requires: :math:`n > m`
                :where: the circuit shows :math:`n=2`, :math:`m=1` and :math:`k=3`

                :class:`~.OutMultiplier` with ``x_wires`` and ``y_wires`` swapped | :math:`1`

    .. details::
        :title: Usage Details

        This template takes as input four different registers of wires.

        The first register is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis.

        The second register is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis.

        The third register is ``output_wires`` which is used
        to encode the integer :math:`(z+ x \cdot y) \; \text{mod} \; mod` in the computational
        basis. Therefore, it will require at least :math:`\lceil \log_2(mod)\rceil` wires
        Note that these wires can be initialized with any integer :math:`z < mod`.

        The fourth register is ``work_wires`` containing the auxiliary qubits used to
        perform the modular multiplication operation. The number of auxiliary wires determines
        which decomposition is available (also see below).

        **Initial state of output wires**

        As indicated above, the initial state of ``output_wires`` can encode any value
        :math:`z<mod`. The following is an example for :math:`z = 1`.

        .. code-block:: python

            z = 1
            x = 2
            y = 7
            mod = 12

            x_wires = [0, 1]
            y_wires = [2, 3, 4]
            output_wires = [6, 7, 8, 9]
            work_wires = [5, 10]

            dev = qp.device("default.qubit")

            @qp.qnode(dev, shots=1)
            def circuit():
                x_bin = qp.math.int_to_binary(x, len(x_wires))
                qp.BasisState(x_bin, wires=x_wires)
                y_bin = qp.math.int_to_binary(y, len(y_wires))
                qp.BasisState(y_bin, wires=y_wires)
                z_bin = qp.math.int_to_binary(z, len(output_wires))
                qp.BasisState(z_bin, wires=output_wires)
                qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
                return qp.sample(wires=output_wires)

        >>> print(circuit())
        [[0 0 1 1]]

        The result :math:`(0011)_2`, is the binary representation of
        :math:`(1 + 2 \cdot 7)\; \text{modulo} \; 12 = 3`:

        If the initial state on the output wires is guaranteed to be :math:`|0\rangle`, this
        can be indicated to ``OutMultiplier`` by setting ``output_wires_zeroed=True``. This
        simplifies some of the available decompositions (also see below), saving quantum resources.

        **Different decompositions**

        There are five decompositions, which differ in the required number of work wires and
        gates, and in whether they support ``mod!=2**len(output_wires)``.

        - The first decomposition is based on the quantum Fourier transform (QFT) method presented
          in `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. We nest
          :class:`~.ControlledSequence` around :class:`~.PhaseAdder` to create doubly-controlled
          in place phase addition in the output register, which is then transformed into
          addition by a basis change using :class:`~.QFT`\ s. It requires zero (two) auxiliary
          wires for ``mod=2**len(output_wires)`` (for other values of ``mod``)
          Any value for ``mod`` is supported, subject to the description above.

        - The second decomposition uses controlled :class:`~.SemiAdder`\ s to realize the
          multiplication. For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L = \min(k, n)` adders with usually varying sizes
          :math:`\min(k - i, m + 1)` for :math:`0\leq i<L`. The decomposition is shown in
          Fig. 2a) (for :math:`k=2m=2n`) and Fig. 2c) (for :math:`k=m=n`) in
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.

        - The third decomposition uses controlled addition/subtraction to replace the controlled
          ``SemiAdder``\ s from the previous decomposition, based on Litinski's
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
          For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L=\min(k, n)` controlled add/subtract operations
          of usually varying size :math:`\min(k + 1 - i, m + 1)` for :math:`0\leq i<L`,
          three ``SemiAdder``\ s of sizes :math:`\min(k + 1 - m, n + 1)`,
          :math:`\min(k + 1 - n, m + 1)` and :math:`k+1`, as well as an incrementer on
          :math:`\min(k + 1, n + m)` qubits and Pauli gates. For :math:`n=m` and
          :math:`k=2n`, this decomposition is shown in Fig. 2b), for :math:`k=n=m` in Fig. 2d).

        - The fourth decomposition uses additional work wires as cache in order to implement
          the multiplier with ``output_wires_zeroed=False`` via a multiplier with
          ``output_wires_zeroed=True``. For this, the latter computes the product into the
          additional cache register, which is then added with a simple :class:`~.SemiAdder` to
          the actual target register. Finally, the cache is uncomputed.

        - The fifth decomposition simply switches the two input registers of the multiplier if
          the ``x_wires`` register is larger than ``y_wires``. This is a valid decomposition
          because multiplication is commutative, and it saves resources because the cost of
          other decompositions is not symmetric with respect to the input register sizes.

    """

    wire_argnames = ("x_wires", "y_wires", "output_wires", "work_wires")
    compilable_argnames = ("mod", "output_wires_zeroed")

    arg_specs = {
        "x_wires": Wire[-1],
        "y_wires": Wire[-1],
        "output_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        output_wires_zeroed: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires([] if work_wires is None else work_wires)
        num_output_wires = len(output_wires)
        num_work_wires = len(work_wires)
        max_mod = 2**num_output_wires

        if mod is not None and mod != max_mod:
            if num_work_wires < 2:
                raise ValueError(
                    f"If mod is not 2^{num_output_wires}, at least two work wires should be provided."
                )
            if mod > max_mod:
                raise ValueError(
                    "OutMultiplier must have enough wires to represent mod. The maximum mod "
                    f"with len(output_wires)={num_output_wires} is {max_mod}, but received {mod}."
                )

        mod, num_work_wires = _resolve_mod_and_num_work_wires(num_output_wires, mod, num_work_wires)
        if mod != max_mod:
            work_wires = Wires(work_wires[:num_work_wires])

        wire_args = {
            "x_wires": x_wires,
            "y_wires": y_wires,
            "output_wires": output_wires,
            "work_wires": work_wires,
        }
        validate_no_wire_overlaps(wire_args)

        super().__init__(
            x_wires,
            y_wires,
            output_wires,
            mod=mod,
            work_wires=work_wires,
            output_wires_zeroed=output_wires_zeroed,
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.x_wires + self.y_wires + self.output_wires + self.work_wires

    # Note that dividing by two does not have to happen explicitly, because the registers are
    # not explicit return values.
