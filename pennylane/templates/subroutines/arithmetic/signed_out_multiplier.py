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
Contains the SignedOutMultiplier template.
"""

from collections import defaultdict
from typing import Any, Hashable, Iterable

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operator
from pennylane.ops import CNOT, Controlled, PauliX
from pennylane.wires import Wires, WiresLike

from .incrementer import Incrementer
from .out_multiplier import OutMultiplier
from .semi_adder import SemiAdder

has_jax = True
try:
    from jax import numpy as jnp
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_jax = False  # pragma: no cover


class SignedOutMultiplier(Operator):
    r"""
    Implements signed out-place multiplication :math:`|x,y,z\rangle \mapsto |x,y,(z + xy) \mod 2^{|z|}\rangle`.

    The inputs and output are given in `2s complement <https://en.wikipedia.org/wiki/Two%27s_complement>`__. 
    The values :math:`x`, :math:`y` and :math:`z` are encoded in big-endian 2s complement. Wire :math:`0`
    stores the sign bit and wire :math:`i` stores the bit with weight :math:`2^{n-1-i}` for a register of
    length :math:`n`. For example, the value :math:`x` encoded by a bitstring :math:`x_0 x_1\dots x_{n-1}`
    is given by the following.

    .. math::
        \begin{align}
            x &= - 2^{n-1} x_0 + \sum_{j=1}^{n-1} x_j 2^{n-1-j}, \\
            y &= - 2^{m-1} y_0 + \sum_{j=1}^{m-1} y_j 2^{m-1-j}, \\
            z &= - 2^{k-1} z_0 + \sum_{j=1}^{k-1} z_j 2^{k-1-j}.
        \end{align}

    The first bit of each encoded bitstring gives the sign of the encoded number. :math:`1 \mapsto -`, :math:`0 \mapsto +`.
    This is however not a sign-magnitude encoding. Iff the encoded number is negative, the rest of the bits do not give the
    magnitude. Instead, the magnitude can be found by calculating e.g. :math:`\bar{x}=(-1)^{x_0}x`. This is done by flipping the bits of
    :math:`x` and adding 1. E.g., :math:`6=(0110)_2` but :math:`-6 = (1010)_2` because :math:`-(1010)_2 \oplus 1 = (0101)_2 \oplus 1 = (0110)_2`.

    Args:
        x_wires (Sequence[int]): wires that store the signed integer :math:`x`
        y_wires (Sequence[int]): wires that store the signed integer :math:`y`
        output_wires (Sequence[int]): wires that store the multiplication result. If the
            register is in a non-zero state :math:`z`, the product :math:`xy` will be added to this value
        work_wires (Sequence[int]): auxiliary wires to use for the multiplication. The needed
            number of work wires depends on the decomposition, the register sizes and
            ``output_wires_zeroed``. If the output wires are zeroed, we only need 2 work wires.
            Otherwise, we need ``2*len(output_wires) + 1`` work wires.
        output_wires_zeroed (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially. Setting this argument to ``True`` reduces the cost of
            the operation.

    **Example**

    This example performs the multiplication of two integers :math:`x=-3` and :math:`y=3`.
    We'll let :math:`z=0`.

    .. code-block:: python

        x = -3
        y = 3

        x_wires = [0, 1, 2]
        y_wires = [3, 4, 5]
        output_wires = [6, 7, 8, 9, 10, 11]
        work_wires = [12, 13, 14, 15]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():
            qp.BasisEmbedding(x, wires=x_wires)
            qp.BasisEmbedding(y, wires=y_wires)
            qp.SignedOutMultiplier(
                x_wires,
                y_wires,
                output_wires,
                work_wires, 
                output_wires_zeroed=True,
            )
            return qp.sample(wires=output_wires)

    >>> print(circuit())
    [[1 1 0 1 1 1]]

    The result :math:`[[1 1 0 1 1 1]]`, is the binary representation of
    :math:`-3 \cdot 3 \; = -9` in 2s complement form. We can tell it is negative since the first bit is 1.
    Then we can find the magnitude by flipping the bits and adding 1. This gives us :math:`[[0 0 1 0 0 1]]`.
    The sum of these bits is :math:`2^3 + 2^0 = 9` for :math:`k=6`.

    >>> print(qp.draw(qp.decompose(circuit, max_expansion=1), max_length=170)())
     0: ─╭|Ψ⟩─╭●────╭X───────╭Incrementer───────────────────────╭OutMultiplier───────────────────────────────────╭X───────╭Incrementer───────────────────────╭●────┤
     1: ─├|Ψ⟩─│─────│──╭X────├Incrementer───────────────────────├OutMultiplier───────────────────────────────────│──╭X────├Incrementer───────────────────────│─────┤
     2: ─╰|Ψ⟩─│─────│──│──╭X─├Incrementer───────────────────────├OutMultiplier───────────────────────────────────│──│──╭X─├Incrementer───────────────────────│─────┤
     3: ─╭|Ψ⟩─│──╭●─│──│──│──│────────────╭X───────╭Incrementer─├OutMultiplier───────────────────────────────────│──│──│──│────────────╭X───────╭Incrementer─│──╭●─┤
     4: ─├|Ψ⟩─│──│──│──│──│──│────────────│──╭X────├Incrementer─├OutMultiplier───────────────────────────────────│──│──│──│────────────│──╭X────├Incrementer─│──│──┤
     5: ─╰|Ψ⟩─│──│──│──│──│──│────────────│──│──╭X─├Incrementer─├OutMultiplier───────────────────────────────────│──│──│──│────────────│──│──╭X─├Incrementer─│──│──┤
     6: ──────│──│──│──│──│──│────────────│──│──│──│────────────│──────────────╭X─╭X─╭●─╭●─╭●─╭●─╭●─╭●───────────│──│──│──│────────────│──│──│──│────────────│──│──┤ ╭Sample
     7: ──────│──│──│──│──│──│────────────│──│──│──│────────────├OutMultiplier─│──│──╰X─│──│──│──│──├Incrementer─│──│──│──│────────────│──│──│──│────────────│──│──┤ ├Sample
     8: ──────│──│──│──│──│──│────────────│──│──│──│────────────├OutMultiplier─│──│─────╰X─│──│──│──├Incrementer─│──│──│──│────────────│──│──│──│────────────│──│──┤ ├Sample
     9: ──────│──│──│──│──│──│────────────│──│──│──│────────────├OutMultiplier─│──│────────╰X─│──│──├Incrementer─│──│──│──│────────────│──│──│──│────────────│──│──┤ ├Sample
    10: ──────│──│──│──│──│──│────────────│──│──│──│────────────├OutMultiplier─│──│───────────╰X─│──├Incrementer─│──│──│──│────────────│──│──│──│────────────│──│──┤ ├Sample
    11: ──────│──│──│──│──│──│────────────│──│──│──│────────────├OutMultiplier─│──│──────────────╰X─├Incrementer─│──│──│──│────────────│──│──│──│────────────│──│──┤ ╰Sample
    12: ──────╰X─│──╰●─╰●─╰●─├●───────────│──│──│──│────────────│──────────────╰●─│─────────────────│────────────╰●─╰●─╰●─├●───────────│──│──│──│────────────╰X─│──┤
    13: ─────────╰X──────────│────────────╰●─╰●─╰●─├●───────────│─────────────────╰●────────────────│─────────────────────│────────────╰●─╰●─╰●─├●──────────────╰X─┤
    14: ─────────────────────├Incrementer──────────├Incrementer─├OutMultiplier──────────────────────├Incrementer──────────├Incrementer──────────├Incrementer───────┤
    15: ─────────────────────╰Incrementer──────────╰Incrementer─╰OutMultiplier──────────────────────╰Incrementer──────────╰Incrementer──────────╰Incrementer───────┤

    .. details::
        :title: Theoretical background
        :href: theory

        We begin with three signed quantum registers, storing :math:`|x\rangle`, :math:`|y\rangle` and :math:`|0\rangle` with register sizes :math:`n`, :math:`m` and :math:`k`, respectively. We will turn to the case of non-zero initial states for the last register later.
        Here, :math:`x`, :math:`y` and :math:`z` are signed integers in big-endian 2s complement representation:

        .. math::
            \begin{align}
                x &= - 2^{n-1} x_0 + \sum_{j=1}^{n-1} x_j 2^{n-1-j}, \\
                y &= - 2^{m-1} y_0 + \sum_{j=1}^{m-1} y_j 2^{m-1-j}, \\
                z &= - 2^{k-1} z_0 + \sum_{j=1}^{k-1} z_j 2^{k-1-j}.
            \end{align}

        We also have the magnitude of the signed integers, which can be computed by flipping all bits and incrementing by one,
        both steps controlled on the sign bit (:math:`x_0` for :math:`x` and :math:`y_0` for :math:`y`). For :math:`x`,

        .. math::
            \begin{align}
                \bar{x}&=(1-x_0)x+x_0\left(1 + \sum_{j=0}^{n-1} (1-x_j) 2^{n-1-j}\right)\\
                % &=2^n-x_u\\
                &=(x-x_0 x) + x_0 (1 + \sum_{j=0}^{n-1} 2^{n-1-j} - \sum_{j=0}^{n-1} x_j 2^{n-1-j}) \\
                &=(x-x_0 x) + x_0 (1 + \sum_{j=0}^{n-1} 2^{n-1-j} - (2^{n-1}x_0 + \sum_{j=1}^{n-1} x_j 2^{n-1-j})) \\
                &=(x-x_0 x) + x_0 (1 + \sum_{j=0}^{n-1} 2^{n-1-j} - (2^n x_0 - 2^{n-1}x_0 + \sum_{j=1}^{n-1} x_j 2^{n-1-j})) \\
                &=(x-x_0 x) + x_0 (1 + (2^n - 1) - (x + 2^n x_0)) \\
                &=(x-x_0 x)+x_02^n-x_0(x+2^nx_0)\\
                &=x-2x_0x+2^nx_0(1-x_0)\\
                &=(1-2x_0)x + 2^n x_0 - 2^n x_0^2 \\
                &=(1-2x_0)x\\
                &=(-1)^{x_0}x.
            \end{align}

        The first step is to copy the sign bit of :math:`x` and :math:`y` to one auxiliary qubit each, and to compute the magnitude of the respective integer, as just described.

        .. code-block::

                           |-Cache Signs-| |--- Compute Absolute Magnitude (In-Place) --|
                                                                      ┌───────┐
            |x_bits⟩ [n-1] ──────────────────────X────────────────────┤       ├─────────  --> |x_abs⟩
                                                 │                    │  +1   │
            |x_sign⟩ [1]   ──────●───────────────X────────────────────┤       ├─────────
                                 │               │                    └───┬───┘
            |0⟩            ──────X───────────────●────────────────────────●─────────────  --> |cache_x⟩
                                                                      ┌───────┐
            |y_bits⟩ [m-1] ─────────────────────────────X─────────────┤       ├─────────  --> |y_abs⟩
                                                        │             │  +1   │
            |y_sign⟩ [1]   ──────●──────────────────────X─────────────┤       ├─────────
                                 │                      │             └───┬───┘
            |0⟩            ──────X──────────────────────●─────────────────●─────────────  --> |cache_y⟩


        At this point we have the state:

        .. math::
            \begin{align}
                |\bar{x}\rangle |x_0\rangle |\bar{y}\rangle |y_0\rangle |0\rangle_s |0\rangle,
            \end{align}

        where we interleaved the two auxiliary qubits with the input registers and the output register, and wrote the sign bit of the output register as a separate qubit, marked with an :math:`s` for clarity.

        Next, we multiply the magnitude registers into the output register, obtaining

        .. math::
            \begin{align}
                |\bar{x}\rangle |x_0\rangle |\bar{y}\rangle |y_0\rangle |0\rangle_s |\bar{x}\bar{y}\rangle.
            \end{align}

        .. code-block::

            |x̄⟩   ─────────────[n]───────────■───────────
                                             │
            |x_0⟩ ───────────────────────────┼───────────
                                             │
            |ȳ⟩   ─────────────[m]───────────■───────────
                                             │
            |y_0⟩ ───────────────────────────┼───────────
                                             │
            |0⟩_s ───────────────────────────┼───────────
                                        ┌────┴────┐
            |0⟩  ──────────────[k-1]────┤ Unsigned├──────
                                        │ Mult x*y│
                                        └─────────┘

        Then, we flip the sign bit of the output register controlled on the (cached) sign bits of each input, respectively:

        .. math::
            \begin{align}
                |\bar{x}\rangle |x_0\rangle |\bar{y}\rangle |y_0\rangle |x_0+y_0 \rangle_s |\bar{x}\bar{y}\rangle.
            \end{align}

        .. code-block::

            |x̄⟩       ─────────────────────────────────

            |x_0⟩     ───────────────●─────────────────
                                     │
            |ȳ⟩       ───────────────┼─────────────────
                                     │
            |y_0⟩     ───────────────┼────────●────────
                                     │        │
            |0⟩_s     ───────────────X────────X────────  --> Becomes |z_s⟩_s

            |x̄ȳ⟩      ─────────────────────────────────

        From here on we write :math:`z_s = z_0 = x_0+y_0`.
        Then, we flip and increment the (non-sign) bits of the output register controlled on the output sign bit to get (where :math:`k` is the size of the output register including the sign bit):

        .. math::
            \begin{align}
                |\bar{x}\rangle |x_0\rangle |\bar{y}\rangle |y_0\rangle |z_s \rangle_s |(-1)^{z_s}\bar{x}\bar{y}+2^{k - 1} z_s\rangle.
            \end{align}

        Arrived at by the following arithmetic.

        .. math::
            \begin{align}
                &(1 - z_s) \bar{x}\bar{y} + z_s (1 + \sum_{j=0}^{k-2} (1 - \bar{x}\bar{y}_j)2^{k-2-j}) \\
                &=(1 - z_s) \bar{x}\bar{y} + z_s(1 + \sum_{j=0}^{k-2}2^{k-2-j} - \sum_{j=0}^{k-2} \bar{x}\bar{y}_j2^{k-2-j}) \\
                &=(1 - z_s) \bar{x}\bar{y} + z_s(1 + 2^{k-1} - 1 - \bar{x}\bar{y}) \\
                &=(1 - z_s) \bar{x}\bar{y} + z_s (2^{k-1} - \bar{x}\bar{y}) \\\
                &=(-1)^{z_s}\bar{x}\bar{y}+2^{k - 1} z_s
            \end{align}

        .. code-block::

            |z_s⟩_s   ───────────────●────────●────────
                                   ┌─┴─┐   ┌──┴──┐
            |x̄ȳ⟩      ─────────────┤NOT├───┤ +1  ├─────
                                   └───┘   └─────┘

        Then we uncompute the magnitudes and the copied sign bits of the input registers, arriving at

        .. math::
            \begin{align}
                |x\rangle |0\rangle |y\rangle |0\rangle |z_s \rangle_s |(-1)^{z_s}\bar{x}\bar{y}+2^{k-1} z_s\rangle.
            \end{align}

        Interpreting the output register as signed integer, we find that we computed

        .. math::
            \begin{align}
                z &= (-1)^{z_0}\bar{x}\bar{y}+2^{k-1}z_0 - 2^{k-1} z_0\\
                &=(-1)^{z_0} \bar{x}\bar{y} \\
                &=(-1)^{x_0}\bar{x} (-1)^{y_0}\bar{y}\\
                &= x y.
            \end{align}

        So we correctly arrive at the product of :math:`x` and :math:`y`.

        **Non-zero initial state of output wires**

        If we have a non-zero initial state :math:`z_i` in the output register, we will end up with :math:`xy + z_i` in the
        output register once the template has executed. This requires more work wires and a more costly decomposition.

        Basically, we use auxiliary registers to first 1) compute the multiplication of the operands into a zeroed register,
        2) use an Adder to add :math:`z_i` to this result.


        .. code-block::

            |x⟩ ───────────[n]─────────■────────────────────────────────────────────────────────
                                       │
            |y⟩ ───────────[m]─────────■────────────────────────────────────────────────────────
                                   ┌───┴───┐
            |0⟩ ───────────[k]─────┤Signed ├─────|x·y⟩ ───────■─────────────────────────────────
                                   │ Mult  │                  │
                                   └───────┘                  │
            |z_s⟩ ─────────[1]────────────────────────────────┼─────────────────────────────────
                                                              │
                                                            ┌─┴─┐
            |z_i⟩ ─────────[k]──────────────────────────────┤ + ├──────|x·y + z_i⟩──────────────
                                                            └───┘

    """

    resource_keys = {
        "num_output_wires",
        "num_work_wires",
        "num_x_wires",
        "num_y_wires",
        "output_wires_zeroed",
    }

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        output_wires_zeroed: bool = False,
    ):  # pylint: disable=too-many-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)

        if any(wire in work_wires for wire in x_wires):
            raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if any(wire in work_wires for wire in y_wires):
            raise ValueError("None of the wires in work_wires should be included in y_wires.")

        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        if any(wire in y_wires for wire in output_wires):
            raise ValueError("None of the wires in y_wires should be included in output_wires.")

        wires_list = [x_wires, y_wires, output_wires, work_wires]
        wires_name = ["x_wires", "y_wires", "output_wires", "work_wires"]

        for name, wires in zip(wires_name, wires_list):
            self.hyperparameters[name] = Wires(wires)

        self.hyperparameters["output_wires_zeroed"] = output_wires_zeroed

        # pylint: disable=consider-using-generator
        all_wires = sum([self.hyperparameters[name] for name in wires_name], start=[])
        super().__init__(wires=all_wires)

    @classmethod
    def _unflatten(cls, data: Iterable[Any], metadata: Hashable):
        hyperparameters_dict = dict(metadata[1])
        return cls(*data, **hyperparameters_dict)

    def map_wires(self, wire_map: dict):
        x_wires = [wire_map.get(w, w) for w in self.hyperparameters["x_wires"]]
        y_wires = [wire_map.get(w, w) for w in self.hyperparameters["y_wires"]]
        output_wires = [wire_map.get(w, w) for w in self.hyperparameters["output_wires"]]
        work_wires = [wire_map.get(w, w) for w in self.hyperparameters["work_wires"]]
        output_wires_zeroed = self.hyperparameters["output_wires_zeroed"]

        return SignedOutMultiplier(
            x_wires,
            y_wires,
            output_wires,
            work_wires,
            output_wires_zeroed,
        )

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "output_wires_zeroed": self.hyperparameters["output_wires_zeroed"],
        }

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)


def _zeroed_signed_out_multiplier_resources(
    num_x_wires, num_y_wires, num_output_wires, num_work_wires, **_
):
    """
    Computes the resources for the SignedOutMultiplier.
    """
    resources = defaultdict(int)

    resources[controlled_resource_rep(PauliX, {}, 1, 0)] = (
        (num_x_wires - 1 + num_y_wires - 1) * 2 + num_output_wires - 1
    )
    resources[
        controlled_resource_rep(
            Incrementer,
            {"num_wires": num_x_wires, "num_work_wires": num_work_wires - 2},
            num_control_wires=1,
        )
    ] += 2
    resources[
        resource_rep(
            OutMultiplier,
            num_output_wires=num_output_wires - 1,
            num_x_wires=num_x_wires,
            num_y_wires=num_y_wires,
            num_work_wires=num_work_wires - 2,
            mod=2 ** (num_output_wires - 1),
            output_wires_zeroed=True,
        )
    ] = 1
    resources[
        controlled_resource_rep(
            Incrementer,
            {
                "num_wires": num_output_wires - 1,
                "num_work_wires": num_work_wires - 2,
            },
            num_control_wires=1,
        )
    ] += 1
    resources[
        controlled_resource_rep(
            Incrementer,
            {"num_wires": num_y_wires, "num_work_wires": num_work_wires - 2},
            num_control_wires=1,
        )
    ] += 2

    resources[resource_rep(CNOT)] = 6 + (num_x_wires + num_y_wires) * 2 + (num_output_wires - 1)

    return resources


def _not_zeroed_signed_out_multiplier_resources(
    num_x_wires, num_y_wires, num_output_wires, num_work_wires, **_
):
    """
    Computes the resources for the SignedOutMultiplier.
    """
    resources = {}

    resources[
        resource_rep(
            SignedOutMultiplier,
            num_output_wires=num_output_wires,
            num_x_wires=num_x_wires,
            num_y_wires=num_y_wires,
            num_work_wires=2 + num_work_wires - (2 * num_output_wires + 1),
            output_wires_zeroed=True,
        )
    ] = 1

    resources[
        resource_rep(
            SemiAdder,
            num_x_wires=num_output_wires,
            num_y_wires=num_output_wires,
            num_work_wires=num_output_wires - 1,
        )
    ] = 1

    return resources


def _twos_complement_helper(input_reg, aux_wire, work_wires):
    # Invert all bits
    @for_loop(len(input_reg))
    def invert(w):
        # sign bit of 1 indicates a negative value
        CNOT([aux_wire, input_reg[w]])

    invert()  # pylint: disable=no-value-for-parameter

    if capture.enabled():
        input = jnp.append(input_reg, work_wires)
    else:
        input = input_reg + work_wires

    # Add one
    Controlled(
        Incrementer(
            wires=input,
            work_wires=work_wires,  # we can use the work wires since they are returned in a clean state
        ),
        control_wires=aux_wire,
        control_values=(1,),
    )


def _not_zeroed_work_wire_condition(num_work_wires, num_output_wires, **_):
    return num_work_wires >= 2 * num_output_wires + 1


def _zeroed_work_wire_condition(num_work_wires, **_):
    return num_work_wires >= 2


def _zeroed_condition(output_wires_zeroed, **_):
    return output_wires_zeroed


def _not_zeroed_condition(output_wires_zeroed, **_):
    return not output_wires_zeroed


@register_condition(_zeroed_condition)
@register_condition(_zeroed_work_wire_condition)
@register_resources(_zeroed_signed_out_multiplier_resources)
def _signed_out_multiplier_decomposition_zeroed(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    **_,
):
    """Computes the decomposition of the operator as a product of other operators when the output wires are zeroed."""
    if capture.enabled():
        x_wires, y_wires, work_wires, output_wires = (
            math.array(x_wires, like="jax"),
            math.array(y_wires, like="jax"),
            math.array(work_wires, like="jax"),
            math.array(output_wires, like="jax"),
        )

    # We have a more efficient decomposition if the output wires are zeroed

    x_aux = work_wires[0]
    y_aux = work_wires[1]

    # Sign extension
    CNOT([x_wires[0], x_aux])
    CNOT([y_wires[0], y_aux])

    # Take 2s complements if necessary
    _twos_complement_helper(x_wires, x_aux, work_wires[2:])
    _twos_complement_helper(y_wires, y_aux, work_wires[2:])

    # at this point the sign is only kept in the auxiliary qubits' states

    # Multiply the magnitudes
    OutMultiplier(
        x_wires,
        y_wires,
        output_wires[1:],
        work_wires=work_wires[2:],
        output_wires_zeroed=True,
    )

    # Compute the sign
    CNOT([x_aux, output_wires[0]])
    CNOT([y_aux, output_wires[0]])

    # Encode the output
    _twos_complement_helper(output_wires[1:], output_wires[0], work_wires[2:])

    # Return inputs to original state
    _twos_complement_helper(x_wires, x_aux, work_wires[2:])
    _twos_complement_helper(y_wires, y_aux, work_wires[2:])

    # Uncompute sign extension
    CNOT([x_wires[0], x_aux])
    CNOT([y_wires[0], y_aux])


@register_condition(_not_zeroed_condition)
@register_condition(_not_zeroed_work_wire_condition)
@register_resources(_not_zeroed_signed_out_multiplier_resources)
def _signed_out_multiplier_decomposition_not_zeroed(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    **_,
):
    """Computes the decomposition of the operator as a product of other operators."""

    if capture.enabled():
        x_wires, y_wires, work_wires, output_wires = (
            math.array(x_wires, like="jax"),
            math.array(y_wires, like="jax"),
            math.array(work_wires, like="jax"),
            math.array(output_wires, like="jax"),
        )

    x_aux = work_wires[0]
    y_aux = work_wires[1]

    # Temp output register for multiplication output
    mult_temp = work_wires[2 : len(output_wires) + 2]

    if capture.enabled():
        signed_work_wires = work_wires[2 * len(output_wires) + 1 :]
        signed_work_wires = jnp.concatenate([jnp.atleast_1d(y_aux), signed_work_wires])
        signed_work_wires = jnp.concatenate([jnp.atleast_1d(x_aux), signed_work_wires])
    else:
        signed_work_wires = [x_aux] + [y_aux] + work_wires[2 * len(output_wires) + 1 :]

    SignedOutMultiplier(x_wires, y_wires, mult_temp, signed_work_wires, True)

    # Add any initial value in the output register
    SemiAdder(
        mult_temp,
        output_wires,
        work_wires=work_wires[len(output_wires) + 2 : 2 * len(output_wires) + 1],
    )


add_decomps(SignedOutMultiplier, _signed_out_multiplier_decomposition_not_zeroed)
add_decomps(SignedOutMultiplier, _signed_out_multiplier_decomposition_zeroed)
