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
from pennylane.ops import CNOT, Controlled, MidMeasure, PauliX, measure, Toffoli, MultiControlledX
from pennylane.wires import Wires, WiresLike

from .incrementer import Incrementer
from .out_multiplier import OutMultiplier


class SignedOutMultiplier(Operator):
    r"""
    Implements the SignedOutMultiplier template.

    The SignedOutMultiplier simply makes use of the :class:`~.OutMultiplier` template to multiply
    the magnitudes of the encoded inputs, and a quantum comparator on their sign bits to determine
    the final sign of the result.

    The inputs and output are given in 2s complement.

    Args:
        x_wires (Sequence[int]): wires that store the signed integer :math:`x`
        y_wires (Sequence[int]): wires that store the signed integer :math:`y`
        output_wires (Sequence[int]): wires that store the multiplication result. If the
            register is in a non-zero state :math:`z`, the solution will be added to this value
        work_wires (Sequence[int]): auxiliary wires to use for the multiplication. The needed
            number of work wires depends on the decomposition, the register sizes and
            ``output_wires_zeroed``. Defaults to an empty tuple, i.e., no work wires.
        output_wires_zeroed (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially. Setting this argument to ``True`` reduces the cost of
            the operation.

    We begin with three signed quantum registers, storing :math:`|x\rangle`, :math:`|y\rangle` and :math:`|0\rangle` with register sizes :math:`n`, :math:`m` and :math:`k`, respectively. We will turn to the case of non-zero initial states for the last register later.
    Here, :math:`x` and :math:`y` are signed integers in two's complement representation:

    .. math::
        \begin{align}
            x = - 2^{n-1} x_{n-1} + \sum_{j=0}^{n-2} x_j 2^j.
        \end{align}
    We also have the magnitude of the signed integer which can be computed by flipping all bits of :math:`x` and incrementing by one,
    both steps controlled on the sign bit of :math:`x`, :math:`x_{n-1}`.

    The first step is to copy the sign bit of :math:`x` and :math:`y` to one auxiliary qubit each, and to compute the magnitude of the respective integer, as just described.
    At this point we have the state

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |0\rangle_s |0\rangle,
        \end{align}
    where we interleaved the two auxiliary qubits with the input registers and the output register, and wrote the sign bit of the output register as a separate qubit, marked with an :math:`s` for clarity.

    Next, we multiply the magnitude registers into the output register, obtaining

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |0\rangle_s |\bar{x}\bar{y}\rangle.
        \end{align}

    Then, we flip the sign bit of the output register controlled on the (cached) sign bits of each input, respectively:

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |x_{n-1}+y_{m-1} \rangle_s |\bar{x}\bar{y}\rangle.
        \end{align}

    From here on we write :math:`z_s = x_{n-1}+y_{m-1}`.
    Then, we flip and increment the (non-sign) bits of the output register controlled on the output sign bit to get (where :math:`k` is the size of the output register including the sign bit):

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |z_s \rangle_s |(-1)^{z_s}\bar{x}\bar{y}+2^{k - 1} z_s\rangle.
        \end{align}

    Arrived at by the following arithmetic.

    .. math::
        \begin{align}
            &(1 - z_s) \bar{x}\bar{y} + z_s (1 + \sum_{j=0}^{k-2} (1 - \bar{x}\bar{y}_j)2^j) \\
            &=(1 - z_s) \bar{x}\bar{y} + z_s(1 + \sum_{j=0}^{k-2}2^j - \sum_{j=0}^{k-2} \bar{x}\bar{y}_j2^j) \\
            &=(1 - z_s) \bar{x}\bar{y} + z_s(1 + 2^{k-1} - 1 - \bar{x}\bar{y}) \\
            &=(1 - z_s) \bar{x}\bar{y} + z_s (2^{k-1} - \bar{x}\bar{y}) \\\
            &=(-1)^{z_s}\bar{x}\bar{y}+2^{k - 1} z_s
        \end{align}
    Then we uncompute the magnitudes and the copied sign bits of the input registers, arriving at

    .. math::
        \begin{align}
            |x\rangle |0\rangle |y\rangle |0\rangle |z_s \rangle_s |(-1)^{z_s}\bar{x}\bar{y}+2^{k-1} z_s\rangle.
        \end{align}

    Interpreting the output register as signed integer, we find that we computed

    .. math::
        \begin{align}
            z &= (-1)^{z_s}\bar{x}\bar{y}+2^{k-1}z_s - 2^{k-1} z_s\\
            &=(-1)^{z_s} \bar{x}\bar{y} \\
            &=(-1)^{x_{n-1}}\bar{x} (-1)^{y_{m-1}}\bar{y}\\
            &= x y.
        \end{align}

    So we correctly arrive at the product of :math:`x` and :math:`y`.

    **Non-zero initial state of output wires**

    Next, let's consider the scenario where the output register starts out in a non-zero state :math:`|z\rangle`. In this case, we can make use of the `output_wires_zeroed` flag to the `OutMultiplier` that we employ in eqn. 15 to ensure that the magnitude is calculated into the output register correctly. This will affect the decomposition of the `OutMultiplier` yielding a more or less efficient circuit. The more efficient case is that of :math:`z = 0`.

    However, there is one caveat. We will always want the output register's sign bit to start in the :math:`\ket{0}_s` state. Therefore, we add a reset of this qubit when `output_wires_zeroed` is `False`.

    Otherwise, the circuit behaves in the same way. Let :math:`z_m` be the initial value of the output register, in the magnitude bits. Then, starting from line 15:

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |0\rangle_s |z_m + \bar{x}\bar{y}\rangle.
        \end{align}

    We flip the sign bit of the output register controlled on the (cached) sign bits of each input, respectively:

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |z_s \rangle_s |z_m + \bar{x}\bar{y}\rangle.
        \end{align}

    Next, we flip and increment the (non-sign) bits of the output register controlled on the output sign bit.

    .. math::
        \begin{align}
            &(1 - z_s) (\bar{x}\bar{y} + z_m) + z_s (1 + \sum_{j=0}^{k-2} (1 - (\bar{x}\bar{y} + z_m)_j)2^j) \\
            &=(1 - z_s) (\bar{x}\bar{y} + z_m) + z_s(1 + \sum_{j=0}^{k-2}2^j - \sum_{j=0}^{k-2} (\bar{x}\bar{y} + z_m)_j2^j \\
            &=(1 - z_s) (\bar{x}\bar{y} + z_m) + z_s(1 + 2^{k-1} - 1 - \bar{x}\bar{y} - z_m) \\
            &=(1 - z_s) (\bar{x}\bar{y} + z_m) + z_s (2^{k-1} - \bar{x}\bar{y} - z_m) \\\
            &=(-1)^{z_s}\bar{x}\bar{y} + 2^{k - 1} z_s + (-1)^{z_s} z_m
        \end{align}

    Yielding a state of:

    .. math::
        \begin{align}
            |\bar{x}\rangle |x_{n-1}\rangle |\bar{y}\rangle |y_{m-1}\rangle |z_s \rangle_s |(-1)^{z_s}\bar{x}\bar{y} + 2^{k - 1} z_s + (-1)^{z_s} z_m\rangle.
        \end{align}

    We uncompute the magnitudes and the copied sign bits of the input registers, arriving at

    .. math::
        \begin{align}
            |x\rangle |0\rangle |y\rangle |0\rangle |z_s \rangle_s |(-1)^{z_s}\bar{x}\bar{y} + 2^{k - 1} z_s + (-1)^{z_s} z_m\rangle.
        \end{align}

    This time, we find that we computed

    .. math::
        \begin{align}
            z &= (-1)^{z_s}\bar{x}\bar{y} + 2^{k - 1} z_s + (-1)^{z_s} z_m - 2^{k-1} z_s\\
            &=(-1)^{z_s} \bar{x}\bar{y} + (-1)^{z_s} z_m \\
            &=(-1)^{x_{n-1}}\bar{x} (-1)^{y_{m-1}}\bar{y} + (-1)^{z_s} z_m\\
            &= x y + (-1)^{z_s} z_m.
        \end{align}

    This is similar to how the `OutMultiplier` will calculate :math:`\ket{b + xy}` for the output register initialized with value :math:`b`.

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
            qp.SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)
            return qp.sample(wires=output_wires)

    >>> print(circuit())
    [[1 1 0 1 1 1]]

    The result :math:`[[1 1 0 1 1 1]]`, is the binary representation of
    :math:`-3 \cdot 3 \; = -9` in 2's complement form.
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
        work_wires: WiresLike = (),
        output_wires_zeroed: bool = False,
    ):  # pylint: disable=too-many-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        if len(work_wires) != 0:
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


def _signed_out_multiplier_resources(
    num_x_wires, num_y_wires, num_output_wires, num_work_wires, output_wires_zeroed
):
    """
    Computes the resources for the SignedOutMultiplier.
    Assumes the worst case that both numbers are negative.
    """
    resources = defaultdict(int)
    resources[controlled_resource_rep(PauliX, {}, 1, 0)] = (
        (num_x_wires - 1 + num_y_wires - 1) * 2 + num_output_wires - 1
    )
    resources[
        controlled_resource_rep(
            Incrementer,
            {"num_wires": num_x_wires + num_work_wires - 2, "num_work_wires": num_work_wires - 2},
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
            output_wires_zeroed=output_wires_zeroed,
        )
    ] = 1
    resources[
        controlled_resource_rep(
            Incrementer,
            {
                "num_wires": num_output_wires + num_work_wires - 3,
                "num_work_wires": num_work_wires - 2,
            },
            num_control_wires=1,
        )
    ] += 1
    resources[
        controlled_resource_rep(
            Incrementer,
            {"num_wires": num_y_wires + num_work_wires - 2, "num_work_wires": num_work_wires - 2},
            num_control_wires=1,
        )
    ] += 2
    resources[resource_rep(CNOT)] = 4

    if not output_wires_zeroed:
        resources[resource_rep(MidMeasure)] = 1

    return resources


def _twos_complement_helper(input_reg, aux_wire, work_wires, invert=False):
    r"""
    The magnitude of `input_reg` can be computed by flipping all bits of `input_reg` and incrementing by one,
    both steps controlled on the sign bit `aux_wire`. Any `work_wires` are used by the `Incrementer`.

    Args:
        input_reg: The register we want to take the 2s complement of.
        aux_wire: The cached sign bit which tells us whether we already have a magnitude.
        work_wires: Any work wires we can use in the decomposition.
        invert: Whether to invert the control.

    .. math::
        \begin{align}
            \bar{x}&=(1-x_{n-1})x+x_{n-1}\left(1 + \sum_{j=0}^{n-1} (1-x_j) 2^j\right)\\
            % &=2^n-x_u\\
            &=(x-x_{n-1} x) + x_{n-1} (1 + \sum_{j=0}^{n-1} 2^j - \sum_{j=0}^{n-1} x_j 2^j) \\
            &=(x-x_{n-1} x) + x_{n-1} (1 + \sum_{j=0}^{n-1} 2^j + 2^{n-1}x_{n-1} + \sum_{j=0}^{n-2} x_j 2^j) \\
            &=(x-x_{n-1} x) + x_{n-1} (1 + \sum_{j=0}^{n-1} 2^j - 2^{n-1}x_{n-1} + 2^n x_{n-1} + \sum_{j=0}^{n-2} x_j 2^j) \\
            &=(x-x_{n-1} x) + x_{n-1} (1 + (2^n - 1) - (x + 2^n x_{n-1})) \\
            &=(x-x_{n-1} x)+x_{n-1}2^n-x_{n-1}(x+2^nx_{n-1})\\
            &=x-2x_{n-1}x+2^nx_{n-1}(1-x_{n-1})\\
            &=(1-2x_{n-1})x + 2^n x_{n-1} - 2^n x_{n-1}^2 \\
            &=(1-2x_{n-1})x\\
            &=(-1)^{x_{n-1}}x.
        \end{align}
    """

    # Invert all bits
    @for_loop(len(input_reg))
    def invert(w):
        # sign bit of 1 indicates a negative value
        Controlled(PauliX(input_reg[w]), (aux_wire,), (not invert,))

    invert()  # pylint: disable=no-value-for-parameter

    # Add one
    Controlled(
        Incrementer(
            wires=input_reg + work_wires,
            work_wires=work_wires,  # we can use the work wires since they are returned in a clean state
        ),
        control_wires=(aux_wire,),
        control_values=(not invert,),
    )


def _work_wire_condition(num_work_wires, **_):
    return (
        num_work_wires >= 2
    )  # or max(len(x_wires), len(y_wires)) + 1 to use incrementer decomp with work wires


@register_condition(_work_wire_condition)
@register_resources(_signed_out_multiplier_resources, exact=False)
def _signed_out_multiplier_decomposition(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
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
    z_aux = work_wires[2]

    # Sign extension
    CNOT([x_wires[0], x_aux])
    CNOT([y_wires[0], y_aux])

    # Compute the final sign
    CNOT([x_aux, z_aux])
    CNOT([y_aux, z_aux])

    # Correct sign of z
    _twos_complement_helper(output_wires, z_aux, work_wires[3:], invert=True)

    # Take 2s complements if necessary
    _twos_complement_helper(x_wires, x_aux, work_wires[3:])
    _twos_complement_helper(y_wires, y_aux, work_wires[3:])

    # at this point the sign is only kept in the auxiliary qubits' states

    # Multiply the magnitudes
    OutMultiplier(
        x_wires,
        y_wires,
        output_wires[1:],
        work_wires=work_wires[3:],
        output_wires_zeroed=output_wires_zeroed,
    )

    measure(output_wires[0], reset=True)

    # Compute the final sign
    CNOT([x_aux, output_wires[0]])
    CNOT([y_aux, output_wires[0]])

    # Encode the output
    _twos_complement_helper(output_wires, z_aux, work_wires[3:])

    # Return inputs to original state
    _twos_complement_helper(x_wires, x_aux, work_wires[3:])
    _twos_complement_helper(y_wires, y_aux, work_wires[3:])

    # Uncompute sign extension
    CNOT([x_wires[0], x_aux])
    CNOT([y_wires[0], y_aux])


add_decomps(SignedOutMultiplier, _signed_out_multiplier_decomposition)
