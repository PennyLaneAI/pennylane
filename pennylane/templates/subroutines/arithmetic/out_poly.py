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
Contains the OutPoly template.
"""
from collections import Counter

from pennylane import math
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import adjoint, ctrl
from pennylane.templates.subroutines.qft import QFT
from pennylane.wires import Wires, WiresLike

from .phase_adder import PhaseAdder


def _get_polynomial(f, mod, *variable_sizes):
    """Calculate the polynomial binary representation of a given function using
    the `Möbius inversion formula <https://en.wikipedia.org/wiki/Möbius_inversion_formula#On_posets>`_ .

    Args:
        f (callable):  the function from which the polynomial is extracted
        mod (int): the modulus to use for the arithmetic operation
        *variable_sizes (int):  variadic arguments that specify the number of bits needed to represent each
                                of the variables of the function

    Returns:
        dict[tuple -> int]: dictionary with keys representing the variable terms of the polynomial
              and values representing the coefficients associated with those terms

    Example:
        For the function `f(x, y) = 4 * x * y`, setting `variable_sizes=(2, 1)`
        means that `x` is represented by two bits and `y` by one bit.

        We can expand `f(x, y)` as `4 * (2x_0 + x_1) * y_0`, where `x_0` and `x_1` are the binary digits of `x`.
        When fully expanded, the function becomes:

        `4 * x1 * y0 + 8 * x0 * y0`.

        Applying modulus 5, this can be represented as

        ```
        {
            (0, 1, 1): 4,  # represents the term x1 * y0 with coefficient 4
            (1, 0, 1): 3   # represents the term x0 * y0 with coefficient 3 because 8 mod 5 = 3
        }
        ```

        Note that in each tuple, the first two bits correspond to the binary representation of the variable `x` and
        the third bit corresponds to the binary representation of the variable `y`. For example, `(0, 1, 1)`
        means `x1 * y0`, since `x0` is absent the first number in the tuple is zero.
    """

    total_wires = sum(variable_sizes)
    num_combinations = 2**total_wires

    all_binary_list = [
        list(map(int, bin(i)[2:].zfill(total_wires))) for i in range(num_combinations)
    ]

    f_values = [0] * num_combinations
    for s in range(num_combinations):
        bin_list = all_binary_list[s]
        decimal_values = []

        start = 0
        for wire_length in variable_sizes:
            segment = bin_list[
                start : start + wire_length
            ]  # segment corresponding to the i-th variable
            decimal = int(
                "".join(map(str, segment)), 2
            )  # decimal representation of the i-th variable
            decimal_values.append(decimal)
            start += wire_length

        # compute the zeta transform of the target polynomial
        f_values[s] = f(*decimal_values) % mod

    f_values = _mobius_inversion_of_zeta_transform(f_values, mod)

    coeffs_dict = {}
    for s, f_value in enumerate(f_values):
        if not math.isclose(f_value, 0.0):
            bin_tuple = tuple(all_binary_list[s])
            coeffs_dict[bin_tuple] = f_value

    return coeffs_dict


def _mobius_inversion_of_zeta_transform(f_values, mod):
    """
    Applies the `Möbius inversion <https://codeforces.com/blog/entry/72488>`_ to reverse the zeta
    transform and recover the original function values.

    The function loops over each bit position (from `total_wires`) and adjusts the values in `f_values`
    based on whether the corresponding bit in the bitmask is set (1) or not (0). The bitmask is a sequence of bits
    used to enable (1) or disable (0) specific positions in binary operations, allowing individual bits in data
    structures or numeric values to be checked or modified. The aim is to reverse
    the effect of the zeta transform by subtracting contributions from supersets in each step, effectively
    recovering the original values before the transformation.

    Args:
        f_values (List[int]): A list of integers representing the zeta transform over subsets of a bitmask.
        mod (int): The modulus used to perform the arithmetic operations.

    Returns:
        List[int]: The list `f_values` after applying the Möbius inversion, reduced modulo `mod`.

    """

    total_wires = int(math.log2(len(f_values)))
    num_combinations = len(f_values)

    for i in range(total_wires):
        ith_bit_on = 1 << i
        for mask in range(num_combinations):
            if mask & ith_bit_on:
                f_values[mask] = (f_values[mask] - f_values[mask ^ ith_bit_on]) % mod

    return f_values


class OutPoly(Operation):
    r"""Performs the out-of-place polynomial operation.

    Given a function :math:`f(x_1, \dots, x_m)` and an integer modulus :math:`mod`, this operator performs:

    .. math::

        \text{OutPoly}_{f, mod} |x_1 \rangle \dots |x_m \rangle |0 \rangle
        = |x_1 \rangle \dots |x_m \rangle |f(x_1, \dots, x_m)\, \text{mod} \; mod\rangle,

    where the integer inputs :math:`x_i` are embedded in the ``input_registers``. The result of the
    polynomial function :math:`f(x_1, \dots, x_m)` is computed modulo :math:`mod` in the computational
    basis and stored in the ``output_wires``. If the output wires are not initialized to zero, the evaluated
    result :math:`f(x_1, \dots, x_m)\ \text{mod}\ mod` will be added to the value initialized in the output register.
    This implementation is based on the Section II-B of `arXiv:2112.10537 <https://arxiv.org/abs/2112.10537>`_.


    .. note::

        The integer values :math:`x_i` stored in each input register must
        be smaller than the modulus ``mod``.

    Args:

        polynomial_function (callable): The polynomial function to be applied. The number of arguments in the function
            must be equal to the number of input registers.
        input_registers (List[Union[Wires, Sequence[int]]]): List containing the wires (or the wire indices) used to
            store each variable of the polynomial.
        output_wires (Union[Wires, Sequence[int]]): The wires (or wire indices) used to store the output of the operation.
        mod (int, optional): The integer for performing the modulo on the result of the polynomial operation. If not provided,
            it defaults to :math:`2^{n}`, where :math:`n` is the number of qubits in the output register.
        work_wires (Union[Wires, Sequence[int]], optional): The auxiliary wires to use for performing the polynomial operation.
            The work wires are not needed if :math:`mod=2^{\text{length(output_wires)}}`, otherwise two work wires should be
            provided. Defaults to empty tuple.

    Raises:
        ValueError: If `mod` is not :math:`2^{\text{length(output_wires)}}` and insufficient number of work wires are provided.
        ValueError: If the wires used in the input and output registers overlap.
        ValueError: If the function is not defined with integer coefficients.

    Example:
        Given a polynomial function :math:`f(x, y) = x^2 + y`,
        we can calculate :math:`f(3, 2)` as follows:

        .. code-block:: python

            wires = qml.registers({"x": 2, "y": 2, "output": 4})

            def f(x, y):
                return x ** 2 + y

            @qml.qnode(qml.device("default.qubit"), shots=1)
            def circuit():
                # load values of x and y
                qml.BasisEmbedding(3, wires=wires["x"])
                qml.BasisEmbedding(2, wires=wires["y"])

                # apply the polynomial
                qml.OutPoly(
                    f,
                    input_registers = [wires["x"], wires["y"]],
                    output_wires = wires["output"])

                return qml.sample(wires=wires["output"])

        >>> print(circuit())
        [[1 0 1 1]]

        The result, :math:`[[1 0 1 1]]`, is the binary representation of :math:`3^2 + 2 = 11`.
        Note that the default value of `mod` in this example is :math:`2^{\text{len(output_wires)}} = 2^4 = 16`.
        For more information on using `mod`, see the Usage Details section.

    .. seealso:: The decomposition of this operator consists of controlled :class:`~.PhaseAdder` gates.

    .. details::
        :title: Usage Details

        If the value of `mod` is not :math:`2^{\text{length(output_wires)}}`, then two auxiliary qubits must be provided.

        .. code-block:: python

            x_wires = [0, 1, 2]
            y_wires = [3, 4, 5]
            input_registers = [x_wires, y_wires]

            output_wires = [6, 7, 8]
            work_wires = [9,10]


            def f(x, y):
                return x ** 2 + y

            @qml.qnode(qml.device("default.qubit"), shots=1)
            def circuit():
                # loading values for x and y
                qml.BasisEmbedding(3, wires=x_wires)
                qml.BasisEmbedding(2, wires=y_wires)
                qml.BasisEmbedding(1, wires=output_wires)

                # applying the polynomial
                qml.OutPoly(
                    f,
                    input_registers,
                    output_wires,
                    mod = 7,
                    work_wires = work_wires
                )

                return qml.sample(wires=output_wires)

        >>> print(circuit())
        [[1 0 1]]

        The result, :math:`[[1 0 1]]`, is the binary representation
        of :math:`1 + f(3, 2) = 1 + 3^2 + 2  \; \text{mod} \; 7 = 5`.
        In this example ``output_wires`` is initialized to :math:`1`, so this value is added to the solution.
        Generically, the expression is definded as:

        .. math::

            \text{OutPoly}_{f, mod} |x_1 \rangle \dots |x_m \rangle |b \rangle
            = |x_1 \rangle \dots |x_m \rangle |b + f(x_1, \dots, x_m) \mod mod \rangle.

    """

    grad_method = None

    resource_keys = {"num_output_wires", "num_work_wires", "mod", "coeffs_list"}

    def __init__(
        self,
        polynomial_function,
        input_registers,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        id=None,
        **kwargs,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        r"""Initialize the OutPoly class"""

        registers_wires = [*input_registers, output_wires]

        work_wires = Wires(() if work_wires is None else work_wires)
        num_work_wires = len(work_wires)
        if mod is None:
            mod = 2 ** len(registers_wires[-1])
        elif mod != 2 ** len(registers_wires[-1]) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(registers_wires[-1])}, two work wires should be provided"
            )

        if not isinstance(mod, int):
            raise ValueError("mod must be an integer.")

        all_wires = []
        inp_regs = []

        for reg in input_registers:
            wires = Wires(reg)
            inp_regs.append(wires)
            all_wires += wires

        self.hyperparameters["input_registers"] = tuple(inp_regs)

        wires = Wires(output_wires)
        self.hyperparameters["output_wires"] = wires
        all_wires += wires

        self.hyperparameters["polynomial_function"] = polynomial_function
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = work_wires

        wires_vars = [len(w) for w in registers_wires[:-1]]

        self.hyperparameters["coeffs_list"] = kwargs.get(
            "coeffs_list",
            tuple(
                (key, value)
                for key, value in _get_polynomial(polynomial_function, mod, *wires_vars).items()
            ),
        )

        coeffs = [c[1] for c in self.hyperparameters["coeffs_list"]]
        assert math.allclose(
            coeffs, math.floor(coeffs)
        ), "The polynomial function must have integer coefficients"

        if len(work_wires) != 0:
            all_wires += work_wires

        if len(all_wires) != sum(len(register) for register in registers_wires) + num_work_wires:
            raise ValueError(
                "None of the wires in a register should be included in other register."
            )

        super().__init__(wires=all_wires, id=id)

    def _flatten(self):
        metadata1 = tuple((key, value) for key, value in self.hyperparameters.items())

        return tuple(), metadata1

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    @property
    def resource_params(self) -> dict:
        return {
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "mod": self.hyperparameters["mod"],
            "coeffs_list": self.hyperparameters["coeffs_list"],
        }

    def map_wires(self, wire_map: dict):

        new_input_registers = [
            Wires([wire_map[wire] for wire in reg])
            for reg in self.hyperparameters["input_registers"]
        ]

        new_output_wires = [wire_map[wire] for wire in self.hyperparameters["output_wires"]]

        new_work_wires = [wire_map[wire] for wire in self.hyperparameters["work_wires"]]

        return OutPoly(
            polynomial_function=self.hyperparameters["polynomial_function"],
            input_registers=new_input_registers,
            output_wires=new_output_wires,
            mod=self.hyperparameters["mod"],
            work_wires=new_work_wires,
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def decomposition(self):
        return self.compute_decomposition(**self.hyperparameters)

    @staticmethod
    def compute_decomposition(
        polynomial_function,
        input_registers,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        **kwargs,
    ):  # pylint: disable=unused-argument, arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.OutPoly.decomposition`.

        **Example:**

        .. code-block:: python

            from pprint import pprint

            ops = qml.OutPoly.compute_decomposition(
                lambda x, y: x + y,
                input_registers=[[0, 1],[2,3]],
                output_wires=[4, 5],
                mod=4,
                )
            pprint(ops)

        .. code-block::

            [QFT(wires=[4, 5]),
            Controlled(PhaseAdder(wires=[4, 5]), control_wires=[3]),
            Controlled(PhaseAdder(wires=[4, 5]), control_wires=[2]),
            Controlled(PhaseAdder(wires=[4, 5]), control_wires=[1]),
            Controlled(PhaseAdder(wires=[4, 5]), control_wires=[0]),
            Adjoint(QFT(wires=[4, 5]))]
        """
        registers_wires = [*input_registers, output_wires]

        if len(work_wires) == 0:
            work_wires = [(), ()]

        list_ops = []

        output_adder_mod = (
            [work_wires[0]] + registers_wires[-1] if work_wires[0] else registers_wires[-1]
        )

        list_ops.append(QFT(wires=output_adder_mod))

        wires_vars = [len(w) for w in registers_wires[:-1]]

        coeffs_list = kwargs.get("coeffs_list")
        if coeffs_list is None:
            coeffs_dic = _get_polynomial(polynomial_function, mod, *wires_vars)
        else:
            coeffs_dic = dict(kwargs["coeffs_list"])

        all_wires_input = sum([*registers_wires[:-1]], start=[])

        for item, coeff in coeffs_dic.items():

            if not 1 in item:
                # Add the constant term
                list_ops.append(PhaseAdder(int(coeff), output_adder_mod))
            else:
                controls = [all_wires_input[i] for i, bit in enumerate(item) if bit == 1]

                list_ops.append(
                    ctrl(
                        PhaseAdder(
                            int(coeff) % mod,
                            output_adder_mod,
                            work_wire=work_wires[1],
                            mod=mod,
                        ),
                        control=controls,
                    )
                )

        list_ops.append(adjoint(QFT)(wires=output_adder_mod))

        return list_ops


def _out_poly_decomposition_resources(num_output_wires, num_work_wires, mod, coeffs_list) -> dict:
    num_output_adder_mod = num_output_wires + 1 if num_work_wires else num_output_wires

    resources = Counter(
        {
            resource_rep(QFT, num_wires=num_output_adder_mod): 1,
        }
    )

    coeffs_dic = dict(coeffs_list)

    for item in coeffs_dic:

        if 1 not in item:
            # `num_output_adder_mod` will always correspond to log2(mod) so we don't need to provide
            # `mod` to the `PhaseAdder` in the decomposition.
            rep = resource_rep(PhaseAdder, num_x_wires=num_output_adder_mod, mod=mod)
            resources[rep] += 1
        else:
            num_controls = sum(1 for bit in item if bit == 1)

            ctrl_phase_rep = controlled_resource_rep(
                base_class=PhaseAdder,
                base_params={"num_x_wires": num_output_adder_mod, "mod": mod},
                num_control_wires=num_controls,
                num_zero_control_values=0,
                num_work_wires=int(num_work_wires > 0),
                work_wire_type="borrowed",
            )
            resources[ctrl_phase_rep] += 1

    resources[adjoint_resource_rep(QFT, {"num_wires": num_output_adder_mod})] = 1

    return dict(resources)


@register_resources(_out_poly_decomposition_resources)
def _out_poly_decomposition(
    polynomial_function,
    input_registers,
    output_wires: WiresLike,
    mod=None,
    work_wires: WiresLike = (),
    **kwargs,
):  # pylint: disable=unused-argument
    registers_wires = [*input_registers, output_wires]

    if len(work_wires) == 0:
        work_wires = [(), ()]

    output_adder_mod = (
        [work_wires[0]] + registers_wires[-1] if work_wires[0] else registers_wires[-1]
    )

    QFT(wires=output_adder_mod)

    coeffs_dic = dict(kwargs["coeffs_list"])

    all_wires_input = sum([*registers_wires[:-1]], start=[])

    for item, coeff in coeffs_dic.items():

        if not 1 in item:
            # Add the constant term
            PhaseAdder(int(coeff), output_adder_mod)
        else:
            controls = [all_wires_input[i] for i, bit in enumerate(item) if bit == 1]

            ctrl(
                PhaseAdder(
                    int(coeff) % mod,
                    output_adder_mod,
                    work_wire=work_wires[1],
                    mod=mod,
                ),
                control=controls,
            )

    adjoint(QFT(wires=output_adder_mod))


add_decomps(OutPoly, _out_poly_decomposition)
