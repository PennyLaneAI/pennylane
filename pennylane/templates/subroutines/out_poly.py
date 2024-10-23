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

import pennylane as qml
from pennylane.operation import Operation


def _get_polynomial(f, mod, *variable_sizes):
    """Calculate the polynomial binary representation of a given function using the `Möbius inversion formula <https://en.wikipedia.org/wiki/Möbius_inversion_formula#On_posets>`_ .

    Args:
        f (callable):  the function from which the polynomial is extracted.
        mod (int): the modulus to use for the result
        *variable_sizes (int):  variable length argument specifying the number of bits used to represent each of the variables of the function

    Return:
        dict: A dictionary where each key is a tuple representing the variable terms of the polynomial (if the term includes the i-th variable, a 1 appears in the i-th position).
              Each key is a tuple containing a bitstring representing a term in the binary polynomial

    Example:
        For the function `f(x, y) = 4 * x * y` with `variable_sizes=(2, 1)` and `mod=5`, the target polynomial is `4 * (2x_0 + x_1) * y_0` that
        is expanded as `4 * x1 * y0 + 8 * x0 * y0`. Therefore, the expected output is:

        ```
        # (x0, x1, y0) -> coefficient
        {
            (0, 1, 1): 4,
            (1, 0, 1): 3,  # 8 mod 5 = 3
        }
        ```

        In this example, the first two bits correspond to the binary representation of the first variable and the last
        bit corresponds to the binary representation of the second variable.
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

        # (f_values is the zeta transform of the target polynomial)
        f_values[s] = f(*decimal_values) % mod

    f_values = _mobius_inversion_of_zeta_transform(f_values, mod)

    coeffs_dict = {}
    for s, f_value in enumerate(f_values):
        if not qml.math.isclose(f_value, 0.0):
            bin_tuple = tuple(all_binary_list[s])
            coeffs_dict[bin_tuple] = f_value

    return coeffs_dict


def _mobius_inversion_of_zeta_transform(f_values, mod):
    """Applies the `Möbius inversion <https://codeforces.com/blog/entry/72488>`_ to a zeta transform.
    The input `f_values` is a list of integers representing a zeta transform
    over subsets of a bitmask. This function performs the Möbius inversion
    of the zeta transform by subtracting terms to recover the original
    values before the transform.

    Args:
        f_values (list): A list of integers representing the zeta transform.
        mod (int): The modulus to be used in the calculations.

    Returns:
        list: The list `f_values` after applying the Möbius inversion.
    """

    total_wires = int(qml.math.log2(len(f_values)))
    num_combinations = len(f_values)

    for i in range(total_wires):
        ith_bit_on = 1 << i
        for mask in range(num_combinations):
            if mask & ith_bit_on:
                f_values[mask] = (f_values[mask] - f_values[mask ^ ith_bit_on]) % mod

    return f_values


class OutPoly(Operation):
    r"""Performs the out-of-place polynomial operation.

    This class implements an out-of-place operation that computes a polynomial function
    over a set of input registers and stores the result in an output register. The result
    is computed modulo a given value.

    Given a function :math:`f(x_1, \dots, x_m)` and a modulus :math:`k`, the operator performs:

    .. math::

        \text{OutPoly}_{f, k} |x_1 \rangle \dots |x_m \rangle |0 \rangle
        = |x_1 \rangle \dots |x_m \rangle |f(x_1, \dots, x_m)\, \text{mod}\, k\rangle.

    This operation leaves the input registers unchanged and stores the result of the
    polynomial function in the output register. It is based on the idea detailed
    in Section II-B of `arXiv:2112.10537 <https://arxiv.org/abs/2112.10537>`_.

    .. note::

        The integer values :math:`x_i` stored in each input register must
        be smaller than the modulus `mod`.

    Args:
        polynomial_function (callable): The polynomial function to be applied to the inputs. It must accept the same number of arguments as there are input registers.
        input_registers (Sequence[WiresLike]): Tuple whose elements are the wires used to store each variable of the polynomial.
        output_wires (Sequence[int]): The wires used to store the output of the operation.
        mod (int, optional): The modulus to use for the result stored in the output register. If not provided, it defaults to :math:`2^{n}`, where :math:`n` is the number of qubits in the output register.
        work_wires (Sequence[int], optional): The auxiliary wires used for intermediate computation, if necessary. If `mod` is not a power of two, then two auxiliary work wires are required.
        id (str or None, optional): The name of the operation.

    Raises:
        ValueError: If `mod` is not a power of 2 and no or insufficient work wires are provided.
        ValueError: If the wires used in the input and output registers overlap.
        ValueError: If the function is not defined with integer coefficients.

    Example:
        Given a polynomial function :math:`f(x, y) = x^2 + y`,
        we can calculate :math:`f(3, 2)` as follows:

        .. code-block:: python

            reg = qml.registers({"x_wires": 3, "y_wires": 3, "output_wires": 4})

            def f(x, y):
                return x ** 2 + y

            @qml.qnode(qml.device("default.qubit", shots = 1))
            def circuit():
                # loading values for x and y
                qml.BasisEmbedding(3, wires=reg["x_wires"])
                qml.BasisEmbedding(2, wires=reg["y_wires"])

                # applying the polynomial
                qml.OutPoly(
                    f,
                    input_registers = (reg["x_wires"], reg["y_wires"]),
                    output_wires = reg["output_wires"])

                return qml.sample(wires=reg["output_wires"])

            print(circuit())

        .. code-block:: pycon

            >>> print(circuit())
            [1 0 1 1]

        The result, :math:`[1 0 1 1]`, is the binary representation of :math:`3^2 + 2 = 11`.
        Note that by not specifying `mod`, the default value :math:`2^{\text{len(output_wires)}} = 2^4 = 16` is used.
        For more information on using `mod`, see the Usage Details section below.


    .. seealso:: :class:`~.PhaseAdder`

    .. details::
        :title: Usage Details

        This template can take a modulus different from powers of two. In these cases it should be provided two auxiliary qubits.

        .. code-block:: python

            x_wires = [0, 1, 2]
            y_wires = [3, 4, 5]
            input_registers = [x_wires, y_wires]

            output_wires = [6, 7, 8]
            work_wires = [9,10]


            def f(x, y):
                return x ** 2 + y

            @qml.qnode(qml.device("default.qubit", shots = 1))
            def circuit():
                # loading values for x and y
                qml.BasisEmbedding(3, wires=wires_x)
                qml.BasisEmbedding(2, wires=wires_y)

                # applying the polynomial
                qml.OutPoly(
                    f,
                    input_registers = input_registers,
                    output_wires = output_wires,
                    mod = 7,
                    work_wires = work_wires
                )

                return qml.sample(wires=output_wires)

        .. code-block:: pycon

            >>> print(circuit())
            [1 0 0]

        The result, :math:`[1 0 0]`, is the binary representation of :math:`3^2 + 2  \; \text{modulo} \; 7 = 4`.
        If the output wires are not initialized to zero, this value will be added to the solution. Generically, the expression is definded as:

        .. math::

            \text{OutPoly}_{f, \text{mod}} |x_1 \rangle \dots |x_m \rangle |b \rangle
            = |x_1 \rangle \dots |x_m \rangle |b + f(x_1, \dots, x_m) \mod \text{mod} \rangle.

    """

    grad_method = None

    def __init__(
        self, polynomial_function, input_registers, output_wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments
        r"""Initialize the OutPoly class"""

        registers_wires = [*input_registers, output_wires]

        num_work_wires = 0 if not work_wires else len(work_wires)
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
            wires = qml.wires.Wires(reg)
            inp_regs.append(wires)
            all_wires += wires

        self.hyperparameters["input_registers"] = tuple(inp_regs)

        wires = qml.wires.Wires(output_wires)
        self.hyperparameters["output_wires"] = wires
        all_wires += wires

        self.hyperparameters["polynomial_function"] = polynomial_function
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires) if work_wires else None

        if work_wires:
            all_wires += work_wires

        if len(all_wires) != sum(len(register) for register in registers_wires) + num_work_wires:
            raise ValueError("A wire appeared in multiple registers.")

        super().__init__(wires=all_wires, id=id)

    def _flatten(self):
        metadata1 = tuple((key, value) for key, value in self.hyperparameters.items())

        return tuple(), metadata1

    @classmethod
    def _unflatten(cls, data, metadata):

        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def map_wires(self, wire_map: dict):

        new_input_registers = [
            qml.wires.Wires([wire_map[wire] for wire in reg])
            for reg in self.hyperparameters["input_registers"]
        ]

        new_output_wires = [wire_map[wire] for wire in self.hyperparameters["output_wires"]]

        new_work_wires = (
            [wire_map[wire] for wire in self.hyperparameters["work_wires"]]
            if self.hyperparameters.get("work_wires")
            else None
        )

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

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(**self.hyperparameters)

    @staticmethod
    def compute_decomposition(
        polynomial_function, input_registers, output_wires, mod=None, work_wires=None
    ):  # pylint: disable=unused-argument, arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.OutPoly.decomposition`.

        **Example:**

        .. code-block:: python

            print(
            qml.OutPoly.compute_decomposition(
                lambda x, y: x + y,
                input_registers=[[0, 1],[2,3]],
                output_wires=[4, 5],
                mod=4,
                )
            )

        .. code-block:: pycon

        [QFT(wires=[4]), Controlled(PhaseAdder(wires=[4, None]), control_wires=[3]), Controlled(PhaseAdder(wires=[4, None]), control_wires=[1]), Adjoint(QFT(wires=[4]))]
        """
        registers_wires = [*input_registers, output_wires]

        if not work_wires:
            work_wires = [None, None]

        list_ops = []

        output_adder_mod = (
            [work_wires[0]] + registers_wires[-1] if work_wires[0] else registers_wires[-1]
        )

        list_ops.append(qml.QFT(wires=output_adder_mod))

        wires_vars = [len(w) for w in registers_wires[:-1]]

        # Extract the coefficients and control wires from the binary polynomial
        coeffs_dic = _get_polynomial(polynomial_function, mod, *wires_vars)
        coeffs = [coeff[1] for coeff in coeffs_dic.items()]

        assert qml.math.allclose(
            coeffs, qml.math.floor(coeffs)
        ), "The polynomial function must have integer coefficients"

        all_wires_input = sum([*registers_wires[:-1]], start=[])

        for item, coeff in coeffs_dic.items():

            if not 1 in item:
                # Add the independent term
                list_ops.append(qml.PhaseAdder(int(coeff), output_adder_mod))
            else:
                controls = [all_wires_input[i] for i, bit in enumerate(item) if bit == 1]

                list_ops.append(
                    qml.ctrl(
                        qml.PhaseAdder(
                            int(coeff) % mod,
                            output_adder_mod,
                            work_wire=work_wires[1],
                            mod=mod,
                        ),
                        control=controls,
                    )
                )

        list_ops.append(qml.adjoint(qml.QFT)(wires=output_adder_mod))

        return list_ops
