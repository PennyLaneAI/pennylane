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
        f (callable):  the function from which the polynomial is extracted
        mod (int): the modulus to use for the result
        *variable_sizes (int):  variable length argument specifying the number of bits used to represent each of the variables of the function

    Return:
        dict: A dictionary where each key is a tuple representing the variable terms of the polynomial (if the term includes the i-th variable, a 1 appears in the i-th position).
              Each value is the corresponding coefficient for that term.

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

    # Calculate the list with all possible keys
    all_binary_list = [list(map(int, bin(i)[2:].zfill(total_wires))) for i in range(num_combinations)]

    # Compute the f values for all combinations (2 ** len(total_wires))
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

    # Compute the Möbius inversion of f_values
    for i in range(total_wires):
        ith_bit_on = 1 << i
        for mask in range(num_combinations):
            if mask & ith_bit_on:
                f_values[mask] = (f_values[mask] - f_values[mask ^ ith_bit_on]) % mod

    # Adjust the algorithm result to the dictionary output format
    coeffs_dict = {}
    for s, f_value in enumerate(f_values):
        if f_value != 0:
            bin_tuple = tuple(all_binary_list[s])
            coeffs_dict[bin_tuple] = f_value

    return coeffs_dict


class OutPoly(Operation):
    r"""Performs the out-place polynomial operation.

    This class implements an out-of-place operation that computes a polynomial function
    over a set of input registers and stores the result in an output register. The result
    is computed modulo a given value.

    Given a function :math:`f(x_1, \dots, x_m)` and a modulus `k`, the operator performs:

    .. math::

        \text{OutPoly}_{f, k} |x_1 \rangle \dots |x_m \rangle |0 \rangle
        = |x_1 \rangle \dots |x_m \rangle |f(x_1, \dots, x_m)\, \text{mod}\, k\rangle.

    This operation leaves the input registers unchanged and stores the result of the
    polynomial function in the output register. It is based on the idea detailed
    in `arXiv:2112.10537 <https://arxiv.org/abs/2112.10537>`_ section II-B.

    .. note::

        To obtain the correct result, the values of the input registers :math:`x_i` must
        be smaller than the modulus `mod`.

    Args:
        f (callable): The polynomial function to be applied to the inputs. It must accept the same number of arguments as there are input registers.
        output_wires (Sequence[int]): The wires used to store the output of the operation.
        mod (int, optional): The modulus to use for the result. If not provided, it defaults to :math:`2^{n}`, where `n` is the number of qubits in the output register.
        work_wires (Sequence[int], optional): The auxiliary wires used for intermediate computation, if necessary. If `mod` is not a power of two, then two auxiliary work wires are required.
        id (str or None, optional): The name of the operation.
        **kwargs: the wires associated with the function arguments. That is to say, if the polynomial takes two arguments, we will need to send two keyword arguments indicating the wires we will use to represent each argument. in the example below, the kwargs are ``x_wires`` and ``y_wires``.

    Raises:
        ValueError: If `mod` is not a power of 2 and no or insufficient work wires are provided.
        ValueError: If the wires used in the input and output registers overlap.

    Example:
        Given a polynomial function :math:`f(x, y) = x^2 + y`,
        we can apply this operation as follows:

        .. code-block:: python

            x_wires = [0, 1, 2]
            y_wires = [3, 4, 5]
            output_wires = [6, 7, 8, 9]

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
                    x_wires = x_wires,
                    y_wires = y_wires,
                    output_wires = output_wires)

                return qml.sample(wires=output_wires)

            print(circuit())

        .. code-block:: pycon

            >>> print(circuit())
            [1 0 1 1]

        The result, :math:`[1 0 1 1]`, is the binary representation of :math:`3^2 + 2 = 11`.
        Note that by not specifying `mod`, the default value :math:`2 ^{\text{len(output_wires)}}` is used.
        In the usage details it is shown an example where a specific modulus is used.


    .. seealso:: :class:`~.PhaseAdder`

    .. details::
        :title: Usage Details

        This template can take a modulus different from powers of two. In these cases it should be provided two auxiliary qubits.

        .. code-block:: python

            x_wires = [0, 1, 2]
            y_wires = [3, 4, 5]
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
                    x_wires = x_wires,
                    y_wires = y_wires,
                    output_wires = output_wires,
                    mod = 7,
                    work_wires = work_wires)

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
        self, f, output_wires, mod=None, work_wires=None, id=None, **kwargs
    ):  # pylint: disable=too-many-arguments
        r"""Initialize the OutPoly class"""

        registers_wires = [*kwargs.values(), output_wires]

        num_work_wires = 0 if not work_wires else len(work_wires)
        if mod is None:
            mod = 2 ** len(registers_wires[-1])
        elif mod != 2 ** len(registers_wires[-1]) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(registers_wires[-1])}, two work wires should be provided"
            )

        if not isinstance(mod, int):
            raise ValueError("mod must be integer.")

        self.hyperparameters["f"] = f

        all_wires = []
        self.hyperparameters["registers_wires"] = {}

        for key, value in kwargs.items():
            wires = qml.wires.Wires(value)
            self.hyperparameters["registers_wires"][key] = wires
            all_wires += wires

        wires = qml.wires.Wires(output_wires)
        self.hyperparameters["registers_wires"]["output_wires"] = wires
        all_wires += wires

        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires) if work_wires else None

        if work_wires:
            all_wires += work_wires

        if len(all_wires) != sum(len(register) for register in registers_wires) + num_work_wires:
            raise ValueError(
                "None of the wires in a register must be contained in another register."
            )

        super().__init__(wires=all_wires, id=id)

    def _flatten(self):
        metadata1 = tuple(
            (key, value) for key, value in self.hyperparameters.items() if key != "registers_wires"
        )
        metadata2 = tuple(self.hyperparameters["registers_wires"].items())

        return tuple(), (*metadata1, *metadata2)

    @classmethod
    def _unflatten(cls, data, metadata):

        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def map_wires(self, wire_map: dict):

        new_registers_wires = {
            key: qml.wires.Wires([wire_map[wire] for wire in wires])
            for key, wires in self.hyperparameters["registers_wires"].items()
        }

        new_work_wires = (
            [wire_map[wire] for wire in self.hyperparameters["work_wires"]]
            if self.hyperparameters.get("work_wires")
            else None
        )

        return OutPoly(
            f=self.hyperparameters["f"],
            mod=self.hyperparameters["mod"],
            work_wires=new_work_wires,
            **new_registers_wires,
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(**self.hyperparameters)

    @staticmethod
    def compute_decomposition(**kwargs):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.OutPoly.decomposition`.

        **Example:**

        .. code-block:: python

            kwargs = {
                "f": lambda x, y: x + y,
                "registers_wires": {"x": [0, 1], "y": [2, 3], "output_wires": [4]},
                "mod": 2,
                "work_wires": None,
            }

        .. code-block:: pycon

        >>> print(qml.OutPoly.compute_decomposition(**kwargs))
        [QFT(wires=[4]), Controlled(PhaseAdder(wires=[4, None]), control_wires=[3]), Controlled(PhaseAdder(wires=[4, None]), control_wires=[1]), Adjoint(QFT(wires=[4]))]

        """

        f = kwargs["f"]
        mod = kwargs["mod"]

        registers_wires = list(kwargs["registers_wires"].values())
        work_wires = kwargs["work_wires"]

        if not work_wires:
            work_wires = [None, None]

        list_ops = []

        output_adder_mod = (
            [work_wires[0]] + registers_wires[-1] if work_wires[0] else registers_wires[-1]
        )

        list_ops.append(qml.QFT(wires=output_adder_mod))

        wires_vars = [len(w) for w in registers_wires[:-1]]

        # Extract the coefficients and control wires from the binary polynomial
        coeffs_list = _get_polynomial(f, mod, *wires_vars)

        all_wires_input = sum([*registers_wires[:-1]], start=[])

        for item, coeff in coeffs_list.items():

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
