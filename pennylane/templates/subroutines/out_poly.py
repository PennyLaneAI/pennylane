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

import inspect

import pennylane as qml
from pennylane.operation import Operation


def _binary_to_decimal(binary_list):
    """Convert a binary list to its decimal value."""
    return sum(val * (2**idx) for idx, val in enumerate(reversed(binary_list)))


def _decimal_to_binary_list(n, length):
    """Convert a decimal number to a binary list of a given length."""
    return [int(x) for x in bin(n)[2:].zfill(length)]


def _get_coefficients_and_controls(f, mod, *wire_lengths):
    """Calculate the coefficients and controls for a given function and wire configuration using Fast Möbius Transform."""
    total_wires = sum(wire_lengths)
    num_combinations = 2**total_wires  # Total number of combinations (2^total_wires)

    # Compute function values for all combinations
    f_values = [0] * num_combinations
    for s in range(num_combinations):
        bin_list = _decimal_to_binary_list(s, total_wires)

        # Split bin_list into arguments based on wire_lengths
        args_values = []
        start = 0
        for wire_length in wire_lengths:
            arg_value = _binary_to_decimal(bin_list[start : start + wire_length])
            args_values.append(arg_value)
            start += wire_length

        f_values[s] = f(*args_values) % mod

    # Compute the Möbius transform of f_values
    n = total_wires
    for i in range(n):
        for mask in range(num_combinations):
            if mask & (1 << i):
                f_values[mask] = (f_values[mask] - f_values[mask ^ (1 << i)]) % mod

    coeffs_dict = {}
    for s in range(num_combinations):
        if f_values[s] != 0:
            # Convert s to binary tuple
            bin_tuple = tuple(_decimal_to_binary_list(s, total_wires))
            coeffs_dict[bin_tuple] = f_values[s]

    return coeffs_dict


class OutPoly(Operation):
    r"""Performs the out-place polynomial operation.

    This class implements an out-of-place operation that computes a polynomial function
    over a set of input registers and stores the result in an output register. The result
    is computed modulo a given value.

    Given a function :math:`f(x_1, \dots, x_m)` and a modulus `mod`, the operator performs:

    .. math::

        \text{OutPoly}_{f, \text{mod}} |x_1 \rangle \dots |x_m \rangle |0 \rangle
        = |x_1 \rangle \dots |x_m \rangle |f(x_1, \dots, x_m) \mod \text{mod} \rangle.

    This operation leaves the input registers unchanged and stores the result of the
    polynomial function in the output register. It is based on the implementation detailed
    in `arXiv:2112.10537 <https://arxiv.org/abs/2112.10537>`_.

    .. note::

        To obtain the correct result, the values of the input registers :math:`x_i` must
        be smaller than the modulus `mod`.

    Args:
        f (callable): The polynomial function to be applied to the inputs. It must accept the same number of arguments as there are input registers.
        registers_wires (Sequence[int]): A list with the wires corresponding to the input registers and the output register. The last argument should correspond to the output register.
        mod (int, optional): The modulus to use for the result. If not provided, it defaults to :math:`2^{n}`, where `n` is the number of qubits in the output register.
        work_wires (Sequence[int], optional): The auxiliary wires used for intermediate computation, if necessary. If `mod` is not a power of 2, two auxiliary work wires are required.
        id (str or None, optional): The name of the operation.

    Raises:
        ValueError: If the function `f` does not accept the correct number of input parameters.
        ValueError: If `mod` is not a power of 2 and no or insufficient work wires are provided.
        ValueError: If the wires used in the input and output registers overlap.

    Example:
        Given a polynomial function :math:`f(x, y) = x^2 + y` with a modulus 7,
        we can apply this operation as follows:

        .. code-block:: python

            wires_x = [0, 1, 2]
            wires_y = [3, 4, 5]
            output_wires = [6, 7, 8]
            work_wires = [9,10]

            registers_wires = [wires_x, wires_y, output_wires]


            def f(x, y):
                return x ** 2 + y


            @qml.qnode(qml.device("default.qubit", shots = 1))
            def circuit():
                # loading values for x and y
                qml.BasisEmbedding(3, wires=wires_x)
                qml.BasisEmbedding(2, wires=wires_y)

                # applying the polynomial
                qml.OutPoly(f, registers_wires, mod = 7, work_wires = work_wires)

                return qml.sample(wires=output_wires)

            print(circuit())

        .. code-block:: pycon

            >>> print(circuit())
            [1 0 0]

        The result, :math:`[1 0 0]`, is the binary representation of :math:`3^2 + 2  \; \text{modulo} \; 7 = 4`.

    .. seealso:: :class:`~.PhaseAdder`

    """

    grad_method = None

    def __init__(
        self, f=None, registers_wires=None, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        if registers_wires is None or f is None:
            raise ValueError("The register wires and the function f must be provided.")

        num_work_wires = 0 if not work_wires else len(work_wires)
        if mod is None:
            mod = 2 ** len(registers_wires[-1])
        elif mod != 2 ** len(registers_wires[-1]) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(registers_wires[-1])}, two work wires should be provided"
            )

        self.mod = mod

        if not isinstance(mod, int):
            raise ValueError("mod must be integer.")

        self.hyperparameters["f"] = f

        if len(inspect.signature(f).parameters) != len(registers_wires) - 1:
            raise ValueError(
                f"The function takes {len(inspect.signature(f).parameters)} input parameters but {len(registers_wires) - 1} has provided."
            )

        self.hyperparameters["registers_wires"] = [
            qml.wires.Wires(register) for register in registers_wires
        ]

        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires) if work_wires else None

        all_wires = sum([*self.hyperparameters["registers_wires"]], start=[])

        if work_wires:
            all_wires += work_wires

        if len(all_wires) != sum(len(register) for register in registers_wires) + num_work_wires:
            raise ValueError(
                "None of the wires in a register must be contained in another register."
            )

        super().__init__(wires=all_wires, id=id)

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple(
            (key, value)
            for key, value in self.hyperparameters.items()
            if key
            not in [
                "f",
                "registers_wires",
            ]
        )
        return (
            self.hyperparameters["f"],
            self.hyperparameters["registers_wires"],
        ), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def map_wires(self, wire_map: dict):

        new_registers_wires = [
            [wire_map[wire] for wire in input_wire]
            for input_wire in self.hyperparameters["registers_wires"]
        ]

        new_work_wires = (
            [wire_map[wire] for wire in self.hyperparameters["work_wires"]]
            if self.hyperparameters.get("work_wires")
            else None
        )

        return OutPoly(
            self.hyperparameters["f"],
            new_registers_wires,
            mod=self.hyperparameters["mod"],
            work_wires=new_work_wires,
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(**self.hyperparameters)

    @staticmethod
    def compute_decomposition(**kwargs):  # pylint: disable=arguments-differ

        f = kwargs["f"]
        mod = kwargs["mod"]

        registers_wires = kwargs["registers_wires"]
        work_wires = kwargs["work_wires"]

        list_ops = []

        output_adder_mod = (
            [work_wires[0]] + registers_wires[-1] if work_wires else registers_wires[-1]
        )
        list_ops.append(qml.QFT(wires=output_adder_mod))

        wires_vars = [len(w) for w in registers_wires[:-1]]

        coeffs_list = _get_coefficients_and_controls(f, mod, *wires_vars)

        for item, coeff in coeffs_list.items():

            if not 1 in item:
                # Add the independent term
                list_ops.append(qml.PhaseAdder(int(coeff), output_adder_mod))
                continue

            all_wires_input = sum([*registers_wires[:-1]], start=[])

            controls = [all_wires_input[i] for i, bit in enumerate(item) if bit == 1]

            if work_wires:
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
            else:
                list_ops.append(
                    qml.ctrl(
                        qml.PhaseAdder(int(coeff) % mod, output_adder_mod),
                        control=controls,
                    )
                )

        list_ops.append(qml.adjoint(qml.QFT)(wires=output_adder_mod))

        return list_ops
