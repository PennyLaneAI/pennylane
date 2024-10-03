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
import re

import numpy as np

import pennylane as qml
from pennylane.operation import Operation

import itertools


def subsets_with_fixed_zeros(bin_list):
    """
    Generate all subsets from a binary list where zeros remain fixed.

    Args:
        bin_list (list): A list of binary values (0s and 1s).

    Returns:
        list: A list of subsets where zeros remain in their original positions.
    """
    # Find positions of ones (1s)
    ones_positions = [i for i, bit in enumerate(bin_list) if bit == 1]

    # Generate all possible combinations of 0s and 1s for the positions of ones
    ones_combinations = itertools.product([0, 1], repeat=len(ones_positions))

    # Create subsets while keeping zeros fixed
    subsets = []
    for combination in ones_combinations:
        new_list = bin_list[:]  # Create a copy of the original list
        for i, value in zip(ones_positions, combination):
            new_list[i] = value  # Set the values of the combination in the positions of ones

        # Exclude the original list itself
        if new_list != bin_list:
            subsets.append(new_list)

    return subsets


def generate_combinations(n, m):
    """
    Generate all unique permutations of binary lists with n elements and m ones.

    Args:
        n (int): Total number of elements.
        m (int): Number of ones in the list.

    Returns:
        list: A list of unique binary combinations.
    """
    # Create a list with n-m zeros and m ones
    base = [0] * (n - m) + [1] * m

    # Get all unique permutations of the base list
    combinations = set(itertools.permutations(base))

    # Convert to a list of lists for easier handling
    return [list(combination) for combination in combinations]


def binary_to_decimal(binary_list):
    """
    Convert a binary list to its decimal value.

    Args:
        binary_list (list): A list of binary digits (0s and 1s).

    Returns:
        int: The decimal value of the binary list.
    """
    binary_list.reverse()
    return sum(val * (2 ** idx) for idx, val in enumerate(binary_list))


def get_coefficients_and_controls(f, mod, *wire_lengths):
    """
    Calculate the coefficients and controls for a given function and wire configuration.

    Args:
        f (callable): A function that takes a variable number of arguments.
        *wire_lengths (int): The number of binary wires (bits) assigned to each argument of the function.

    Returns:
        dict: A dictionary with the binary combinations as keys and their corresponding function values as values.
    """
    coeffs_dict = {}
    total_wires = sum(wire_lengths)

    for i in range(total_wires):
        combs = generate_combinations(total_wires, i)

        for comb in combs:
            comb_tuple = tuple(comb)

            # Divide the combination into parts for each argument based on wire_lengths
            args_values = []
            start = 0
            for wire_length in wire_lengths:
                arg_value = binary_to_decimal(comb[start:start + wire_length])
                args_values.append(arg_value)
                start += wire_length

            # Call the function f with the calculated argument values
            coeffs_dict[comb_tuple] = f(*args_values)

            # Subtract values from subsets where zeros remain fixed
            subtract_value = 0
            for subset in subsets_with_fixed_zeros(comb):
                subset_tuple = tuple(subset)
                if subset_tuple in coeffs_dict:
                    subtract_value += coeffs_dict[subset_tuple]

            coeffs_dict[comb_tuple] = (coeffs_dict[comb_tuple] - subtract_value) % mod

    return coeffs_dict


class OutPoly(Operation):
    r"""Performs the out-place polynomial operation.

    This class implements an out-of-place operation that computes a polynomial function
    over a set of input registers and stores the result in an output register. The result
    is computed modulo a given value.

    Given a function :math:`f(x_1, \dots, x_m)` and a modulus `mod`, the operator performs:

    .. math::

        \text{OutPoly}(f, \text{mod}) |x_1 \rangle \dots |x_m \rangle |0 \rangle
        = |x_1 \rangle \dots |x_m \rangle |f(x_1, \dots, x_m) \mod \text{mod} \rangle.

    This operation leaves the input registers unchanged and stores the result of the
    polynomial function in the output register. It is based on the implementation detailed
    in `arXiv:2112.10537 <https://arxiv.org/abs/2112.10537>`_.

    .. note::

        To obtain the correct result, the values of the input registers :math:`x_i` must
        be smaller than the modulus `mod`.

    Args:
        f (callable): The polynomial function to be applied to the inputs. It must accept the same number of arguments as there are input registers.
        register_wires (Sequence[int]): The wires corresponding to the input registers and the output register. The last argument should correspond to the output register.
        mod (int, optional): The modulus to use for the result. If not provided, it defaults to :math:`2^{n}`, where `n` is the number of qubits in the output register.
        work_wires (Sequence[int], optional): The auxiliary wires used for intermediate computation, if necessary. If `mod` is not a power of 2, two auxiliary work wires are required.
        id (str or None, optional): The name of the operation.

    Raises:
        ValueError: If the function `f` does not accept the correct number of input parameters.
        ValueError: If `mod` is not a power of 2 and no or insufficient work wires are provided.
        ValueError: If the wires used in the input and output registers overlap.

    Example:
        Given a polynomial function :math:`f(x, y, z) = x^2 + yxz^5 - z^3 + 3` with a modulus 7,
        we can apply this operation as follows:

        .. code-block:: python

            import pennylane as qml

            wires = qml.registers({"x": 3, "y": 3, "z": 3, "output": 3, "work": 2})

            def f(x, y, z):
                return x**2 + y*x*z**5 - z**3 + 3

            x, y, z = 1, 2, 3
            mod = 7

            dev = qml.device("default.qubit", wires=14)

            @qml.qnode(dev)
            def circuit():

                # loading values for x, y and z
                qml.BasisEmbedding(x, wires=wires["x"])
                qml.BasisEmbedding(y, wires=wires["y"])
                qml.BasisEmbedding(z, wires=wires["z"])

                # applying the polynomial
                qml.OutPoly(
                f,
                [wires["x"], wires["y"], wires["z"], wires["output"]],
                mod=6,
                work_wires=wires["work"],
                )
                return qml.sample(wires = wires["output"])

        .. code-block:: pycon

            >>> print(circuit())
            [0 0 1]

    .. seealso:: :class:`~.PhaseAdder`

    """

    grad_method = None

    def __init__(
        self, f=None, register_wires=None, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        if register_wires is None or f is None:
            raise ValueError("The register wires and the function f must be provided.")

        num_work_wires = 0 if not work_wires else len(work_wires)
        if mod is None:
            mod = 2 ** len(register_wires[-1])
        elif mod != 2 ** len(register_wires[-1]) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(register_wires[-1])}, two work wires should be provided"
            )

        self.mod = mod

        if not isinstance(mod, int):
            raise ValueError("mod must be integer.")

        self.hyperparameters["f"] = f

        if len(inspect.signature(f).parameters) != len(register_wires) - 1:
            raise ValueError(
                f"The function takes {len(inspect.signature(f).parameters)} input parameters but {len(register_wires) - 1} has provided."
            )

        self.hyperparameters["register_wires"] = [
            qml.wires.Wires(register) for register in register_wires
        ]

        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires) if work_wires else None

        all_wires = sum([*self.hyperparameters["register_wires"]], start=[])

        if work_wires:
            all_wires += work_wires

        if len(all_wires) != sum(len(register) for register in register_wires) + num_work_wires:
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
                "register_wires",
            ]
        )
        return (
            self.hyperparameters["f"],
            self.hyperparameters["register_wires"],
        ), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(*data, **hyperparams_dict)

    def map_wires(self, wire_map: dict):

        new_register_wires = [
            [wire_map[wire] for wire in input_wire]
            for input_wire in self.hyperparameters["register_wires"]
        ]

        new_work_wires = (
            [wire_map[wire] for wire in self.hyperparameters["work_wires"]]
            if self.hyperparameters.get("work_wires")
            else None
        )

        return OutPoly(
            self.hyperparameters["f"],
            new_register_wires,
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

        register_wires = kwargs["register_wires"]
        work_wires = kwargs["work_wires"]

        list_ops = []

        output_adder_mod = (
            [work_wires[0]] + register_wires[-1] if work_wires else register_wires[-1]
        )
        list_ops.append(qml.QFT(wires=output_adder_mod))

        wires_vars = [len(w) for w in register_wires[:-1]]

        coeffs_list = get_coefficients_and_controls(f, mod, *wires_vars)

        for item in coeffs_list:
            if np.isclose(coeffs_list[item], 0.):
                continue


            # Bias
            if not 1 in item:
                list_ops.append(qml.PhaseAdder(int(coeffs_list[item]), output_adder_mod))
                continue

            all_wires_input = sum([*register_wires[:-1]], start=[])

            controls = [all_wires_input[i] for i, bit in enumerate(item) if bit == 1]

            if work_wires:
                list_ops.append(
                    qml.ctrl(
                        qml.PhaseAdder(
                            int(coeffs_list[item]) % mod, output_adder_mod, work_wire=work_wires[1], mod=mod
                        ),
                        control=controls,
                    )
                )
            else:
                list_ops.append(
                    qml.ctrl(qml.PhaseAdder(int(coeffs_list[item])% mod, output_adder_mod), control=controls)
                )

        list_ops.append(qml.adjoint(qml.QFT)(wires=output_adder_mod))

        return list_ops
