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

import sympy

import pennylane as qml
from pennylane.operation import Operation


def _function_to_binary_poly(f, vars):
    r"""Convert a function and a set of variables into a sympy binary polynomial representation.

    Args:
        f (callable): The function that defines the arithmetic operation to be applied. Must be a polynomial
        vars (list of tuples): A list of variables, where each variable is a tuple containing:
            - name (str): The name of the variable.
            - num_wires (int): The number of binary wires (bits) corresponding to the variable.

    Returns:
        (sympy.Poly): The binary polynomial that represents the function
    """

    symbols, terms = [], []
    for name, num_wires in vars:
        simbolic_vars = sympy.symbols(" ".join([f"{name}_{i}" for i in range(num_wires)]))
        symbols += simbolic_vars
        terms.append(sum([simbolic_vars[i] * 2**i for i in range(num_wires)]))

    output = f(*terms)
    if not output.is_polynomial():
        raise ValueError("The function must be polynomial in terms of its parameters")

    poly = sympy.expand(output)
    return _adjust_exponents(poly)


def _adjust_exponents(expr):
    """
    Adjust the exponents of a polynomial expression by filtering out terms with zero exponents.

    Args:
        expr (sympy.Poly): A symbolic polynomial composed of multiple terms.

    Returns:
        sympy.Poly: The adjusted expression where terms with zero exponents are omitted.

    """

    result = 0
    for term in expr.as_ordered_terms():

        coeff, bases = term.as_coeff_Mul()
        bases_dict = bases.as_powers_dict()

        non_zero_exponent_bases = []
        for base, exp in bases_dict.items():
            if exp != 0:
                non_zero_exponent_bases.append(base)

        filtered_term = coeff * qml.math.prod(non_zero_exponent_bases)

        result += filtered_term

    return result


def _extract_numbers(s):
    """Extract two integers `m` and `n` from a string based on a the pattern `m_n`.

    Args:
        s (str): The input string containing the pattern `number_number`.

    Returns:
        tuple: A tuple containing the two integers `m` and `n`.

    """

    pattern = r"(\d+)_(\d+)"
    match = re.search(pattern, s)
    int1, int2 = map(int, match.groups())
    return (int1, int2)


def _polynomial_to_list(poly):
    """Convert a polynomial into a list of terms and their coefficients.

    Args:
        poly (sympy.Poly): A symbolic polynomial expression.

    Returns:
        (list): A list of tuples, where each tuple contains the factors and the coefficients.

    """
    coeff_dict = poly.as_coefficients_dict()
    return [(tuple(term.as_ordered_factors()), coeff_dict[term]) for term in coeff_dict]


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
        f (callable): The polynomial function to be applied to the inputs. It must accept
                      the same number of arguments as there are input registers.
        *args (Sequence[int]): The wires corresponding to the input registers and the
                               output register. The last argument should correspond to
                               the output register.
        mod (int, optional): The modulus to use for the result. If not provided, it defaults
                             to :math:`2^{n}`, where `n` is the number of qubits in the output register.
        work_wires (Sequence[int], optional): The auxiliary wires used for intermediate
                                              computation, if necessary. If `mod` is not a power of 2,
                                              two auxiliary work wires are required.
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

            wires = qml.registers({"x": 3, "y": 3, "z": 3, "output": 3, "aux": 2})

            def f(x, y, z):
                return x**2 + y*x*z**5 - z**3 + 3

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
                    wires["x"],
                    wires["y"],
                    wires["z"],
                    wires["output"],
                    mod=6,
                    work_wires=wires["aux"],
                )
                return qml.sample(wires = wires["output"])

        .. code-block:: pycon

            >>> print(circuit())
            [0 0 1]

    .. seealso:: :class:`~.PhaseAdder`

    """

    grad_method = None

    def __init__(self, f=None, register_wires = None, mod=None, work_wires=None, id=None):

        if register_wires is None or f is None:
            raise ValueError("The arguments and the function f must be provided.")

        if not mod:
            mod = 2 ** len(register_wires[-1])

        num_work_wires = 0 if not work_wires else len(work_wires)
        if mod is None:
            mod = 2 ** len(register_wires[-1])
        elif mod != 2 ** len(register_wires[-1]) and num_work_wires != 2:
            raise ValueError(f"If mod is not 2^{len(register_wires[-1])}, two work wires should be provided")

        self.mod = mod

        if not isinstance(mod, int):
            raise ValueError("mod must be integer.")

        self.hyperparameters["f"] = f

        if len(inspect.signature(f).parameters) != len(register_wires) - 1:
            raise ValueError(
                f"The function takes {len(inspect.signature(f).parameters)} input parameters but {len(register_wires) - 1} has provided."
            )

        self.hyperparameters["register_wires"] = [qml.wires.Wires(register) for register in register_wires]

        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires) if work_wires else None

        all_wires = sum(
            [*self.hyperparameters["register_wires"]], start=[]
        )

        if work_wires:
            all_wires += work_wires

        if len(all_wires) != sum([len(register) for register in register_wires]) + num_work_wires:
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

        output_adder_mod = [work_wires[0]] + register_wires[-1] if work_wires else register_wires[-1]
        list_ops.append(qml.QFT(wires=output_adder_mod))

        vars = []  # variable naming and size
        for ind, wires in enumerate(register_wires[:-1]):
            vars.append((f"x{ind}", len(wires)))

        poly = _function_to_binary_poly(f, vars)
        coeffs_list = _polynomial_to_list(poly)

        for item in coeffs_list:

            # Bias
            if item[0] == (1,):
                list_ops.append(qml.PhaseAdder(int(item[1]), output_adder_mod))
                continue

            controls_aux = []

            for variable in item[0]:
                controls_aux.append(_extract_numbers(str(variable)))

            controls = []
            for aux in controls_aux:
                controls.append(register_wires[:-1][aux[0]][-1 - aux[1]])

            if work_wires:
                list_ops.append(
                    qml.ctrl(
                        qml.PhaseAdder(
                            int(item[1]) % mod, output_adder_mod, work_wire=work_wires[1], mod=mod
                        ),
                        control=controls,
                    )
                )
            else:
                list_ops.append(
                    qml.ctrl(qml.PhaseAdder(int(item[1]), output_adder_mod), control=controls)
                )

        list_ops.append(qml.adjoint(qml.QFT)(wires=output_adder_mod))

        return list_ops
