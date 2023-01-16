# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule contains the ParametrizedHamiltonian class
"""
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from pennylane.wires import Wires


# pylint: disable= too-many-instance-attributes
class ParametrizedHamiltonian:
    r"""Callable object holding the information representing a parametrized Hamiltonian. Passing parameters to
    the ParametrizedHamiltonian returns an Operator representing the Hamiltonian for that set of parameters.

    The Hamiltonian can be represented as a linear combination of other operators, e.g.,
    :math:`H(v, t) = H_\text{drift} + \sum_j f_j(v, t) H_j`, where the :math:`v` are trainable parameters,
    and t is time.

    For example, a time-dependent ``ParametrizedHamiltonian`` with a single trainable parameter, :math:`a`, could be :math:`H = 2 * X_1 X_2 + sin(a, t) * Y_1 Y_2`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be constants or
            parametrized functions. All functions passed as ``coeffs`` must accept the same parameters as arguments.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same length as coeffs

    A ParametrizedHamiltonian is callable, and passing parameters to the ParametrizedHamiltonian will return an
    Operator representing an instance of the Hamiltonian with the specified parameter values.

    **Example:**

    A ParametrizedHamiltonian can be created by passing a list of coefficients (scalars or functions), as well as
    a list of corresponding observables. The functions must have identical signatures, though they may not all
    use all the parameters.

    >>> def f1(params, t): return np.sin(params[0]*t)
    >>> def f2(params, t): return params[1] * np.cos(t)

    The functions, along with scalar coefficients, can then be used to initialize a ParametrizedHamiltonian,
    which will be split into a fixed and parametrized term. The fixed term is an Operator, while the parametrized
    term must be initialized with concrete parameters to obtain an Operator.
    >>> coeffs = [2, f1, f2]
    >>> obs = [qml.PauliX(0)@qml.PauliX(1), qml.PauliY(0)@qml.PauliY(1), qml.PauliZ(0)@qml.PauliZ(1)]
    >>> H = ParametrizedHamiltonian(coeffs, obs)
    >>> H.H_fixed
      2*(PauliX(wires=[0]) @ PauliX(wires=[1]))
    >>> H.H_parametrized([2.5, 3.6], t)
      (0.5984721441039564*(PauliY(wires=[0]) @ PauliY(wires=[1]))) + (1.9450883011253033*(PauliZ(wires=[0]) @ PauliZ(wires=[1])))

    The resulting object can be passed parameters, and will return an Operator representing the
    ParametrizedHamiltonian with the specified parameters:

    >>> H([1.2, 2.3], 0.5)
    (2*(PauliX(wires=[0]) @ PauliX(wires=[1]))) + ((0.5646424733950354*(PauliY(wires=[0]) @ PauliY(wires=[1]))) + (2.0184398923478573*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))))

    """

    def __init__(self, coeffs, observables):

        if qml.math.shape(coeffs)[0] != len(observables):
            raise ValueError(
                "Could not create valid Hamiltonian; "
                "number of coefficients and operators does not match."
            )

        # ToDo: assuming this is how we want to do things, make this clear in the documentation
        # since organizing into H_fixed and H_parametrized potentially moves around the order of the terms compared
        # to how the terms were entered, sorting seems the least ambiguous organization for the wires
        self.wires = Wires.all_wires([op.wires for op in observables], sort=True)

        self._ops = list(observables)
        self._coeffs = coeffs

        self.H_coeffs_fixed = []
        self.H_coeffs_parametrized = []
        self.H_ops_fixed = []
        self.H_ops_parametrized = []

        for coeff, obs in zip(coeffs, observables):
            if callable(coeff):
                self.H_coeffs_parametrized.append(coeff)
                self.H_ops_parametrized.append(obs)
            else:
                self.H_coeffs_fixed.append(coeff)
                self.H_ops_fixed.append(obs)

    def __call__(self, params, t, wires=None):
        H = self.H_fixed() + self.H_parametrized(params, t)

        if wires:
            wire_map = dict(zip(self.wires, wires))
            H = qml.map_wires(H, wire_map)  # ToDo: is this the behaviour we want?

        return H

    def __repr__(self):
        return f"ParametrizedHamiltonian: terms={qml.math.shape(self.coeffs)[0]}"

    def H_fixed(self):
        """The fixed term(s) of the ParametrizedHamiltonian. Returns a qml.Sum operator of qml.SProd operators
        (or a single qml.SProd operator in the event that there is only one term in H_drift)."""
        if self.H_coeffs_fixed:
            return qml.ops.dot(self.H_coeffs_fixed, self.H_ops_fixed)  # pylint: disable=no-member
        return 0

    def H_parametrized(self, params, t):
        """The parametrized terms of the Hamiltonian. Returns a function that can be evaluated
        to get a snapshot of the operator at a set time and with set parameters. When the function is
        evaluated, the returned operator is a qml.Sum of qml.S_Prod operators (or a single qml.SProd
        operator in the event that there is only one term in H_ts)."""

        coeffs = [f(params, t) for f in self.H_coeffs_parametrized]
        if len(coeffs) == 0:
            return 0
        return qml.ops.dot(coeffs, self.H_ops_parametrized)  # pylint: disable=no-member

    @property
    def coeffs(self):
        """Return the coefficients defining the Hamiltonian, including the unevaluated
        functions for the parametrized terms.

        Returns:
            Iterable[float]): coefficients in the Hamiltonian expression
        """
        return self._coeffs

    @property
    def ops(self):
        """Return the operators defining the ParametrizedHamiltonian.

        Returns:
            Iterable[Observable]): observables in the Hamiltonian expression
        """
        return self._ops

    def __add__(self, H):
        r"""The addition operation between a ParametrizedHamiltonian and a
        Hamiltonian or ParametrizedHamiltonian."""
        ops = self.ops.copy()
        coeffs = self.coeffs.copy()

        if isinstance(H, (Hamiltonian, ParametrizedHamiltonian)):
            coeffs.extend(H.coeffs.copy())
            ops.extend(H.ops.copy())
            return ParametrizedHamiltonian(coeffs, ops)

        if isinstance(H, qml.ops.SProd):  # pylint: disable=no-member
            coeffs.append(H.scalar)
            ops.append(H.base)
            return ParametrizedHamiltonian(coeffs, ops)

        if isinstance(H, Operator):
            coeffs.append(1)
            ops.append(H)
            return ParametrizedHamiltonian(coeffs, ops)

        raise ValueError(f"Cannot add ParametrizedHamiltonian and {type(H)}")

    __radd__ = __add__
