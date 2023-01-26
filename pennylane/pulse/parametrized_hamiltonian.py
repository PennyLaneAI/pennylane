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
    r"""Callable object holding the information representing a parametrized Hamiltonian.

    Passing parameters to the ``ParametrizedHamiltonian`` returns an
    :class:`~pennylane.operation.Operator` representing the Hamiltonian for that set of parameters.

    The Hamiltonian can be represented as a linear combination of other operators, e.g.,
    :math:`H(v, t) = H_\text{drift} + \sum_j f_j(v, t) H_j`, where the :math:`v` are trainable
    parameters, and t is time.

    For example, a time-dependent ``ParametrizedHamiltonian`` with a single trainable parameter can
    be: :math:`a`, could be :math:`H = 2 X_1 X_2 + \sin(a t) Y_1 Y_2`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    A ``ParametrizedHamiltonian`` is a callable with the fixed signature ``H(params, t)``,
    with ``params`` being an iterable where each element corresponds to the parameters of each
    scalar-valued function of the hamiltonian. Calling the ``ParametrizedHamiltonian`` returns an
    ``Operator`` representing an instance of the Hamiltonian with the specified parameter values.

    .. note::

        The parameters used in the ``ParametrizedHamiltonian`` call should have the same order
        as the functions used to define this hamiltonian. For example, if we build a
        ``ParametrizedHamiltonian`` that contains two functions:

        >>> import jax.numpy as jnp
        >>> def f1(p, t):
        ...     return p * t
        >>> def f2(p, t):
        ...     return p * jnp.sin(t)
        >>> coeffs = [f1, f2]
        >>> ops = [qml.PauliX(0), qml.PauliY(1)]
        >>> H = qml.dot(coeffs, ops)

        And we call it using the following parameters:

        >>> params = [4, 5]
        >>> H(params, t=0.5)
        (2.0*(PauliX(wires=[0]))) + (2.397127628326416*(PauliY(wires=[1])))

        Internally we are computing ``f1(4, 0.5)`` and ``f2(5, 0.5)``.

    **Example:**

    .. code-block:: python3

        coeffs = [2, lambda v, t: v[0] * jnp.sin(v[1] * t)]
        observables =  [qml.PauliX(0), qml.PauliY(1)]
        H = ParametrizedHamiltonian(coeffs, observables)

    >>> H([jnp.ones(2)], 1.)
    (2*(PauliX(wires=[0]))) + (0.8414710164070129*(PauliY(wires=[1])))

    A ``ParametrizedHamiltonian`` can be created by passing a list of coefficients (scalars or functions), as well as
    a list of corresponding observables. The functions must have two arguments, the first one being the
    trainable parameters and the second one being time.

    >>> def f1(p, t): return np.sin(p[0] * t) + p[1]
    >>> def f2(p, t): return p * np.cos(t)

    The functions, along with scalar coefficients, can then be used to initialize a ``ParametrizedHamiltonian``:

    .. code-block:: python3

        coeffs = [2, f1, f2]
        obs = [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        H = ParametrizedHamiltonian(coeffs, obs)

    The resulting object can be passed parameters, and will return an ``Operator`` representing the
    ``ParametrizedHamiltonian`` with the specified parameters:

    >>> H([[1.2, 2.3], 4.5], 0.5)
    (2*(PauliX(wires=[0]) @ PauliX(wires=[1]))) + ((2.864642473395035*(PauliY(wires=[0]) @ PauliY(wires=[1]))) + (3.9491215285066774*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))))

    Here [1.2, 2.3] is passed to f1, and 4.5 is passed to f2, while both receive t=0.5.

    We can also access the fixed and parametrized terms of the ``ParametrizedHamiltonian``.
    The fixed term is an ``Operator``, while the parametrized term must be initialized with concrete
    parameters to obtain an ``Operator``:

    >>> H.H_fixed()
    2*(PauliX(wires=[0]) @ PauliX(wires=[1]))
    >>> H.H_parametrized([[1.2, 2.3], 4.5], 0.5)
    (2.864642473395035*(PauliY(wires=[0]) @ PauliY(wires=[1]))) + (3.9491215285066774*(PauliZ(wires=[0]) @ PauliZ(wires=[1])))
    """

    def __init__(self, coeffs, observables):

        if len(coeffs) != len(observables):
            raise ValueError(
                "Could not create valid Hamiltonian; "
                "number of coefficients and operators does not match."
                f"Got len(coeffs) = {len(coeffs)} and len(observables) = {len(observables)}."
            )

        self.coeffs_fixed = []
        self.coeffs_parametrized = []
        self.ops_fixed = []
        self.ops_parametrized = []

        for coeff, obs in zip(coeffs, observables):
            if callable(coeff):
                self.coeffs_parametrized.append(coeff)
                self.ops_parametrized.append(obs)
            else:
                self.coeffs_fixed.append(coeff)
                self.ops_fixed.append(obs)

        self.wires = Wires.all_wires(
            [op.wires for op in self.ops_fixed] + [op.wires for op in self.ops_parametrized]
        )

    def __call__(self, params, t):
        if len(params) != len(self.coeffs_parametrized):
            raise ValueError(
                "The length of the params argument and the number of scalar-valued functions must be the same."
                f"Received len(params) = {len(params)} but expected {len(self.coeffs_parametrized)}"
            )
        return self.H_fixed() + self.H_parametrized(params, t)

    def __repr__(self):
        return f"ParametrizedHamiltonian: terms={qml.math.shape(self.coeffs)[0]}"

    def H_fixed(self):
        """The fixed term(s) of the ``ParametrizedHamiltonian``. Returns a ``Sum`` operator of ``SProd``
        operators (or a single ``SProd`` operator in the event that there is only one term in ``H_fixed``)."""
        return qml.dot(self.coeffs_fixed, self.ops_fixed) if self.coeffs_fixed else 0

    def H_parametrized(self, params, t):
        """The parametrized terms of the Hamiltonian for the specified parameters and time.

        Args:
            params(tensor_like): the parameters values used to evaluate the operators
            t(float): the time at which the operator is evaluated

        Returns: an operator that is a ``Sum`` of ``~S_Prod`` operators (or a single
        ``~SProd`` operator in the event that there is only one term in ``H_parametrized``)."""

        coeffs = [f(param, t) for f, param in zip(self.coeffs_parametrized, params)]
        return qml.dot(coeffs, self.ops_parametrized) if coeffs else 0

    @property
    def coeffs(self):
        """Return the coefficients defining the ``ParametrizedHamiltonian``, including the unevaluated
        functions for the parametrized terms.

        Returns:
            Iterable[float]): coefficients in the Hamiltonian expression
        """
        return self.coeffs_fixed + self.coeffs_parametrized

    @property
    def ops(self):
        """Return the operators defining the ``ParametrizedHamiltonian``.

        Returns:
            Iterable[Observable]): observables in the Hamiltonian expression
        """
        return self.ops_fixed + self.ops_parametrized

    def __add__(self, H):
        r"""The addition operation between a ``ParametrizedHamiltonian`` and an ``Operator``
        or ``ParametrizedHamiltonian``."""
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

        return NotImplemented

    __radd__ = __add__
