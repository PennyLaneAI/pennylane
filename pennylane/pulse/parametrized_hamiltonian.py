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
from copy import copy

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops import Sum
from pennylane.typing import TensorLike
from pennylane.wires import Wires


# pylint: disable= too-many-instance-attributes
class ParametrizedHamiltonian:
    r"""Callable object holding the information representing a parametrized Hamiltonian.

    The Hamiltonian can be represented as a linear combination of other operators, e.g.,

    .. math::
        H(\{v_j\}, t) = H_\text{drift} + \sum_j f_j(v_j, t) H_j

    where the :math:`\{v_j\}` are trainable parameters for each scalar-valued parametrization :math:`f_j`, and t is time.

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Operator]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    A ``ParametrizedHamiltonian`` is a callable with the fixed signature ``H(params, t)``,
    with ``params`` being an iterable where each element corresponds to the parameters of each
    scalar-valued function of the Hamiltonian.

    Calling the ``ParametrizedHamiltonian`` returns an :class:`~.Operator` representing an instance of the
    Hamiltonian with the specified parameter values.

    .. seealso::

        :func:`~.pennylane.evolve`, :class:`~.ParametrizedEvolution`

    .. note::
        The ``ParametrizedHamiltonian`` must be Hermitian at all times. This is not explicitly
        checked; ensuring a correctly defined Hamiltonian is the responsibility of the user.

    **Example**

    A ``ParametrizedHamiltonian`` can be created using :func:`~.pennylane.dot`, by passing a list of coefficients,
    as well as a list of corresponding observables. Each coefficient function must take two arguments, the first one
    being the trainable parameters and the second one being time, though it need not use them both.

    .. code-block:: python3

        f1 = lambda p, t: p[0] * jnp.sin(p[1] * t)
        f2 = lambda p, t: p * t
        coeffs = [2., f1, f2]
        observables =  [qml.X(0), qml.Y(0), qml.Z(0)]
        H = qml.dot(coeffs, observables)

    The resulting object can be passed parameters, and will return an :class:`~.Operator` representing the
    ``ParametrizedHamiltonian`` with the specified parameters. Note that parameters must be passed in the order
    the functions were passed in creating the ``ParametrizedHamiltonian``:

    .. code-block:: python3

        p1 = jnp.array([1., 1.])
        p2 = 1.
        params = [p1, p2]  # p1 is passed to f1, and p2 to f2

    >>> H(params, t=1.)
    (
        2.0 * X(0)
      + 0.8414709848078965 * Y(0)
      + 1.0 * Z(0)
    )

    .. note::
        To be able to compute the time evolution of the Hamiltonian with :func:`~.pennylane.evolve`,
        these coefficient functions should be defined using ``jax.numpy`` rather than ``numpy``.

    We can also access the fixed and parametrized terms of the ``ParametrizedHamiltonian``.
    The fixed term is an :class:`~.Operator`, while the parametrized term must be initialized with concrete
    parameters to obtain an :class:`~.Operator`:

    >>> H.H_fixed()
    2.0 * X(0)
    >>> H.H_parametrized([[1.2, 2.3], 4.5], 0.5)
    1.095316728312625 * Y(0) + 2.25 * Z(0)


    .. details::
        :title: Usage Details

        An alternative method for creating a ``ParametrizedHamiltonian`` is to multiply operators and callable
        coefficients:

        .. code-block:: python3

            def f1(p, t):
                return jnp.sin(p[0] * t**2) + p[1]

            def f2(p, t):
                return p * jnp.cos(t)

            H = 2 * qml.X(0) + f1 * qml.Y(0) + f2 * qml.Z(0)

        .. note::
            Whichever method is used for initializing a :class:`~.ParametrizedHamiltonian`, the terms defined with fixed
            coefficients should come before parametrized terms to prevent discrepancy in the wire order.

        .. note::
            The parameters used in the ``ParametrizedHamiltonian`` call should have the same order
            as the functions used to define this Hamiltonian. For example, we could call the above
            Hamiltonian using the following parameters:

            >>> params = [[4.6, 2.3], 1.2]
            >>> H(params, t=0.5)
            (
                2 * X(0)
              + 3.212763940260521 * Y(0)
              + 1.0530990742684472 * Z(0)
            )

            Internally we are computing ``f1([4.6, 2.3], 0.5)`` and ``f2(1.2, 0.5)``.

        Parametrized coefficients can be any callable that takes ``(p, t)`` and returns a scalar. It is not a
        requirement that both ``p`` and ``t`` be used in the callable: for example, the convenince function
        :func:`~pulse.constant` takes ``(p, t)`` and returns ``p``.

        .. warning::
            When initializing a :class:`~.ParametrizedHamiltonian` via a list of parametrized coefficients, it
            is possible to create a list of multiple coefficients of the same form iteratively using lambda
            functions, i.e.

            ``coeffs = [lambda p, t: p for _ in range(3)]``.

            Do **not**, however, define the function as dependent on the value that is iterated over. That is, it is not
            possible to define

            ``coeffs = [lambda p, t: p * t**i for i in range(3)]``

            to create a list

            ``coeffs = [(lambda p, t: p), (lambda p, t: p * t), (lambda p, t: p * t**2)]``.

            The value of ``i`` when
            creating the lambda functions is set to be the final value in the iteration, such that this will
            produce three identical functions

            ``coeffs = [(lambda p, t: p * t**2)] * 3``.

        We can visualize the behaviour in time of the parametrized coefficients for a given set of parameters. Here
        we look at the Hamiltonian created above:

        .. code-block:: python

            import matplotlib.pyplot as plt

            times = jnp.linspace(0., 5., 1000)
            fs = tuple(c for c in H.coeffs if callable(c))
            params = [[4.6, 2.3], 1.2]

            fig, axs = plt.subplots(nrows=len(fs))

            for n, f in enumerate(fs):
                ax = axs[n]
                ax.plot(times, f(params[n], times), label=f"p={params[n]}")
                ax.set_ylabel(f"f{n}")
                ax.legend(loc="upper left")

            ax.set_xlabel("Time")
            axs[0].set_title(f"H parametrized coefficients")
            plt.tight_layout()
            plt.show()

        .. figure:: ../../_static/pulse/parametrized_coefficients_example.png
                    :align: center
                    :width: 60%
                    :target: javascript:void(0);


        It is possible to add two instance of ``ParametrizedHamiltonian`` together. The resulting
        ``ParametrizedHamiltonian`` takes a list of parameters that is a concatenation of the initial
        two Hamiltonian parameters:

        .. code-block:: python3

            coeffs = [lambda p, t: jnp.sin(p*t) for _ in range(2)]
            ops = [qml.X(0), qml.Y(1)]
            H1 = qml.dot(coeffs, ops)

            def f1(p, t): return t + p
            def f2(p, t): return p[0] * jnp.sin(p[1] * t**2)
            H2 = f1 * qml.Y(0) + f2 * qml.X(1)

            params1 = [2., 3.]
            params2 = [4., [5., 6.]]

        >>> H3 = H2 + H1
        >>> H3([4., [5., 6.], 2., 3.], t=1)
        (
            5.0 * Y(0)
          + -1.3970774909946293 * X(1)
          + 0.9092974268256817 * X(0)
          + 0.1411200080598672 * Y(1)
        )

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
                "The length of the params argument and the number of scalar-valued functions "
                f"must be the same. Received len(params) = {len(params)} parameters but "
                f"expected {len(self.coeffs_parametrized)} parameters."
            )
        H_fixed = self.H_fixed()
        H_param = self.H_parametrized(params, t)
        if H_param == 0:
            return H_fixed
        if H_fixed == 0:
            return H_param
        return qml.sum(self.H_fixed(), self.H_parametrized(params, t))

    def __repr__(self):
        terms = []

        for coeff, op in zip(self.coeffs_fixed, self.ops_fixed):
            term = f"{coeff} * {op}"
            terms.append(term)

        for i, (coeff, op) in enumerate(zip(self.coeffs_parametrized, self.ops_parametrized)):
            op_repr = f"({op})" if isinstance(op, Sum) else str(op)
            named_coeff = coeff if callable(coeff) and hasattr(coeff, "__name__") else type(coeff)
            term = f"{named_coeff.__name__}(params_{i}, t) * {op_repr}"
            terms.append(term)

        res = "\n  + ".join(terms)
        return f"(\n    {res}\n)"

    def map_wires(self, wire_map):
        """Returns a copy of the current ParametrizedHamiltonian with its wires changed according
        to the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .ParametrizedHamiltonian: A new instance with mapped wires
        """
        new_ph = copy(self)
        new_ph.ops_parametrized = [op.map_wires(wire_map) for op in self.ops_parametrized]
        new_ph.ops_fixed = [op.map_wires(wire_map) for op in self.ops_fixed]
        new_ph.wires = Wires.all_wires(
            [op.wires for op in new_ph.ops_fixed] + [op.wires for op in new_ph.ops_parametrized]
        )
        return new_ph

    def H_fixed(self):
        """The fixed term(s) of the ``ParametrizedHamiltonian``. Returns a ``Sum`` operator of ``SProd``
        operators (or a single ``SProd`` operator in the event that there is only one term in ``H_fixed``).
        """
        if self.coeffs_fixed:
            return sum(qml.s_prod(c, o) for c, o in zip(self.coeffs_fixed, self.ops_fixed))
        return 0

    def H_parametrized(self, params, t):
        """The parametrized terms of the Hamiltonian for the specified parameters and time.

        Args:
            params(tensor_like): the parameters values used to evaluate the operators
            t(float): the time at which the operator is evaluated

        Returns:
            Operator: a ``Sum`` of ``SProd`` operators (or a single
            ``SProd`` operator in the event that there is only one term in ``H_parametrized``).
        """
        coeffs = [f(param, t) for f, param in zip(self.coeffs_parametrized, params)]
        return sum(qml.s_prod(c, o) for c, o in zip(coeffs, self.ops_parametrized)) if coeffs else 0

    @property
    def coeffs(self):
        """Return the coefficients defining the ``ParametrizedHamiltonian``, including the unevaluated
        functions for the parametrized terms.

        Returns:
            Iterable[float, Callable]): coefficients in the Hamiltonian expression
        """
        return self.coeffs_fixed + self.coeffs_parametrized

    @property
    def ops(self):
        """Return the operators defining the ``ParametrizedHamiltonian``.

        Returns:
            Iterable[Operator]: observables in the Hamiltonian expression
        """
        return self.ops_fixed + self.ops_parametrized

    def __add__(self, H):
        r"""The addition operation between a ``ParametrizedHamiltonian`` and an ``Operator``
        or ``ParametrizedHamiltonian``."""
        ops = self.ops.copy()
        coeffs = self.coeffs.copy()

        if isinstance(H, (qml.ops.Hamiltonian, qml.ops.LinearCombination, ParametrizedHamiltonian)):
            # if Hamiltonian, coeffs array must be converted to list
            new_coeffs = coeffs + list(H.coeffs.copy())
            new_ops = ops + H.ops.copy()
            return ParametrizedHamiltonian(new_coeffs, new_ops)

        if isinstance(H, qml.ops.SProd):  # pylint: disable=no-member
            new_coeffs = coeffs + [H.scalar]
            new_ops = ops + [H.base]
            return ParametrizedHamiltonian(new_coeffs, new_ops)

        if isinstance(H, Operator):
            new_coeffs = coeffs + [1]
            new_ops = ops + [H]

            return ParametrizedHamiltonian(new_coeffs, new_ops)

        return NotImplemented

    def __radd__(self, H):
        r"""The addition operation between a ``ParametrizedHamiltonian`` and an ``Operator``
        or ``ParametrizedHamiltonian``."""
        ops = self.ops.copy()
        coeffs = self.coeffs.copy()

        if isinstance(H, (qml.ops.Hamiltonian, qml.ops.LinearCombination, ParametrizedHamiltonian)):
            # if Hamiltonian, coeffs array must be converted to list
            new_coeffs = list(H.coeffs.copy()) + coeffs
            new_ops = H.ops.copy() + ops
            return ParametrizedHamiltonian(new_coeffs, new_ops)

        if isinstance(H, qml.ops.SProd):  # pylint: disable=no-member
            new_coeffs = [H.scalar] + coeffs
            new_ops = [H.base] + ops
            return ParametrizedHamiltonian(new_coeffs, new_ops)

        if isinstance(H, Operator):
            new_coeffs = [1] + coeffs
            new_ops = [H] + ops

            return ParametrizedHamiltonian(new_coeffs, new_ops)

        return NotImplemented

    def __mul__(self, other):
        ops = self.ops.copy()
        coeffs_fixed = self.coeffs_fixed.copy()
        coeffs_parametrized = self.coeffs_parametrized.copy()
        if isinstance(other, TensorLike) and qml.math.ndim(other) == 0:
            coeffs_fixed = [other * c for c in coeffs_fixed]
            coeffs_parametrized = [
                lambda p, t, new_c=c: other * new_c(p, t) for c in coeffs_parametrized
            ]
            return ParametrizedHamiltonian(coeffs_fixed + coeffs_parametrized, ops)
        return NotImplemented

    __rmul__ = __mul__
