# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the qml.evolve function.
"""
from functools import singledispatch
from typing import overload
from warnings import warn

from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian
from pennylane.typing import TensorLike


@overload
def evolve(op: ParametrizedHamiltonian, **kwargs) -> ParametrizedEvolution: ...
@overload
def evolve(op: Operator, coeff: TensorLike = 1, num_steps: int | None = None) -> Evolution: ...
@singledispatch
def evolve(*args, **kwargs):  # pylint: disable=unused-argument
    r"""This method is dispatched and its functionality depends on the type of the input ``op``.

    .. raw:: html

        <html>
            <h3>Input: Operator</h3>
            <hr>
        </html>

    Returns a new operator that computes the evolution of ``op``.

    .. math::

        e^{-i x \bm{O}}

    Args:
        op (.Operator): operator to evolve. This must be passed as a *positional* argument. Passing it as a *keyword* argument will result in an error.
        coeff (float): coefficient multiplying the exponentiated operator
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. Defaults to `None`. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.

    Returns:
        .Evolution: evolution operator

    .. warning::

        Providing ``num_steps`` to ``qml.evolve`` and ``Evolution`` is deprecated and will be removed in a future version.
        Instead, use :class:`~.TrotterProduct` for approximate methods, providing the ``n`` parameter to perform the
        Suzuki-Trotter product approximation of a Hamiltonian with the specified number of Trotter steps.

        As a concrete example, consider the following case:

        >>> coeffs = [0.5, -0.6]
        >>> ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
        >>> H_flat = qml.dot(coeffs, ops)

        Instead of computing the Suzuki-Trotter product approximation as:

        >>> qml.evolve(H_flat, num_steps=2).decomposition()
        [RX(0.5, wires=[0]),
        PauliRot(-0.6, XY, wires=[0, 1]),
        RX(0.5, wires=[0]),
        PauliRot(-0.6, XY, wires=[0, 1])]

        The same result can be obtained using :class:`~.TrotterProduct` as follows:

        >>> decomp_ops = qml.adjoint(qml.TrotterProduct(H_flat, time=1.0, n=2)).decomposition()
        >>> [simp_op for op in decomp_ops for simp_op in map(qml.simplify, op.decomposition())]
        [RX(0.5, wires=[0]),
        PauliRot(-0.6, XY, wires=[0, 1]),
        RX(0.5, wires=[0]),
        PauliRot(-0.6, XY, wires=[0, 1])]

    **Examples**

    We can use ``qml.evolve`` to compute the evolution of any PennyLane operator:

    >>> op = qml.evolve(qml.X(0), coeff=2)
    >>> op
    Exp(-2j PauliX)

    .. raw:: html

        <html>
            <h3>Input: ParametrizedHamiltonian</h3>
            <hr>
        </html>

    Args:
        op (.ParametrizedHamiltonian): Hamiltonian to evolve. This must be passed as a *positional* argument.

    Returns:
        .ParametrizedEvolution: time evolution :math:`U(t_0, t_1)` of the Hamiltonian


    The function takes a :class:`.ParametrizedHamiltonian` and solves the time-dependent Schrodinger equation

    .. math:: \frac{\partial}{\partial t} |\psi\rangle = -i H(t) |\psi\rangle

    It returns a :class:`~.ParametrizedEvolution`, :math:`U(t_0, t_1)`, which is the solution to the time-dependent
    Schrodinger equation for the :class:`~.ParametrizedHamiltonian`, such that

    .. math:: |\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle

    The :class:`~.ParametrizedEvolution` class uses a numerical ordinary differential equation
    solver (`here <https://github.com/google/jax/blob/main/jax/experimental/ode.py>`_).

    **Examples**

    When evolving a :class:`.ParametrizedHamiltonian`, a :class:`.ParametrizedEvolution`
    instance is returned:

    .. code-block:: python3

        coeffs = [lambda p, t: p * t for _ in range(4)]
        ops = [qml.X(i) for i in range(4)]

        # ParametrizedHamiltonian
        H = qml.dot(coeffs, ops)

        # ParametrizedEvolution
        ev = qml.evolve(H)

    >>> ev
    ParametrizedEvolution(wires=[0, 1, 2, 3])

    The :class:`.ParametrizedEvolution` is an :class:`~.Operator`, but does not have a defined matrix unless it
    is evaluated at set parameters. This is done by calling the :class:`.ParametrizedEvolution`, which has the call
    signature ``(p, t)``:

    >>>  qml.matrix(ev([1., 2., 3., 4.], t=[0, 4]))
    Array([[ 0.04930558+0.j        ,  0.        -0.03259093j,
         0.        +0.1052632j ,  0.06957878+0.j        ,
         0.        -0.01482305j, -0.00979751+0.j        ,
         0.03164552+0.j        ,  0.        -0.0209179j ,
         0.        +0.33526757j,  0.22161038+0.j        ,
         ...
         ...
         ...
         0.        -0.03259093j,  0.04930566+0.j        ]],      dtype=complex64)

    Additional options regarding how the matrix is calculated can be passed to the :class:`.ParametrizedEvolution`
    along with the parameters, as keyword arguments. These options are:

    - ``atol (float, optional)``: Absolute error tolerance
    - ``rtol (float, optional)``: Relative error tolerance
    - ``mxstep (int, optional)``: maximum number of steps to take for each time point
    - ``hmax (float, optional)``: maximum step size

    If not specified, they will default to predetermined values.

    The :class:`~.ParametrizedEvolution` can be implemented in a QNode:

    .. code-block:: python

        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.Z(0))

    >>> params = [1., 2., 3., 4.]
    >>> circuit(params)
    Array(0.86231063, dtype=float64)

    >>> jax.grad(circuit)(params)
    [Array(50.391273, dtype=float64),
    Array(-9.42415807e-05, dtype=float64),
    Array(-0.0001049, dtype=float64),
    Array(-0.00010601, dtype=float64)]

    .. note::
        In the example above, the decorator ``@jax.jit`` is used to compile this execution just-in-time. This means
        the first execution will typically take a little longer with the benefit that all following executions
        will be significantly faster, see the jax docs on jitting. JIT-compiling is optional, and one can remove
        the decorator when only single executions are of interest.
    """
    raise ValueError(
        f"No dispatch rule for first argument of type {type(args[0])}. Options are Operator and ParametrizedHamiltonian"
    )


# pylint: disable=missing-docstring
@evolve.register
def parametrized_evolution(op: ParametrizedHamiltonian, **kwargs):
    return ParametrizedEvolution(H=op, **kwargs)


# pylint: disable=missing-docstring
@evolve.register
def evolution(op: Operator, coeff: float = 1, num_steps: int = None):
    if num_steps is not None:
        warn(
            "Providing ``num_steps`` to ``qml.evolve`` and ``Evolution`` is deprecated and will be removed in a future version. "
            "Instead, you can use ``qml.TrotterProduct`` providing the ``n`` parameter to perform the "
            "Suzuki-Trotter product approximation of a Hamiltonian with the specified number of Trotter steps.",
            PennyLaneDeprecationWarning,
        )
    return Evolution(op, coeff, num_steps)
