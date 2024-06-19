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

from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian


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
        op (.Operator): operator to evolve
        coeff (float): coefficient multiplying the exponentiated operator

    Returns:
        .Evolution: evolution operator

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
        op (.ParametrizedHamiltonian): Hamiltonian to evolve

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

        dev = qml.device("default.qubit.jax", wires=4)
        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.Z(0))

    >>> params = [1., 2., 3., 4.]
    >>> circuit(params)
    Array(0.8627419, dtype=float32)

    >>> jax.grad(circuit)(params)
    [Array(50.690746, dtype=float32),
    Array(-6.296886e-05, dtype=float32),
    Array(-6.3341584e-05, dtype=float32),
    Array(-7.052516e-05, dtype=float32)]

    .. note::
        In the example above, the decorator ``@jax.jit`` is used to compile this execution just-in-time. This means
        the first execution will typically take a little longer with the benefit that all following executions
        will be significantly faster, see the jax docs on jitting. JIT-compiling is optional, and one can remove
        the decorator when only single executions are of interest.
    """


# pylint: disable=missing-docstring
@evolve.register
def parametrized_evolution(op: ParametrizedHamiltonian, **kwargs):
    return ParametrizedEvolution(H=op, **kwargs)


# pylint: disable=missing-docstring
@evolve.register
def evolution(op: Operator, coeff: float = 1, num_steps: int = None):
    return Evolution(op, coeff, num_steps)
