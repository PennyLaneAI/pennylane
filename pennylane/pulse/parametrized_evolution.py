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

# pylint: disable=too-few-public-methods,function-redefined

"""
This file contains the ``ParametrizedEvolution`` operator and the ``evolve`` constructor.
"""

from typing import List, Union

import pennylane as qml
from pennylane.operation import AnyWires, Operation

from .parametrized_hamiltonian import ParametrizedHamiltonian

has_jax = True
try:
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
except ImportError as e:
    has_jax = False


class ParametrizedEvolution(Operation):
    r"""Parametrized evolution gate.

    For a time-dependent Hamiltonian of the form
    :math:`H(v, t) = H_\text{drift} + \sum_j f_j(v, t) H_j` it implements the corresponding
    time-evolution operator :math:`U(t_1, t_2)`, which is the solution to the time-dependent
    Schrodinger equation.

    .. math:: \frac{d}{dt}U(t) = -i H(v, t) U(t).


    Under the hood, it is using a numerical ordinary differential equation solver.

    .. note::

        The default parameters of the numerical ordinary differential equation solver can be
        overwritten when calling the :class:`ParametrizedEvolution` class:

        >>> qml.evolve(H)(params=[1., 2., 3.], t=[4, 10], mxstep=1, atol=1e-6)

    Args:
        H (ParametrizedHamiltonian): hamiltonian to evolve
        params (ndarray): trainable parameters
        t (Union[float, List[float]]): If a float, it corresponds to the duration of the evolution.
            If a list of floats, the ``odeint`` solver will use all the provided time values, and
            perform intermediate steps if necessary. It is recommended to just provide a start and end time.
            Note that such absolute times only have meaning within an instance of
            ``ParametrizedEvolution`` and will not affect other gates.
        time (str, optional): The name of the time-based parameter in the parametrized Hamiltonian.
            Defaults to "t".
        do_queue (bool): determines if the scalar product operator will be queued. Default is True.
        id (str or None): id for the scalar product operator. Default is None.

    Keyword Args:
        atol (float, optional): Absolute error tolerance. Defaults to 1.4e-8.
        rtol (float, optional): Relative error tolerance. The error is estimated
            from comparing a 4th and 5th order Runge-Kutta step in the Dopri5 algorithm. This error
            is guaranteed to stay below ``tol = atol + rtol * abs(y)`` through adaptive step size
            selection. Defaults to 1.4e-8.
        mxstep (int, optional): maximum number of steps to take for each timepoint. Defaults to
            ``jnp.inf``.
        hmax (float, optional): maximum step size allowed. Defaults to ``jnp.inf``.

    .. warning::

        The time argument ``t`` corresponds to the time window used to compute the scalar-valued
        functions present in the :class:`ParametrizedHamiltonian` class. Consequently, executing
        two ``ParametrizedEvolution`` gates using the same time window does not mean both gates
        are executed simultaneously, but rather both gates evaluate their respective scalar-valued
        functions using the same time window.

    .. note::

        To execute two :class:`ParametrizedHamiltonian` instances simultaneously one must wrap them
        with the same ``ParametrizedEvolution`` gate.

    **Example**

    Let's first build two :class:`ParametrizedHamiltonian` objects:

    >>> ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
    >>> coeffs = [lambda p, t: p for _ in range(3)]
    >>> H1 = qml.dot(coeffs, ops)  # time-independent parametrized Hamiltonian
    >>> ops = [qml.PauliZ(0), qml.PauliY(1), qml.PauliX(2)]
    >>> coeffs = [lambda p, t: p * jnp.sin(t) for _ in range(3)]
    >>> H2 = qml.dot(coeffs, ops) # time-dependent parametrized Hamiltonian
    >>> H1, H2
    (ParametrizedHamiltonian: terms=3, ParametrizedHamiltonian: terms=3)

    Now we can execute the evolution of these parametrized hamiltonians:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> @qml.qnode(dev, interface="jax")
    ... def circuit(params):
    ...     qml.evolve(H1)(params, t=[0, 10])
    ...     qml.evolve(H2)(params, t=[0, 10])
    ...     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
    >>> params = jnp.array([1., 2., 3.])
    >>> circuit(params)
    Array(-0.01543971, dtype=float32)

    We can also compute the gradient of this evolution with respect to the input parameters!

    >>> jax.grad(circuit)(params)
    Array([ 0.6908507,  0.0865578, -1.4594607], dtype=float32)

    As mentioned in the warning above, ``circuit`` is not executing the evolution of ``H1`` and ``H2``
    simultaneously, but rather executing ``H1`` in the ``[0, 10]`` time window and then executing
    ``H2`` with the same time window.

    If we want to execute both hamiltonians simultaneously, we need to wrap them with the same
    ``ParametrizedEvolution`` operator:

    >>> @qml.qnode(dev, interface="jax")
    ... def circuit(params):
    ...     qml.evolve(H1 + H2)(params, t=[0, 10])
    ...     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
    >>> params = jnp.concatenate([params, params])  # H1 + H2 requires 6 parameters!
    >>> circuit(params)
    Array(-0.78236955, dtype=float32)
    >>> jax.grad(circuit)(params)
    Array([-4.8066125,  3.7038102, -1.3294725, -2.4061902,  0.6811545,
        -0.5226515], dtype=float32)

    One can also provide a list of time values that the odeint will use to calculate the parametrized
    hamiltonian's evolution. Keep in mind that our odeint solver uses an adaptive step size, thus
    it might use intermediate time values.

    >>> t = jnp.arange(0., 10.1, 0.1)
    >>> @qml.qnode(dev, interface="jax")
    ... def circuit(params):
    ...     qml.evolve(H1 + H2)(params, t=t)
    ...     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
    >>> circuit(params)
    Array(-0.78236955, dtype=float32)
    >>> jax.grad(circuit)(params)
    Array([-4.8066125 ,  3.703827  , -1.3297377 , -2.406232  ,  0.6811726 ,
        -0.52277344], dtype=float32)

    Given that we used the same time window ([0, 10]), the results are the same as before.
    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires
    # pylint: disable=too-many-arguments, super-init-not-called
    def __init__(
        self,
        H: ParametrizedHamiltonian,
        params: list = None,
        t: Union[float, List[float]] = None,
        time="t",
        do_queue=True,
        id=None,
        **odeint_kwargs
    ):
        if not has_jax:
            raise ImportError(
                "Module jax is required for the ``ParametrizedEvolution`` class. You can install jax via: pip install jax"
            )
        if not all(op.has_matrix or isinstance(op, qml.Hamiltonian) for op in H.ops):
            raise ValueError(
                "All operators inside the parametrized hamiltonian must have a matrix defined."
            )
        self.H = H
        self.time = time
        self.params = params
        self.odeint_kwargs = odeint_kwargs
        if t is None:
            self.t = None
        else:
            self.t = jnp.array([0, t] if qml.math.size(t) == 1 else t, dtype=float)
        super().__init__(wires=H.wires, do_queue=do_queue, id=id)

    def __call__(self, params, t, **odeint_kwargs):
        self.params = params
        self.t = jnp.array([0, t] if qml.math.size(t) == 1 else t, dtype=float)
        if odeint_kwargs:
            self.odeint_kwargs.update(odeint_kwargs)
        return self

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return True

    # pylint: disable=import-outside-toplevel
    def matrix(self, wire_order=None):
        if self.params is None or self.t is None:
            raise ValueError(
                "The parameters and the time window are required to compute the matrix. "
                "You can update its values by calling the class: EV(params, t)."
            )
        y0 = jnp.eye(2 ** len(self.wires), dtype=complex)

        def fun(y, t):
            """dy/dt = -i H(t) y"""
            return -1j * qml.matrix(self.H(self.params, t=t)) @ y

        result = odeint(fun, y0, self.t, **self.odeint_kwargs)
        mat = result[-1]
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)
