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
This file contains the ``ParametrizedEvolution`` operator.
"""

from typing import List, Union

import pennylane as qml
from pennylane.operation import AnyWires, Operation

from .parametrized_hamiltonian import ParametrizedHamiltonian

has_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint

    from .parametrized_hamiltonian_pytree import ParametrizedHamiltonianPytree
except ImportError as e:
    has_jax = False


class ParametrizedEvolution(Operation):
    r"""
    ParametrizedEvolution(H, params=None, t=None, do_queue=True, id=None, **odeint_kwargs)

    Parametrized evolution gate, created by passing a :class:`~.ParametrizedHamiltonian` to the
    :func:`~.pennylane.evolve` function

    For a time-dependent Hamiltonian of the form

    .. math:: H(\{v_j\}, t) = H_\text{drift} + \sum_j f_j(v_j, t) H_j

    it implements the corresponding time-evolution operator :math:`U(t_1, t_2)`, which is the
    solution to the time-dependent Schrodinger equation.

    .. math:: \frac{d}{dt}U(t) = -i H(\{v_j\}, t) U(t).

    Under the hood, it is using a numerical ordinary differential equation (ODE) solver. It requires ``jax``,
    and will not work with other machine learning frameworks typically encountered in PennyLane.

    Args:
        H (ParametrizedHamiltonian): Hamiltonian to evolve
        params (Optional[list]): trainable parameters, passed as list where each element corresponds to
            the parameters of a scalar-valued function of the Hamiltonian being evolved.
        t (Union[float, List[float]]): If a float, it corresponds to the duration of the evolution.
            If a list of floats, the ODE solver will use all the provided time values, and
            perform intermediate steps if necessary. It is recommended to just provide a start and end time.
            Note that such absolute times only have meaning within an instance of
            ``ParametrizedEvolution`` and will not affect other gates.
        do_queue (bool): determines if the scalar product operator will be queued. Default is True.
        id (str or None): id for the scalar product operator. Default is None.

    Keyword Args:
        atol (float, optional): Absolute error tolerance for the ODE solver. Defaults to ``1.4e-8``.
        rtol (float, optional): Relative error tolerance for the ODE solver. The error is estimated
            from comparing a 4th and 5th order Runge-Kutta step in the Dopri5 algorithm. This error
            is guaranteed to stay below ``tol = atol + rtol * abs(y)`` through adaptive step size
            selection. Defaults to 1.4e-8.
        mxstep (int, optional): maximum number of steps to take for each timepoint for the ODE solver. Defaults to
            ``jnp.inf``.
        hmax (float, optional): maximum step size allowed for the ODE solver. Defaults to ``jnp.inf``.

    .. warning::
        The :class:`~.ParametrizedHamiltonian` must be Hermitian at all times. This is not explicitly checked
        when creating a :class:`~.ParametrizedEvolution` from the :class:`~.ParametrizedHamiltonian`.

    **Example**

    To create a :class:`~.ParametrizedEvolution`, we first define a :class:`~.ParametrizedHamiltonian`
    describing the system, and then pass it to :func:`~pennylane.evolve`:

    .. code-block:: python

        from jax import numpy as jnp

        f1 = lambda p, t: jnp.sin(p * t)
        H = f1 * qml.PauliY(0)

        ev = qml.evolve(H)

    The initial :class:`~.ParametrizedEvolution` does not have set parameters, and so will not
    have a matrix defined. To obtain an Operator with a matrix, it must be passed parameters and
    a time interval:

    >>> qml.matrix(ev([1.2], t=[0, 4]))
    Array([[ 0.72454906+0.j, -0.6892243 +0.j],
           [ 0.6892243 +0.j,  0.72454906+0.j]], dtype=complex64)

    The parameters can be updated by calling the :class:`~.ParametrizedEvolution` again with different inputs.

    When calling the :class:`~.ParametrizedEvolution`, keyword arguments can be passed to specify
    behaviour of the ODE solver.

    The :class:`~.ParametrizedEvolution` can be implemented in a QNode:

    .. code-block:: python

        import jax

        dev = qml.device("default.qubit.jax", wires=1)
        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [1.2]
    >>> circuit(params)
    Array(0.96632576, dtype=float32)

    >>> jax.grad(circuit)(params)
    [Array(2.3569832, dtype=float32)]

    .. note::
        In the example above, the decorator ``@jax.jit`` is used to compile this execution just-in-time. This means
        the first execution will typically take a little longer with the benefit that all following executions
        will be significantly faster, see the jax docs on jitting. JIT-compiling is optional, and one can remove
        the decorator when only single executions are of interest.

    .. warning::

        The time argument ``t`` corresponds to the time window used to compute the scalar-valued
        functions present in the :class:`ParametrizedHamiltonian` class. Consequently, executing
        two ``ParametrizedEvolution`` operators using the same time window does not mean both
        operators are executed simultaneously, but rather that both evaluate their respective
        scalar-valued functions using the same time window. See Usage Details.


    .. details::
        :title: Usage Details

        The parameters used when calling the ``ParametrizedEvolution`` are expected to have the same order
        as the functions used to define the :class:`~.ParametrizedHamiltonian`. For example:

        .. code-block:: python3

            def f1(p, t):
                return jnp.sin(p[0] * t**2) + p[1]

            def f2(p, t):
                return p * jnp.cos(t)

            H = 2 * qml.PauliX(0) + f1 * qml.PauliY(0) + f2 * qml.PauliZ(0)
            ev = qml.evolve(H)

        >>> params = [[4.6, 2.3], 1.2]
        >>> qml.matrix(ev(params, t=0.5))
        Array([[-0.18354285-0.26303384j, -0.7271658 -0.606923j  ],
               [ 0.7271658 -0.606923j  , -0.18354285+0.26303384j]],      dtype=complex64)

        Internally the solver is using ``f1([4.6, 2.3], t)`` and ``f2(1.2, t)`` at each timestep when
        finding the matrix.

        In the case where we have defined two Hamiltonians, ``H1`` and ``H2``, and we want to find a time evolution
        where the two are driven simultaneously for some period of time, it is important that both are included in
        the same call of :func:`~.pennylane.evolve`.
        For non-commuting operations, applying ``qml.evolve(H1)(params, t=[0, 10])`` followed by
        ``qml.evolve(H2)(params, t=[0, 10])`` will **not** apply the two pulses simultaneously, despite the overlapping
        time window. Instead, it will execute ``H1`` in the ``[0, 10]`` time window, and then subsequently execute
        ``H2`` using the same time window to calculate the evolution, but without taking into account how the time
        evolution of ``H1`` affects the evolution of ``H2`` and vice versa.

        Consider two non-commuting :class:`ParametrizedHamiltonian` objects:

        .. code-block:: python

            from jax import numpy as jnp

            ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
            coeffs = [lambda p, t: p for _ in range(3)]
            H1 = qml.dot(coeffs, ops)  # time-independent parametrized Hamiltonian

            ops = [qml.PauliZ(0), qml.PauliY(1), qml.PauliX(2)]
            coeffs = [lambda p, t: p * jnp.sin(t) for _ in range(3)]
            H2 = qml.dot(coeffs, ops) # time-dependent parametrized Hamiltonian

        The evolutions of the :class:`ParametrizedHamiltonian` can be used in a QNode.

        .. code-block:: python

            dev = qml.device("default.qubit.jax", wires=3)

            @qml.qnode(dev, interface="jax")
            def circuit1(params):
                qml.evolve(H1)(params, t=[0, 10])
                qml.evolve(H2)(params, t=[0, 10])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

            @qml.qnode(dev, interface="jax")
            def circuit2(params):
                qml.evolve(H1 + H2)(params, t=[0, 10])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        In ``circuit1``, the two Hamiltonians are evolved over the same time window, but inside different operators.
        In ``circuit2``, we add the two to form a single :class:`~.ParametrizedHamiltonian`. This will combine the
        two so that the expected parameters will be ``params1 + params2`` (as an addition of ``list``).
        They can then be included inside a single :class:`~.ParametrizedEvolution`.

        The resulting evolutions of ``circuit1`` and ``circuit2`` are **not** identical:

        >>> params = jnp.array([1., 2., 3.])
        >>> circuit1(params)
        Array(-0.01543971, dtype=float32)

        >>> params = jnp.concatenate([params, params])  # H1 + H2 requires 6 parameters!
        >>> circuit2(params)
        Array(-0.78236955, dtype=float32)

        Here, ``circuit1`` is not executing the evolution of ``H1`` and ``H2`` simultaneously, but rather
        executing ``H1`` in the ``[0, 10]`` time window and then executing ``H2`` with the same time window,
        without taking into account how the time evolution of ``H1`` affects the evolution of ``H2`` and vice versa!

        One can also provide a list of time values that the ODE solver will use to calculate the evolution of the
        ``ParametrizedHamiltonian``. Keep in mind that the ODE solver uses an adaptive step size, thus
        it might use additional intermediate time values.

        .. code-block:: python

            t = jnp.arange(0., 10.1, 0.1)
            @qml.qnode(dev, interface="jax")
            def circuit(params):
                qml.evolve(H1 + H2)(params, t=t)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        >>> circuit(params)
        Array(-0.78236955, dtype=float32)
        >>> jax.grad(circuit)(params)
        Array([-4.8066125 ,  3.703827  , -1.3297377 , -2.406232  ,  0.6811726 ,
            -0.52277344], dtype=float32)

        Given that we used the same time window (``[0, 10]``), the results are the same as before.
    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        H: ParametrizedHamiltonian,
        params: list = None,
        t: Union[float, List[float]] = None,
        do_queue=True,
        id=None,
        **odeint_kwargs
    ):
        if not has_jax:
            raise ImportError(
                "Module jax is required for the ``ParametrizedEvolution`` class. "
                "You can install jax via: pip install jax"
            )
        if not all(op.has_matrix or isinstance(op, qml.Hamiltonian) for op in H.ops):
            raise ValueError(
                "All operators inside the parametrized hamiltonian must have a matrix defined."
            )
        self._has_matrix = params is not None and t is not None
        self.H = H
        self.odeint_kwargs = odeint_kwargs
        if t is None:
            self.t = None
        else:
            self.t = jnp.array([0, t] if qml.math.ndim(t) == 0 else t, dtype=float)
        params = [] if params is None else params
        super().__init__(*params, wires=H.wires, do_queue=do_queue, id=id)

    def __call__(self, params, t, **odeint_kwargs):
        # Need to cast all elements inside params to `jnp.arrays` to make sure they are not cast
        # to `np.arrays` inside `Operator.__init__`
        params = [jnp.array(p) for p in params]
        odeint_kwargs = {**self.odeint_kwargs, **odeint_kwargs}
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(self)

        return ParametrizedEvolution(
            H=self.H, params=params, t=t, do_queue=True, id=self.id, **odeint_kwargs
        )

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self._has_matrix

    # pylint: disable=import-outside-toplevel
    def matrix(self, wire_order=None):
        if not self.has_matrix:
            raise ValueError(
                "The parameters and the time window are required to compute the matrix. "
                "You can update its values by calling the class: EV(params, t)."
            )
        y0 = jnp.eye(2 ** len(self.wires), dtype=complex)

        with jax.ensure_compile_time_eval():
            H_jax = ParametrizedHamiltonianPytree.from_hamiltonian(
                self.H, dense=len(self.wires) < 3, wire_order=self.wires
            )

        def fun(y, t):
            """dy/dt = -i H(t) y"""
            return (-1j * H_jax(self.data, t=t)) @ y

        result = odeint(fun, y0, self.t, **self.odeint_kwargs)
        mat = result[-1]
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)
