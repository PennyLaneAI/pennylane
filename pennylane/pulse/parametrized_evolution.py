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

from typing import List, Union, Sequence
import warnings

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.typing import TensorLike
from pennylane.ops import functions

from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian

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
    ParametrizedEvolution(H, params=None, t=None, return_intermediate=False, complementary=False, id=None, **odeint_kwargs)

    Parametrized evolution gate, created by passing a :class:`~.ParametrizedHamiltonian` to
    the :func:`~.pennylane.evolve` function

    For a time-dependent Hamiltonian of the form

    .. math:: H(\{v_j\}, t) = H_\text{drift} + \sum_j f_j(v_j, t) H_j

    it implements the corresponding time-evolution operator :math:`U(t_0, t_1)`, which is the
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
            perform intermediate steps if necessary. It is recommended to just provide a start
            and end time unless matrices of the time evolution at intermediate times need
            to be computed. Note that such absolute times only have meaning within an instance of
            ``ParametrizedEvolution`` and will not affect other gates.
            To return the matrix at intermediate evolution times, activate ``return_intermediate``
            (see below).
        id (str or None): id for the scalar product operator. Default is None.

    Keyword Args:
        atol (float, optional): Absolute error tolerance for the ODE solver. Defaults to ``1.4e-8``.
        rtol (float, optional): Relative error tolerance for the ODE solver. The error is estimated
            from comparing a 4th and 5th order Runge-Kutta step in the Dopri5 algorithm. This error
            is guaranteed to stay below ``tol = atol + rtol * abs(y)`` through adaptive step size
            selection. Defaults to 1.4e-8.
        mxstep (int, optional): maximum number of steps to take for each timepoint for the
            ODE solver. Defaults to ``jnp.inf``.
        hmax (float, optional): maximum step size allowed for the ODE solver. Defaults to ``jnp.inf``.
        return_intermediate (bool): Whether or not the ``matrix`` method returns all intermediate
            solutions of the time evolution at the times provided in ``t = [t_0,...,t_f]``.
            If ``False`` (the default), only the matrix for the full time evolution is returned.
            If ``True``, all solutions including the initial condition are returned;
            when used in a circuit, this results in ``ParametrizedEvolution`` being a broadcasted
            operation, see the usage details ("Computing intermediate time evolution") below.
        complementary (bool): Whether or not to compute the complementary time evolution when using
            ``return_intermediate=True`` (ignored otherwise).
            If ``False`` (the default), the usual solutions to the Schrodinger equation
            :math:`\{U(t_0, t_0), U(t_0, t_1),\dots, U(t_0, t_f)\}` are computed,
            where :math:`t_i` are the additional times provided in ``t``.
            If ``True``, the *remaining* time evolution to :math:`t_f` is computed instead, returning
            :math:`\{U(t_0, t_f), U(t_1, t_f),\dots, U(t_{f-1}, t_f), U(t_f, t_f)\}`.
        dense (bool): Whether the evolution should use dense matrices. Per default, this is decided by
            the number of wires, i.e. ``dense = len(wires) < 3``.

    .. warning::
        The :class:`~.ParametrizedHamiltonian` must be Hermitian at all times. This is not explicitly checked
        when creating a :class:`~.ParametrizedEvolution` from the :class:`~.ParametrizedHamiltonian`.

    **Example**

    To create a :class:`~.ParametrizedEvolution`, we first define a :class:`~.ParametrizedHamiltonian`
    describing the system, and then pass it to :func:`~pennylane.evolve`:



    .. code-block:: python

        from jax import numpy as jnp

        f1 = lambda p, t: jnp.sin(p * t)
        H = f1 * qml.Y(0)

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
            return qml.expval(qml.Z(0))

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

    .. note::

        Using ``return_intermediate`` in a quantum circuit leads to broadcasted execution,
        which can lead to unintended additional computational cost.
        Also consider the usage details below.

    .. details::
        :title: Usage Details

        The parameters used when calling the ``ParametrizedEvolution`` are expected to have the same order
        as the functions used to define the :class:`~.ParametrizedHamiltonian`. For example:

        .. code-block:: python3

            def f1(p, t):
                return jnp.sin(p[0] * t**2) + p[1]

            def f2(p, t):
                return p * jnp.cos(t)

            H = 2 * qml.X(0) + f1 * qml.Y(0) + f2 * qml.Z(0)
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

            ops = [qml.X(0), qml.Y(1), qml.Z(2)]
            coeffs = [lambda p, t: p for _ in range(3)]
            H1 = qml.dot(coeffs, ops)  # time-independent parametrized Hamiltonian

            ops = [qml.Z(0), qml.Y(1), qml.X(2)]
            coeffs = [lambda p, t: p * jnp.sin(t) for _ in range(3)]
            H2 = qml.dot(coeffs, ops) # time-dependent parametrized Hamiltonian

        The evolutions of the :class:`ParametrizedHamiltonian` can be used in a QNode.

        .. code-block:: python

            dev = qml.device("default.qubit.jax", wires=3)

            @qml.qnode(dev, interface="jax")
            def circuit1(params):
                qml.evolve(H1)(params, t=[0, 10])
                qml.evolve(H2)(params, t=[0, 10])
                return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))

            @qml.qnode(dev, interface="jax")
            def circuit2(params):
                qml.evolve(H1 + H2)(params, t=[0, 10])
                return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))

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
                return qml.expval(qml.Z(0) @ qml.Z(1) @ qml.Z(2))

        >>> circuit(params)
        Array(-0.78236955, dtype=float32)
        >>> jax.grad(circuit)(params)
        Array([-4.8066125 ,  3.703827  , -1.3297377 , -2.406232  ,  0.6811726 ,
            -0.52277344], dtype=float32)

        Given that we used the same time window (``[0, 10]``), the results are the same as before.

        **Computing intermediate time evolution**

        As discussed above, the ODE solver will evaluate the Schrodinger equation at
        intermediate times in any case. By passing additional time values explicitly in the time
        window ``t`` and setting ``return_intermediate=True``, the ``matrix`` method will
        return the matrices for the intermediate time evolutions as well:

        .. math::

            \{U(t_0, t_0), U(t_0, t_1), \dots, U(t_0, t_{f-1}), U(t_0, t_f)\}.

        The first entry here is the initial condition :math:`U(t_0, t_0)=1`. For a simple
        time-dependent single-qubit Hamiltonian, this feature looks like the following:

        .. code-block:: python

            ops = [qml.Z(0), qml.Y(0), qml.X(0)]
            coeffs = [lambda p, t: p * jnp.cos(t) for _ in range(3)]
            H = qml.dot(coeffs, ops) # time-dependent parametrized Hamiltonian

            param = [jnp.array(0.2), jnp.array(1.1), jnp.array(-1.3)]
            time = jnp.linspace(0.1, 0.4, 6) # Six time points from 0.1 to 0.4

            ev = qml.evolve(H)(param, time, return_intermediate=True)

        >>> ev_mats = ev.matrix()
        >>> ev_mats.shape
        (6, 2, 2)

        Note that the broadcasting axis has length ``len(time)`` and is the first axis of the
        returned tensor.
        We may use this feature within QNodes executed on a simulator, returning the
        measurements for all intermediate time steps:

        .. code-block:: python

            dev = qml.device("default.qubit.jax", wires=1)

            @qml.qnode(dev, interface="jax")
            def circuit(param, time):
                qml.evolve(H)(param, time, return_intermediate=True)
                return qml.probs(wires=[0])

        >>> circuit(param, time)
        Array([[1.        , 0.        ],
               [0.9897738 , 0.01022595],
               [0.9599043 , 0.04009585],
               [0.9123617 , 0.08763832],
               [0.84996957, 0.15003097],
               [0.7761489 , 0.22385144]], dtype=float32)


        **Computing complementary time evolution**

        When using ``return_intermediate=True``, the partial time evolutions share the *initial*
        time :math:`t_0`. For some applications, however, it may be useful to compute the
        complementary time evolutions, i.e. the partial evolutions that share the *final* time
        :math:`t_f`. This can be activated by setting ``complementary=True``, which will make
        ``ParametrizedEvolution.matrix`` return the matrices

        .. math::

            \{U(t_0, t_f), U(t_1, t_f), \dots, U(t_f, t_f)\}.

        Using the Hamiltonian from the example above:

        >>> complementary_ev = ev(param, time, return_intermediate=True, complementary=True)
        >>> comp_ev_mats = complementary_ev.matrix()
        >>> comp_ev_mats.shape
        (6, 2, 2)

        If we multiply the matrices computed before with ``complementary=False`` with these
        complementary evolution matrices from the left, we obtain the full time evolution,
        which we can check by comparing to the last entry of ``ev_mats``:

        >>> for mat, c_mat in zip(ev_mats, comp_ev_mats):
        ...     print(qml.math.allclose(c_mat @ mat, ev_mats[-1]))
        True
        True
        True
        True
        True
        True

    """

    _name = "ParametrizedEvolution"
    num_wires = AnyWires
    grad_method = "A"

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        H: ParametrizedHamiltonian,
        params: list = None,
        t: Union[float, List[float]] = None,
        return_intermediate: bool = False,
        complementary: bool = False,
        dense: bool = None,
        id=None,
        **odeint_kwargs,
    ):
        if not all(op.has_matrix or isinstance(op, qml.ops.Hamiltonian) for op in H.ops):
            raise ValueError(
                "All operators inside the parametrized hamiltonian must have a matrix defined."
            )
        self._has_matrix = params is not None and t is not None
        self.H = H
        self.odeint_kwargs = odeint_kwargs
        if t is None:
            self.t = None
        else:
            if isinstance(t, (list, tuple)):
                t = qml.math.stack(t)
            self.t = qml.math.cast(qml.math.stack([0.0, t]) if qml.math.ndim(t) == 0 else t, float)
        if complementary and not return_intermediate:
            warnings.warn(
                "The keyword argument complementary does not have any effect if "
                "return_intermediate is set to False."
            )
        if params is None:
            params = []
        else:
            if not isinstance(H, HardwareHamiltonian) and len(params) != len(H.coeffs_parametrized):
                raise ValueError(
                    "The length of the params argument and the number of scalar-valued functions "
                    f"in the Hamiltonian must be the same. Received {len(params)=} parameters but "
                    f"expected {len(H.coeffs_parametrized)} parameters."
                )
        super().__init__(*params, wires=H.wires, id=id)
        self.hyperparameters["return_intermediate"] = return_intermediate
        self.hyperparameters["complementary"] = complementary
        self._check_time_batching()
        self.dense = len(self.wires) < 3 if dense is None else dense

    def __call__(
        self, params, t, return_intermediate=None, complementary=None, dense=None, **odeint_kwargs
    ):
        if not has_jax:
            raise ImportError(
                "Module jax is required for the ``ParametrizedEvolution`` class. "
                "You can install jax via: pip install jax"
            )
        # Need to cast all elements inside params to `jnp.arrays` to make sure they are not cast
        # to `np.arrays` inside `Operator.__init__`
        params = [jnp.array(p) for p in params]
        # Inherit return_intermediate and complementary from self if not provided.
        if return_intermediate is None:
            return_intermediate = self.hyperparameters["return_intermediate"]
        if complementary is None:
            complementary = self.hyperparameters["complementary"]
        if dense is None:
            dense = self.dense
        odeint_kwargs = {**self.odeint_kwargs, **odeint_kwargs}
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(self)

        return ParametrizedEvolution(
            H=self.H,
            params=params,
            t=t,
            return_intermediate=return_intermediate,
            complementary=complementary,
            dense=dense,
            id=self.id,
            **odeint_kwargs,
        )

    def _check_time_batching(self):
        """Check whether the time argument is broadcasted/batched."""
        if not self.hyperparameters["return_intermediate"] or self.t is None:
            return
        # Subtract 1 because the identity is never returned by `matrix`. If `complementary=True`,
        # subtract an additional 1 because the full time evolution is not being returned.
        self._batch_size = self.t.shape[0]

    def map_wires(self, wire_map):
        mapped_op = super().map_wires(wire_map)
        mapped_op.H = self.H.map_wires(wire_map)
        return mapped_op

    @property
    def hash(self):
        """int: Integer hash that uniquely represents the operator."""
        return hash(
            (
                str(self.name),
                tuple(self.wires.tolist()),
                str(self.hyperparameters.values()),
                str(self.t),
                str(self.data),
                self.H,
                str(self.odeint_kwargs.values()),
            )
        )

    def _flatten(self):
        data = self.data
        odeint_kwargs_tuples = tuple((key, value) for key, value in self.odeint_kwargs.items())
        t = self.t if self.t is None else tuple(self.t)
        metadata = (
            t,
            self.H,
            self.hyperparameters["return_intermediate"],
            self.hyperparameters["complementary"],
            self.dense,
            odeint_kwargs_tuples,
        )

        return data, metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        t, H, return_intermediate, complementary, dense, odeint_kwargs = metadata

        return cls(
            H,
            None if len(data) == 0 else data,
            t,
            return_intermediate=return_intermediate,
            complementary=complementary,
            dense=dense,
            **dict(odeint_kwargs),
        )

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self._has_matrix

    # pylint: disable=import-outside-toplevel
    def matrix(self, wire_order=None):
        if not has_jax:
            raise ImportError(
                "Module jax is required for the ``ParametrizedEvolution`` class. "
                "You can install jax via: pip install jax"
            )
        if not self.has_matrix:
            raise ValueError(
                "The parameters and the time window are required to compute the matrix. "
                "You can update its values by calling the class: EV(params, t)."
            )
        y0 = jnp.eye(2 ** len(self.wires), dtype=complex)

        with jax.ensure_compile_time_eval():
            H_jax = ParametrizedHamiltonianPytree.from_hamiltonian(
                self.H, dense=self.dense, wire_order=self.wires
            )

        def fun(y, t):
            """dy/dt = -i H(t) y"""
            return (-1j * H_jax(self.data, t=t)) @ y

        mat = odeint(fun, y0, self.t, **self.odeint_kwargs)
        if self.hyperparameters["return_intermediate"] and self.hyperparameters["complementary"]:
            # Compute U(t_0, t_f)@U(t_0, t_i)^\dagger, where i indexes the first axis of mat
            mat = qml.math.tensordot(mat[-1], qml.math.conj(mat), axes=[[1], [-1]])
            # The previous line leaves the axis indexing the t_i as second, so we move it up
            mat = qml.math.moveaxis(mat, 1, 0)
        elif not self.hyperparameters["return_intermediate"]:
            mat = mat[-1]
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> H = qml.X(1) + qml.pulse.constant * qml.Y(0) + jnp.polyval * qml.Y(1)
        >>> params = [0.2, [1, 2, 3]]
        >>> op = qml.evolve(H)(params, t=2)
        >>> cache = {'matrices': []}

        >>> op.label()
        "Parametrized\nEvolution"
        >>> op.label(decimals=2, cache=cache)
        "Parametrized\nEvolution\n(p=[0.20,M0], t=[0. 2.])"
        >>> op.label(base_label="my_label")
        "my_label"
        >>> op.label(decimals=2, base_label="my_label", cache=cache)
        "my_label\n(p=[0.20,M0], t=[0. 2.])"

        Array-like parameters are stored in ``cache['matrices']``.
        """
        op_label = base_label or "Parametrized\nEvolution"

        if self.num_params == 0:
            return op_label

        if decimals is None:
            return op_label

        params = self.parameters
        has_cache = cache and isinstance(cache.get("matrices", None), list)

        if any(qml.math.ndim(p) for p in params) and not has_cache:
            return op_label

        def _format_number(x):
            return format(qml.math.toarray(x), f".{decimals}f")

        def _format_arraylike(x):
            for i, mat in enumerate(cache["matrices"]):
                if qml.math.shape(x) == qml.math.shape(mat) and qml.math.allclose(x, mat):
                    return f"M{i}"
            mat_num = len(cache["matrices"])
            cache["matrices"].append(x)
            return f"M{mat_num}"

        param_strings = [_format_arraylike(p) if p.shape else _format_number(p) for p in params]

        p = ",".join(s for s in param_strings)
        return f"{op_label}\n(p=[{p}], t={self.t})"


@functions.bind_new_parameters.register
def _bind_new_parameters_parametrized_evol(op: ParametrizedEvolution, params: Sequence[TensorLike]):
    return ParametrizedEvolution(
        op.H,
        params=params,
        t=op.t,
        return_intermediate=op.hyperparameters["return_intermediate"],
        complementary=op.hyperparameters["complementary"],
        dense=op.dense,
        **op.odeint_kwargs,
    )
