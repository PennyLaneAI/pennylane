qml.pulse
=========

.. automodule:: pennylane.pulse

Creating a parametrized Hamiltonian
-----------------------------------

The :mod:`~.pulse` module provides a framework to create a time-dependent Hamiltonian of the form

.. math::
    H(p, t) = H_\text{drift} + \sum_j f_j(p, t) H_j

with constant operators :math:`H_j` and scalar functions :math:`f_j(p, t)` that may depend on
parameters :math:`p` and time :math:`t`.

Defining a :class:`~.ParametrizedHamiltonian` requires coefficients and operators, where some of the coefficients
are callables. The callables defining the parameterized coefficients must have the call signature ``(p, t)``, where ``p`` can be a ``float``,
``list`` or ``jnp.array``. These functions should be defined using ``jax.numpy`` rather than ``numpy`` where relevant.

.. code-block:: python

    import pennylane as qml
    from jax import numpy as jnp

    # defining the coefficients fj(v, t) for the two parametrized terms
    f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
    f2 = lambda p, t: p[0] * jnp.cos(p[1]* t ** 2)

    # defining the operations for the three terms in the Hamiltonian
    XX = qml.PauliX(0) @ qml.PauliX(1)
    YY = qml.PauliY(0) @ qml.PauliY(1)
    ZZ = qml.PauliZ(0) @ qml.PauliZ(1)



There are two ways to construct a :class:`~.ParametrizedHamiltonian` from the coefficients
and operators:

.. code-block:: python

    # Option 1
    H1 =  2 * XX + f1 * YY + f2 * ZZ

    # Option 2
    coeffs = [2, f1, f2]
    ops = [XX, YY, ZZ]
    H2 =  qml.dot(coeffs, ops)


.. warning::
    When initializing a :class:`~.ParametrizedHamiltonian` via a list of parametrized coefficients, it
    is possible to create a list of multiple coefficients of the same form iteratively using lambda
    functions, i.e.

    ``coeffs = [lambda p, t: p for _ in range(3)]``.

    Do **not**, however, define the function as dependent on the value that is iterated over. That is, it is not
    possible to define ``coeffs = [lambda p, t: p * t**i for i in range(3)]`` to create a list
    ``coeffs = [(lambda p, t: p), (lambda p, t: p * t), (lambda p, t: p * t**2)]``. The value of ``i`` when
    creating the lambda functions is set to be the final value in the iteration, such that this will
    produce three identical functions ``coeffs = [(lambda p, t: p * t**2)] * 3``.

The :class:`~.ParametrizedHamiltonian` is a callable, and can return an :class:`~.Operator` if passed a set of
parameters and a time at which to evaluate the coefficients :math:`f_j`.

>>> H1
ParametrizedHamiltonian: terms=3

>>> params = [1.2, [2.3, 3.4]]  # f1 takes a single parameter, f2 takes 2
>>> H1(params, t=0.5)
(2*(PauliX(wires=[0]) @ PauliX(wires=[1]))) + ((-0.2876553535461426*(PauliY(wires=[0]) @ PauliY(wires=[1]))) + (1.5179612636566162*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))))


When passing parameters, ensure that the order of the coefficient functions and the order of
the parameters match.

When initializing a :class:`~.ParametrizedHamiltonian`, terms defined with fixed coefficients have to come
before parametrized terms to prevent discrepancy in the wire order.


ParametrizedEvolution
---------------------
During a pulse sequence spanning time :math:`(t_0, t_1)`, the state evolves according to the time-dependent Schrodinger equation

.. math::
    \frac{\partial}{\partial t} |\psi\rangle = -i H(t) |\psi\rangle

realizing a unitary evolution :math:`U(t_0, t_1)` of the input state, i.e.

.. math::
    |\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle

A :class:`~.ParametrizedEvolution` is this solution :math:`U(t_0, t_1)` to the time-dependent
Schrödinger equation for a :class:`~.ParametrizedHamiltonian`.

.. note::
    The :class:`~.ParametrizedHamiltonian` must be Hermitian at all times. This is not explicitly
    checked; ensuring a correctly defined Hamiltonian is the responsibility of the user.

The :class:`~.ParametrizedEvolution` class uses a numerical ordinary differential equation
solver (see `jax.experimental.ode <https://github.com/google/jax/blob/main/jax/experimental/ode.py>`_). It
can be created using the :func:`~.functions.evolve` function:

.. code-block:: python

    from jax import numpy as jnp

    f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
    H = 2 * qml.PauliX(0) + f1 * qml.PauliY(1)
    ev = qml.evolve(H)

>>> ev
ParametrizedEvolution(wires=[0, 1])

The initial :class:`~.ParametrizedEvolution` does not have parameters defined, and so will
not have a matrix defined. To obtain an :class:`~.Operator` with a matrix, we have to pass
parameters and a time interval:

>>> ev([1.2], t=[0, 4]).matrix()
Array([[-0.14115842+0.j        ,  0.03528605+0.j        ,
         0.        -0.95982337j,  0.        +0.23993255j],
       [-0.03528605+0.j        , -0.14115842+0.j        ,
         0.        -0.23993255j,  0.        -0.95982337j],
       [ 0.        -0.95982337j,  0.        +0.23993255j,
        -0.14115842+0.j        ,  0.03528605+0.j        ],
       [ 0.        -0.23993255j,  0.        -0.95982337j,
        -0.03528605+0.j        , -0.14115842+0.j        ]],      dtype=complex64)


The parameters can be updated by calling the :class:`~.ParametrizedEvolution` again with different inputs.

Additional options with regards to how the matrix is calculated can be passed to the :class:`.ParametrizedEvolution`
along with the parameters, as keyword arguments:

>>> qml.evolve(H)(params=[1.2], t=[0, 4], atol=1e-6, mxstep=1)

The available keyword arguments can be found in in :class:`~.ParametrizedEvolution`. If not specified, they
will default to predetermined values.

Using qml.evolve in a QNode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~.ParametrizedEvolution` can be implemented in a QNode. We will evolve the
:class:`~.ParametrizedHamiltonian`:

.. code-block:: python

        from jax import numpy as jnp

        f1 = lambda p, t: jnp.sin(p * t)
        H = f1 * qml.PauliY(0)


Now we can execute the evolution of this Hamiltonian in a QNode and compute its gradient:

.. code-block:: python

    import jax

    dev = qml.device("default.qubit", wires=1)
    @jax.jit
    @qml.qnode(dev, interface="jax")
    def circuit(params):
        qml.evolve(H)(params, t=[0, 10])
        return qml.expval(qml.PauliZ(0))

>>> params = [1.2]
>>> circuit(params)
Array(0.96632576, dtype=float32)

>>> jax.grad(circuit)(params)
Array([2.3569832], dtype=float32)

We can use the decorator ``jax.jit`` to compile this execution just-in-time. This means the first execution
will typically take a little longer with the benefit that all following executions will be significantly faster.
JIT-compiling is optional, and one can remove the decorator when only single executions are of interest. See the
``jax`` docs on jitting for more information.

.. warning::
    To find the simultaneous evolution of the two operators, so it is important that they are both included
    in the same :func:`~.functions.evolve`. For non-commuting operations, applying
    ``qml.evolve(H1)(params, t=[0, 10])`` followed by ``qml.evolve(H2)(params, t=[0, 10])`` will **not**
    apply the two pulses simultaneously, despite the overlapping time window.

    See Usage Details of :class:`~.ParametrizedEvolution` for a detailed example.