.. role:: html(raw)
   :format: html

.. _intro_ref_pulse:

Pulse control
=============

Pulse control is used in a variety of quantum systems for low-level control of quantum operations. A
time-dependent electromagnetic field tuned to the characteristic energies is applied,
leading to a time-dependent Hamiltonian interaction :math:`H(t)`. We call driving the system with such an
electromagnetic field for a fixed time window a *pulse sequence*. These pulse sequences can then be tuned to
implement the higher level gates used for quantum computation.

The :mod:`~.pulse` module provides functions and classes used to simulate pulse-level control of quantum
systems. It contains a :class:`~.ParametrizedHamiltonian` class, which can be used to define the time-dependent
Hamiltonian describing the interaction between the applied pulses and the system. A
:class:`~.ParametrizedHamiltonian` instance can be used to create a :class:`~.ParametrizedEvolution`
(an :class:`~.Operation`) that provides the time evolution under the Hamiltonian. The :mod:`~.pulse` module also
includes several convenience functions for defining pulse envelopes.

The :mod:`~.pulse` module relies on the external package `JAX <https://jax.readthedocs.io/en/latest/>`_, which
requires separate installation. The module is written for ``jax`` and will not work with other machine learning
frameworks typically encountered in PennyLane.



Creating a parametrized Hamiltonian
-----------------------------------

The :mod:`~.pulse` module provides a framework to create a time-dependent Hamiltonian of the form

.. math:: H(p, t) = H_\text{drift} + \sum_j f_j(p, t) H_j

with constant operators :math:`H_j` and scalar functions :math:`f_j(p, t)` that may depend on
parameters :math:`p` and time :math:`t`.

Defining a :class:`~.ParametrizedHamiltonian` requires coefficients and operators. In the example below, we define a
Hamiltonian with a single drift term, and two parametrized terms.

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

The functions defining the parameterized coefficients must have the call signature ``(p, t)``, where ``p`` can be a ``float``,
``list`` or ``jnp.array``. These functions should be defined using ``jax.numpy`` rather than ``numpy`` where relevant.

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

.. code-block:: python

    >>> H1
    ParametrizedHamiltonian: terms=3
    >>> params = [1.2, [2.3, 3.4]]  # f1 takes a single parameter, f2 takes 2
    >>> H1(params, t=0.5)
    (2*(PauliX(wires=[0]) @ PauliX(wires=[1]))) + ((-0.2876553535461426*(PauliY(wires=[0]) @ PauliY(wires=[1]))) + (1.5179612636566162*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))))
    >>> qml.equal(H1(params, t=0.5), H2(params, t=0.5))
    True

.. warning::
    The order of the coefficients and operators matters. When initializing the
     :class:`~.ParametrizedHamiltonian`, terms defined with fixed coefficients
    have to come before parametrized terms to prevent discrepancy in the wire order. When passing parameters, ensure
    that the order of the coefficient functions and the order of the parameters match.

Convenience functions for building a ParametrizedHamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following convenience functions currently are available in PennyLane to assist in creating coefficient functions
for a :class:`~.ParametrizedHamiltonian`:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.pulse.constant
    ~pennylane.pulse.pwc
    ~pennylane.pulse.pwc_from_function
    ~pennylane.pulse.rect

:html:`</div>`

Further examples
^^^^^^^^^^^^^^^^

A few additional examples of defining a :class:`~.ParametrizedHamiltonian` are provided here.

Using ``rect`` to create a parametrized coefficient that has a value of 0 outside the time interval
:math:`t=(1, 7)`, and is defined by ``jnp.polyval(p, t)`` within the interval:

.. code-block:: python


    def f(p, t):
        return jnp.polyval(p, t)
    H = qml.pulse.rect(f1, windows=[(1, 7)]) * qml.PauliX(0)

    # inside the window
    >>> H([3], t=2)
    2.7278921604156494*(PauliX(wires=[0]))

    # outside the window
    H([3], t=0.5 )
    >>> 0.0*(PauliX(wires=[0]))

Using ``pwc`` to create a parametrized coefficient function that is piecewise constant
within the interval ``t``, and 0 outside of it.

.. code-block:: python

    from pennylane.pulse.convenience_functions import pwc

    f1 = pwc(timespan=(2, 7))  # TODO: maybe this should be renamed window for uniformity?
    H = f1 * qml.PauliX(0)

    # passing pwc((2, 7)) an array evenly distributes the array values in the interval t=2 to t=7
    H(params=[[1, 2, 3, 4, 5]], t=2.3)
    >>> 1.0*(PauliX(wires=[0]))

    # different time, same bin, same result
    H(params=[[1, 2, 3, 4, 5]], t=2.5)
    >>> 1.0*(PauliX(wires=[0]))

    # next bin
    H(params=[[1, 2, 3, 4, 5]], t=3.1)
    >>> 2.0*(PauliX(wires=[0]))

    # outside the window where the function is assigned non-zero values
    H(params=[[1, 2, 3, 4, 5]], t=8)
    >>> 0.0*(PauliX(wires=[0]))

ParametrizedEvolution
---------------------
During a pulse sequence spanning time :math:`(t_0, t_1)`, the state evolves according to the time-dependent Schrodinger equation

.. math:: \frac{\partial}{\partial t} |\psi\rangle = -i H(t) |\psi\rangle

realizing a unitary evolution :math:`U(t_0, t_1)` of the input state, i.e.

.. math:: |\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle

A :class:`~.ParametrizedEvolution` is this solution :math:`U(t_0, t_1)` to the time-dependent
Schrodinger equation for a :class:`~.ParametrizedHamiltonian`.

The :class:`~.ParametrizedEvolution` class uses a numerical ordinary differential equation
solver (`here <https://github.com/google/jax/blob/main/jax/experimental/ode.py>`_). It
can be created using the :func:`~qml.evolve` function:

.. code-block:: python

    from jax import numpy as jnp

    f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
    H = 2 * qml.PauliX(0) + f1 * qml.PauliY(1)
    ev = qml.evolve(H)
    ev
    >>> ParametrizedEvolution(wires=[0, 1])

The initial :class:`~.ParametrizedEvolution` does not have parameters defined, and so will
not have a matrix defined. To obtain an :class:`~.Operator` with a matrix, it must be passed
parameters and a time interval:

.. code-block:: python

    ev([1.2], t=[0, 4]).matrix()
    >>> Array([[-0.14115842+0.j        ,  0.03528605+0.j        ,
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

.. code-block:: python

    qml.evolve(H)(params=[1.2], t=[0, 4], atol=1e-6, mxstep=1)

These options are:

- ``atol (float, optional)``: Absolute error tolerance
- ``rtol (float, optional)``: Relative error tolerance
- ``mxstep (int, optional)``: maximum number of steps to take for each time point
- ``hmax (float, optional)``: maximum step size

If not specified, they will default to predetermined values. See :class:`.ParametrizedEvolution` for details.

Using qml.evolve in a QNode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~.ParametrizedEvolution` can be implemented in the QNode in the same way as any other
PennyLane :class:`~.Operator`.

To look at an example of this, let's start with two instances of :class:`~.ParametrizedHamiltonian`:

.. code-block:: python

    ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
    coeffs = [lambda p, t: p for _ in range(3)]
    H1 = qml.ops.dot(coeffs, ops)  # time-independent parametrized hamiltonian

.. code-block:: python

    ops = [qml.PauliZ(0), qml.PauliY(1), qml.PauliX(2)]
    coeffs = [lambda p, t: p * jnp.sin(t) for _ in range(3)]
    H2 = qml.ops.dot(coeffs, ops) # time-dependent parametrized hamiltonian

Now we can execute the evolution of these Hamiltonians applied simultaneously:

.. code-block:: python

    dev = qml.device("default.qubit", wires=3)
    @qml.qnode(dev, interface="jax")
    def circuit(params):
        qml.evolve(H1 + H2)(params, t=[0, 10])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    params = jnp.array([1., 2., 3., 4., 5., 6.])
    circuit(params)
    >>> Array(-0.78236955, dtype=float32)


.. warning::
    In this example, it is important that ``H1`` and ``H2`` are included in the same ``qml.evolve`` operation.
    For non-commuting operations, applying ``qml.evolve(H1)(params, t=[0, 10])`` followed by
    ``qml.evolve(H2)(params, t=[0, 10])`` will NOT apply the two pulses simultaneously, despite the overlapping
    time window. Instead, it will execute ``H1`` in the ``[0, 10]`` time window, and then subsequently execute
    ``H2`` using the same time window to calculate the evolution, but without taking into account how the time
    evolution of ``H1`` affects the evolution of ``H2`` and vice versa.

We can also compute the gradient of this evolution with respect to the input parameters:

.. code-block:: python

    jax.grad(circuit)(params)
    >>> Array([-4.8066125,  3.7038102, -1.3294725, -2.4061902,  0.6811545,
        -0.5226515], dtype=float32)

