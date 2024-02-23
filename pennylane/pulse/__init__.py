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
r"""
Pulse programming is used in a variety of quantum systems for low-level control of quantum operations. A
time-dependent electromagnetic field tuned to the characteristic energies is applied,
leading to a time-dependent Hamiltonian interaction :math:`H(t)`. Driving the system with such an
electromagnetic field for a fixed time window is a **pulse program**. This pulse program can be tuned to
implement the higher level gates used for quantum computation.

The :mod:`~.pulse` module provides functions and classes used to simulate pulse-level control of quantum
systems.

It contains a :class:`~.ParametrizedHamiltonian` and :class:`~.ParametrizedEvolution` class for
describing time-dependent Hamiltonian interactions. The :mod:`~.pulse` module also includes several convenience
functions for defining pulses.

The :mod:`~.pulse` module is written for ``jax`` and will not work with other machine learning frameworks
typically encountered in PennyLane. It requires separate installation, see
`jax.readthedocs.io <https://jax.readthedocs.io/en/latest/>`_.

For a demonstration of the basic pulse functionality in PennyLane and running a ctrl-VQE example, see our demo on
`differentiable pulse programming <https://pennylane.ai/qml/demos/tutorial_pulse_programming101.html>`_.

Overview
--------

Time evolution classes
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.pulse

.. autosummary::
    :toctree: api

    ~ParametrizedHamiltonian
    ~ParametrizedEvolution

Convenience Functions
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.pulse

.. autosummary::
    :toctree: api

    ~constant
    ~pwc
    ~pwc_from_function
    ~rect

Hardware Compatible Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.pulse

.. autosummary::
    :toctree: api

    ~rydberg_interaction
    ~rydberg_drive
    ~transmon_interaction
    ~transmon_drive


Creating a parametrized Hamiltonian
-----------------------------------

The :mod:`~.pulse` module provides a framework to create a time-dependent Hamiltonian of the form

.. math::
    H(\{v_j\}, t) = H_\text{drift} + \sum_j f_j(v_j, t) H_j

with constant operators :math:`H_j` and scalar functions :math:`f_j(v_j, t)` that may depend on
parameters :math:`p` and time :math:`t`.

Defining a :class:`~.ParametrizedHamiltonian` requires coefficients and operators, where some of the coefficients
are callables. The callables defining the parameterized coefficients must have the call signature ``(p, t)``, where ``p`` can be a ``float``,
``list`` or ``jnp.array``. These functions should be defined using ``jax.numpy`` rather than ``numpy`` where relevant.

.. code-block:: python

    import pennylane as qml
    from jax import numpy as jnp

    # defining the coefficients fj(p, t) for the two parametrized terms
    f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
    f2 = lambda p, t: p[0] * jnp.cos(p[1]* t ** 2)

    # defining the operations for the three terms in the Hamiltonian
    XX = qml.X(0) @ qml.X(1)
    YY = qml.Y(0) @ qml.Y(1)
    ZZ = qml.Z(0) @ qml.Z(1)



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

    ``coeffs = [lambda p, t: p * t for _ in range(3)]``.

    Be careful when defining coefficients using lambda functions within a list comprehension. Avoid
    doing ``coeffs = [lambda p, t: p * t**i for i in range(3)]``, which will only use the final index ``i=2``
    in the ``lambda`` and will thus behave as ``coeffs = [(lambda p, t: p * t**2)] * 3``.
    Instead, use ``coeffs = [lambda p, t, power=i: p * t**power for i in range(3)]``

The :class:`~.ParametrizedHamiltonian` is a callable, and can return an :class:`~.Operator` if passed a set of
parameters and a time at which to evaluate the coefficients :math:`f_j`.

>>> H1
(
    2 * X(0) @ X(1)
  + <lambda>(params_0, t) * Y(0) @ Y(1)
  + <lambda>(params_1, t) * Z(0) @ Z(1)
)

>>> params = [1.2, [2.3, 3.4]]  # f1 takes a single parameter, f2 takes 2
>>> H1(params, t=0.5)
(
    2 * (X(0) @ X(1))
  + -0.2876553231625218 * (Y(0) @ Y(1))
  + 1.517961235535459 * (Z(0) @ Z(1))
)


When passing parameters, ensure that the order of the coefficient functions and the order of
the parameters match.

When initializing a :class:`~.ParametrizedHamiltonian`, terms defined with fixed coefficients have to come
before parametrized terms to prevent discrepancy in the wire order.

.. note::
    The :class:`~.ParametrizedHamiltonian` must be Hermitian at all times. This is not explicitly
    checked; ensuring a correctly defined Hamiltonian is the responsibility of the user.


ParametrizedEvolution
---------------------
During a pulse program spanning time :math:`(t_0, t_1)`, the state evolves according to the time-dependent Schrodinger equation

.. math::
    \frac{\partial}{\partial t} |\psi\rangle = -i H(t) |\psi\rangle

realizing a unitary evolution :math:`U(t_0, t_1)` of the input state, i.e.

.. math::
    |\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle

A :class:`~.ParametrizedEvolution` is this solution :math:`U(t_0, t_1)` to the time-dependent
Schrödinger equation for a :class:`~.ParametrizedHamiltonian`.

The :class:`~.ParametrizedEvolution` class uses a numerical ordinary differential equation
solver (see `jax.experimental.ode <https://github.com/google/jax/blob/main/jax/experimental/ode.py>`_). It
can be created using the :func:`~.pennylane.evolve` function:

.. code-block:: python

    from jax import numpy as jnp

    f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
    H = 2 * qml.X(0) + f1 * qml.Y(1)
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
ParametrizedEvolution(Array(1.2, dtype=float32, weak_type=True), wires=[0, 1])

The available keyword arguments can be found in in :class:`~.ParametrizedEvolution`. If not specified, they
will default to predetermined values.

Using qml.evolve in a QNode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~.ParametrizedEvolution` can be implemented in a QNode. We will evolve the
following :class:`~.ParametrizedHamiltonian`:

.. code-block:: python

        from jax import numpy as jnp

        f1 = lambda p, t: jnp.sin(p * t)
        H = f1 * qml.Y(0)


Now we can execute the evolution of this Hamiltonian in a QNode and compute its gradient:

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

We can use the decorator ``jax.jit`` to compile this execution just-in-time. This means the first execution
will typically take a little longer with the benefit that all following executions will be significantly faster.
JIT-compiling is optional, and one can remove the decorator when only single executions are of interest. See the
``jax`` docs on jitting for more information.

.. warning::
    To find the simultaneous evolution of the two operators, it is important that they are included
    in the same :func:`~.pennylane.evolve`. For two non-commuting :class:`~.ParametrizedHamiltonian`'s, applying
    ``qml.evolve(H1)(params, t=[0, 10])`` followed by ``qml.evolve(H2)(params, t=[0, 10])`` will **not**
    apply the two pulses simultaneously, despite the overlapping time window. Instead, they will be evolved
    over the same timespan, but without taking into account how the evolution of ``H1`` affects ``H2``.

    See Usage Details of :class:`~.ParametrizedEvolution` for a detailed example.
"""

from .convenience_functions import constant, pwc, pwc_from_function, rect
from .parametrized_evolution import ParametrizedEvolution
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian, HardwarePulse, drive
from .rydberg import rydberg_interaction, rydberg_drive
from .transmon import transmon_interaction, transmon_drive
