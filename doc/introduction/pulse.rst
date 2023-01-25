.. role:: html(raw)
   :format: html

.. _intro_ref_pulse:

Pulse Control
=============

Pulse control is used in a variety of quantum systems for low-level control of quantum operations. A
time-dependent electromagnetic field tuned to the characteristic energies is applied,
leading to a time-dependent Hamiltonian interaction :math:`H(t)`. We call driving the system with such an
electromagnetic field for a fixed time window a *pulse sequence*. These pulse sequences can then be tuned to
implement the higher level gates used for quantum computation.

The :mod:`~.pulse` module provides functions and classes used to simulate pulse-level control of quantum
systems. It contains a :class:`~.ParametrizedHamiltonian` class, which can be used to define the time-dependent
Hamiltonian describing the interaction between the applied pulses and the system. The
:class:`~.ParametrizedHamiltonian` can be used to create a :class:`~.ParametrizedEvolution`
(:class:`~.Operation`) that provides the time evolution of the Hamiltonian. The :mod:`~.pulse` module also
includes several convenience functions for defining pulse envelopes.

The :mod:`~.pulse` module relies on the external package `JAX <https://jax.readthedocs.io/en/latest/>`_, which
requires separate installation. The module is written for ``jax`` and will not work with the other machine learning
frameworks typically encountered in PennyLane.



Creating a parametrized Hamiltonian
-----------------------------------

The :mod:`~.pulse` module provides a framework to create a time-dependent Hamiltonian of the form

.. math:: H(p, t) = H_\text{drift} + \sum_j f_j(p, t) H_j

with constant operators :math:`H_j` and scalar functions :math:`f_j(p, t)` that may depend on
parameters :math:`p` and time :math:`t`.

.. note::
    The :class:`~.ParametrizedHamiltonian` is not a PennyLane :class:`~.Operator`. If an :class:`~.Operator`
    representing the :class:`~.ParametrizedHamiltonian` is needed, the initialized :class:`~.ParametrizedHamiltonian`
    must be called with fixed parameters and time.

Defining a ``ParametrizedHamiltonian`` requires coefficients and operators. In the example below, we define a
Hamiltonian with a single drift term, and two parametrized terms.

.. code-block:: python

    import pennylane as qml
    from jax import numpy as np

    # defining the coefficients fj(v, t) for the two parametrized terms
    f1 = lambda p, t: p * np.sin(t) * (t - 1)
    f2 = lambda p, t: p[0] * np.cos(p[1]* t ** 2)

    # defining the operations for the three terms in the Hamiltonian
    XX = qml.PauliX(0) @ qml.PauliX(1)
    YY = qml.PauliY(0) @ qml.PauliY(1)
    ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

There are two way to construct a :class:`~.ParametrizedHamiltonian` from the coefficients and operators:

.. code-block:: python

    # Option 1
    H1 =  2 * XX + f1 * YY + f2 * ZZ

    # Option 2
    coeffs = [2, f1, f2]
    ops = [XX, YY, ZZ]
    H2 =  qml.ops.dot(coeffs, ops)

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
    Order matters here. When initializing the :class:`~.ParametrizedHamiltonian`, terms defined with fixed coefficients
    should come before parametrized terms to prevent discrepancy in the wire order. When passing parameters, ensure
    that the order of the coefficient functions and the order of the parameters match.

Convenience functions for building a ParametrizedHamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The convenience functions currently available in PennyLane to assist in creating coefficients functions
for a :class:`~.ParametrizedHamiltonian` are:

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

A few additional examples are provided here...

The ``rect`` function defines can be used to create a parametrized hamiltonian

.. code-block:: python

    >>> def f1(p, t):
    ...     return jnp.polyval(p, t)
    >>> windows = [(1, 7), (9, 14)]
    >>> H = qml.pulse.rect(f1, windows) * qml.PauliX(0)

# pwc example also once merged



ParametrizedEvolution
---------------------

A :class:`~.ParametrizedEvolution` is the solution :math:`U(t_1, t_2)` to the time-dependent
Schrodinger equation for a :class:`~.ParametrizedHamiltonian`:

.. math:: \frac{d}{d t}U(t) = -i H(p, t) U(t).

Creation of the :class:`~.ParametrizedEvolution` uses an a numerical ordinary differential equation
solver (`here <https://github.com/google/jax/blob/main/jax/experimental/ode.py>`_).

EXAMPLE

The parameters can be updated...
A call of a :class:`~.ParametrizedEvolution` will return a normal :class:`~Operator` defining the time
evolution for the input parameters.

.. note::
    The :class:`~.ParametrizedEvolution` does not have parameters defined in the intial... etc.



During a pulse sequence, the state evolves according to the time-dependent Schrodinger equation

.. math:: \frac{\partial}{\partial t} |\psi\rangle = -i H(t) |\psi\rangle

realizing a unitary evolution :math:`U(t_0, t_1)` from times :math:`t_0` to :math:`t_1` of the input state, i.e.
:math:`|\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle`.