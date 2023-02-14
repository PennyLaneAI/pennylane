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

The :mod:`~.pulse` module is written for ``jax`` and will not work with other machine learning frameworks typically encountered in PennyLane. It requires separate installation, see `jax.readthedocs.i <https://jax.readthedocs.io/en/latest/>`_.

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
"""

from .convenience_functions import constant, rect, pwc, pwc_from_function
from .parametrized_evolution import ParametrizedEvolution
from .parametrized_hamiltonian import ParametrizedHamiltonian
