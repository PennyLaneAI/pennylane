# Copyright 2023 Xanadu Quantum Technologies Inc.

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
.. currentmodule:: pennylane

QJIT API
--------

.. autosummary::
    :toctree: api

    ~qjit

Compiler developer functions
----------------------------

.. autosummary::
    :toctree: api

    ~compiler.available_compilers
    ~compiler.available
    ~compiler.active

Compiler
--------
The compiler package exclusively functions as a wrapper for PennyLane's JIT compiler packages, without independently
implementing any compiler itself. Currently, it supports the ``pennylane-catalyst`` package, with plans to
incorporate additional packages in the near future.

For any compiler packages seeking to be registered, it is imperative that they expose the 'entry_points' metadata
under the designated group name: ``pennylane.compilers``.

Basic usage
-----------

    In just-in-time (JIT) mode, the compilation is triggered at the call site the
    first time the quantum function is executed. For example, ``circuit`` is
    compiled as early as the first call.

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(wires=1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    Alternatively, if argument type hints are provided, compilation
    can occur 'ahead of time' when the function is decorated.

    .. code-block:: python

        from jax.core import ShapedArray

        @qjit  # compilation happens at definition
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

    >>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
    array([0.75634905-0.52801002j, 0. +0.j,
        0.35962678+0.14074839j, 0. +0.j])

    Catalyst also supports capturing imperative Python control flow in compiled programs. You can
    enable this feature via the ``autograph=True`` parameter. Note that it does come with some
    restrictions, in particular whenever global state is involved. Refer to the documentation page
    for a complete discussion of the supported and unsupported use-cases.

    .. code-block:: python

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: int):

            if x < 5:
                qml.Hadamard(wires=0)
            else:
                qml.T(wires=0)

            return qml.expval(qml.PauliZ(0))

    >>> circuit(3)
    array(0.)

    >>> circuit(5)
    array(1.)
"""

from .compiler import available_compilers, available, active

from .qjit_api import qjit, while_loop, for_loop
