# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for default.qubits execute_and_jvp method."""
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pennylane.devices.qubit.jaxpr_adjoint import (  # pylint: disable=wrong-import-position
    execute_and_jvp,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


def test_basic_circuit():
    """Test the calculation of results and jvp for a basic circuit."""

    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    jaxpr = jax.make_jaxpr(f)(0.5)

    args = (0.82,)
    tangents = (2.0,)

    results, dresults = execute_and_jvp(jaxpr.jaxpr, args, tangents, num_wires=1)

    assert len(results) == 1
    assert qml.math.allclose(results, jnp.cos(args[0]))
    assert len(dresults) == 1
    assert qml.math.allclose(dresults[0], tangents[0] * -jnp.sin(args[0]))


def test_abstract_zero_tangent():
    """Test we get results"""
