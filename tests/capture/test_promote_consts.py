# Copyright 2018-2025 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""Unit tests for qml.capture.promote_consts."""

import pytest

from pennylane.capture import promote_consts

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.jax
def test_promote_consts():
    """Test that promote consts works as expected."""

    const = jnp.array([2.0])

    def f(x):
        return const + x

    args = (jnp.array([0.5]),)

    jaxpr = jax.make_jaxpr(f)(*args)

    new_jaxpr, new_args = promote_consts(jaxpr, args)

    assert isinstance(new_jaxpr, jax.extend.core.Jaxpr)
    assert new_jaxpr.constvars == []

    assert len(new_args) == 2
    assert jnp.allclose(new_args[0], const)
    assert jnp.allclose(new_args[1], args[0])
