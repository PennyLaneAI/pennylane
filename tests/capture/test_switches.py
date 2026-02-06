# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the backbone of the :mod:`pennylane.capture` module.
"""
import pytest

import pennylane as qp


@pytest.mark.jax
def test_switches_with_jax():
    """Test switches and status reporting function."""

    assert qp.capture.enabled() is False
    assert qp.capture.enable() is None
    assert qp.capture.enabled() is True
    assert qp.capture.disable() is None
    assert qp.capture.enabled() is False


# pylint: disable=unused-argument
@pytest.mark.capture
def test_pause():
    """Test that pause recording stops program capture."""
    import jax

    def f():
        with qp.capture.pause():
            qp.X(0)
        return 2

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 0


# pylint: disable=unused-argument
@pytest.mark.capture
def test_pause_with_error():
    """Test that program capture stays enabled if an error occurs."""

    def f():
        with qp.capture.pause():
            raise ValueError
        return 2

    with pytest.raises(ValueError):
        f()

    assert qp.capture.enabled()


def test_switches_without_jax():
    """Test switches and status reporting function."""
    # We want to skip the test if jax is installed
    try:
        # pylint: disable=import-outside-toplevel, unused-import
        import jax
    except ImportError:

        assert qp.capture.enabled() is False
        with pytest.raises(ImportError, match="plxpr requires JAX to be installed."):
            qp.capture.enable()
        assert qp.capture.enabled() is False
        assert qp.capture.disable() is None
        assert qp.capture.enabled() is False
