# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file tests the ClassicalConstant measurment process."""
# pylint: disable=protected-access
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import ClassicalConstant


class TestClassicalConstantUnit:
    """Unit tests for methods on ClassicalConstant"""

    def test_initialization(self):
        """Test the initialization of ClassicalConstant."""

        mp = ClassicalConstant((1, 2, 3))
        assert mp.constant == (1, 2, 3)
        assert mp.obs is None
        assert mp.wires == qml.wires.Wires(())

    def test_pytree(self):
        """Test that ClassicalConstant is a pytree."""

        mp = ClassicalConstant(3)
        data, leaves = mp._flatten()
        assert data == (3,)
        assert leaves is None
        new_mp = ClassicalConstant._unflatten(data, leaves)
        qml.assert_equal(new_mp, mp)

    def test_shape(self):
        """Tests for the shape method."""

        assert ClassicalConstant(1).shape() == ()
        assert ClassicalConstant(1).shape(100, 2) == ()

        assert ClassicalConstant(np.zeros((4, 2))).shape() == (4, 2)
        assert ClassicalConstant((1, 2, 3)).shape() == (3,)

    def test_numeric_type(self):
        """Test for the numeric type."""

        assert ClassicalConstant(1).numeric_type == int
        assert ClassicalConstant(np.zeros(2, dtype=np.int32)).numeric_type == np.int32
        assert ClassicalConstant(np.zeros(2, dtype=np.complex128)).numeric_type == np.complex128

    def test_repr(self):
        """Test for the repr."""
        assert repr(ClassicalConstant(np.array([0, 1]))) == "ClassicalConstant([0 1])"

    def test_hash(self):
        """Tests for the hash."""
        mp1 = ClassicalConstant(1)
        mp2 = ClassicalConstant(1)
        mp3 = ClassicalConstant(np.array([1, 2, 3]))
        mp4 = ClassicalConstant(np.array([1, 2, 3]))

        assert hash(mp1) == hash(mp2)
        assert hash(mp1) != hash(mp3)
        assert hash(mp3) == hash(mp4)

    def test_comparison(self):
        """Test for equality."""
        mp1 = ClassicalConstant(1)
        mp2 = ClassicalConstant(1)
        mp3 = ClassicalConstant(np.array([1, 2, 3]))
        mp4 = ClassicalConstant(np.array([1, 2, 3]))

        qml.assert_equal(mp1, mp2)
        qml.assert_equal(mp3, mp4)
        assert not qml.equal(mp1, mp3)

        assert not qml.equal(mp3, ClassicalConstant(qml.numpy.array([1, 2, 3])))
        assert not qml.equal(mp3, ClassicalConstant(np.array([1.1, 2, 3])))
        assert not qml.equal(
            ClassicalConstant(qml.numpy.array(1, requires_grad=True)),
            ClassicalConstant(qml.numpy.array(1, requires_grad=False)),
        )

    def test_process_methods(self):
        """Tests for the various process methods."""

        mp = ClassicalConstant(10)

        assert (
            mp.process_samples(
                [[0, 0, 0, 0]],
                (0,),
            )
            == 10
        )
        assert mp.process_counts({0: 100}, (0,)) == 10
        assert (
            mp.process_state(
                np.zeros(
                    4,
                ),
                (0,),
            )
            == 10
        )
        assert mp.process_density_matrix(np.zeros((4, 4)), (0,)) == 10

    @pytest.mark.capture
    def test_primitives_none(self):
        """Test that the standard measurement primitives are all None."""
        assert ClassicalConstant._obs_primitive is None
        assert ClassicalConstant._mcm_primitive is None
        assert ClassicalConstant._wires_primitive is None


@pytest.mark.parametrize("device_name", ("default.qubit", "reference.qubit"))
@pytest.mark.parametrize("shots", (None, 10))
def test_qnode_execution(device_name, shots):
    """Test that some devices can handle returning a ClassicalConstant."""

    @qml.qnode(qml.device(device_name, wires=2), shots=shots)
    def f():
        return ClassicalConstant(np.array([0, 1, 2]))

    out = f()
    assert qml.math.allclose(out, np.array([0, 1, 2]))


def test_differentiation():
    """Test that classical constants can be differentiated."""

    @qml.qnode(qml.device("reference.qubit", wires=2))
    def c(x):
        return qml.measurements.ClassicalConstant(x**2)

    g = qml.grad(c)(qml.numpy.array(2.0))
    assert qml.math.allclose(g, 4)


@pytest.mark.capture
def test_capture_into_plxpr():
    """Test that ClassicalConstant can be captured into plxpr."""

    import jax  # pylint: disable=import-outside-toplevel

    @qml.qnode(qml.device("default.qubit", wires=2))
    def c(x):
        return qml.measurements.ClassicalConstant(x)

    jaxpr = jax.make_jaxpr(c)(np.array([0, 1]))

    assert jaxpr.out_avals == [jax.core.ShapedArray((2,), int)]
    qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
    assert (
        qfunc_jaxpr.eqns[0].primitive == ClassicalConstant._primitive
    )  # pylint: disable=protected-access
    assert qfunc_jaxpr.eqns[0].invars[0] == qfunc_jaxpr.invars[0]
