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
Tests the support for JAX arrays in the ``Wires`` class.
"""
import pytest

from pennylane.wires import WireError, Wires

jax = pytest.importorskip("jax")

pytestmark = pytest.mark.jax


class TestWiresJax:
    """Tests the support for JAX arrays in the ``Wires`` class."""

    @pytest.mark.parametrize(
        "iterable, expected",
        [
            (jax.numpy.array([0, 1, 2]), (0, 1, 2)),
            (jax.numpy.array([0]), (0,)),
            (jax.numpy.array(0), (0,)),
            (jax.numpy.array([]), ()),
        ],
    )
    def test_creation_from_jax_array(self, iterable, expected):
        """Tests that a Wires object can be created from a JAX array."""

        wires = Wires(iterable)
        assert wires.labels == expected

    @pytest.mark.parametrize(
        "input",
        [
            [jax.numpy.array([0, 1, 2]), jax.numpy.array([3, 4])],
            [jax.numpy.array([0, 1, 2]), 3],
            jax.numpy.array([[0, 1, 2]]),
        ],
    )
    def test_error_for_incorrect_jax_arrays(self, input):
        """Tests that a Wires object cannot be created from incorrect JAX arrays."""

        with pytest.raises(WireError, match="Wires must be hashable"):
            Wires(input)

    @pytest.mark.parametrize("iterable", [jax.numpy.array([4, 1, 1, 3]), jax.numpy.array([0, 0])])
    def test_error_for_repeated_wires_jax(self, iterable):
        """Tests that a Wires object cannot be created from a JAX array with repeated indices."""

        with pytest.raises(WireError, match="Wires must be unique"):
            Wires(iterable)

    def test_array_representation_jax(self):
        """Tests that Wires object has an array representation with JAX."""

        wires = Wires([4, 0, 1])
        array = jax.numpy.array(wires.labels)
        assert isinstance(array, jax.numpy.ndarray)
        assert array.shape == (3,)
        for w1, w2 in zip(array, jax.numpy.array([4, 0, 1])):
            assert w1 == w2

    @pytest.mark.parametrize(
        "source", [jax.numpy.array([0, 1, 2]), jax.numpy.array([0]), jax.numpy.array(0)]
    )
    def test_jax_wires_pytree(self, source):
        """Test that Wires class supports the PyTree flattening interface with JAX arrays."""

        wires = Wires(source)
        wires_flat, tree = jax.tree_util.tree_flatten(wires)
        wires2 = jax.tree_util.tree_unflatten(tree, wires_flat)
        assert isinstance(wires2, Wires), f"{wires2} is not Wires"
        assert wires == wires2, f"{wires} != {wires2}"
