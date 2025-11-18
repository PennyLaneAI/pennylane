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
"""
Tests for the Patcher context manager and patching utilities.
"""
import pytest

import pennylane as qml
from pennylane.capture.jax_patches import get_jax_patches
from pennylane.capture.patching import Patcher

jax = pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.capture]
# pylint: disable=too-few-public-methods


class TestPatcher:
    """Tests for the Patcher context manager."""

    def test_single_patch(self):
        """Test patching a single attribute."""

        class MockObject:
            value = "original"

        obj = MockObject()
        assert obj.value == "original"

        with Patcher((obj, "value", "patched")):
            assert obj.value == "patched"

        assert obj.value == "original"

    def test_multiple_patches(self):
        """Test patching multiple attributes simultaneously."""

        class MockObject:
            value1 = "original1"
            value2 = "original2"

        obj = MockObject()

        with Patcher((obj, "value1", "patched1"), (obj, "value2", "patched2")):
            assert obj.value1 == "patched1"
            assert obj.value2 == "patched2"

        assert obj.value1 == "original1"
        assert obj.value2 == "original2"

    def test_nested_patchers(self):
        """Test nested Patcher contexts."""

        class MockObject:
            value = "original"

        obj = MockObject()

        with Patcher((obj, "value", "patch1")):
            assert obj.value == "patch1"

            with Patcher((obj, "value", "patch2")):
                assert obj.value == "patch2"

            assert obj.value == "patch1"

        assert obj.value == "original"

    def test_patch_method(self):
        """Test patching a method."""

        class MockObject:
            def method(self):
                return "original"

        obj = MockObject()
        assert obj.method() == "original"

        def patched_method(self):
            return "patched"

        with Patcher((obj.__class__, "method", patched_method)):
            assert obj.method() == "patched"

        assert obj.method() == "original"

    def test_exception_in_context(self):
        """Test that patches are restored even when exception occurs."""

        class MockObject:
            value = "original"

        obj = MockObject()

        with pytest.raises(ValueError):
            with Patcher((obj, "value", "patched")):
                assert obj.value == "patched"
                raise ValueError("test exception")

        # Patch should be restored even though exception was raised
        assert obj.value == "original"


class TestGetJaxPatches:
    """Tests for JAX patches integration."""

    def test_get_jax_patches_returns_tuple(self):
        """Test that get_jax_patches returns a tuple."""

        patches = get_jax_patches()
        assert isinstance(patches, tuple)

    def test_patches_with_patcher_context(self):
        """Test that patches can be used with Patcher."""

        qml.capture.enable()

        def f(x):
            return jax.numpy.sin(x) + jax.numpy.cos(x)

        # Should work with patches applied
        with Patcher(*get_jax_patches()):
            jaxpr = jax.make_jaxpr(f)(0.5)
            assert jaxpr is not None

        qml.capture.disable()

    def test_patches_are_temporary(self):
        """Test that patches don't leak outside context.

        This test verifies that the Patcher context manager properly saves and
        restores state, ensuring patches are truly temporary and don't leak.
        """
        from jax._src import pjit
        from jax._src.lax import lax

        # Get the current state before our Patcher context (original JAX state)
        original_dyn_shape = lax._dyn_shape_staging_rule
        original_pjit_staging = pjit.pjit_staging_rule

        # Get the patches
        patches = get_jax_patches()

        with Patcher(*patches):
            # Inside context, functions should be replaced with new patch instances
            patched_dyn_shape = lax._dyn_shape_staging_rule
            patched_pjit_staging = pjit.pjit_staging_rule

            # These should be different objects from the originals
            assert patched_dyn_shape is not original_dyn_shape
            assert patched_pjit_staging is not original_pjit_staging

        # After exiting context, should be back to the original JAX state
        after_dyn_shape = lax._dyn_shape_staging_rule
        after_pjit_staging = pjit.pjit_staging_rule

        # Should be restored to original state
        assert after_dyn_shape is original_dyn_shape
        assert after_pjit_staging is original_pjit_staging
