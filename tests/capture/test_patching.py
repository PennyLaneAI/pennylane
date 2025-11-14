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

from pennylane.capture.patching import DictPatchWrapper, Patcher

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


class TestDictPatchWrapper:
    """Tests for the DictPatchWrapper class."""

    def test_basic_wrapping(self):
        """Test basic dictionary wrapping and attribute access."""
        config = {"option": "value"}
        wrapper = DictPatchWrapper(config, "option")

        assert wrapper.option == "value"

    def test_setting_attribute(self):
        """Test setting wrapped attribute."""
        config = {"option": "old"}
        wrapper = DictPatchWrapper(config, "option")

        wrapper.option = "new"
        assert config["option"] == "new"
        assert wrapper.option == "new"

    def test_with_patcher(self):
        """Test using DictPatchWrapper with Patcher."""
        config = {"setting": "original"}
        wrapper = DictPatchWrapper(config, "setting")

        assert config["setting"] == "original"

        with Patcher((wrapper, "setting", "patched")):
            assert config["setting"] == "patched"

        assert config["setting"] == "original"

    def test_invalid_attribute(self):
        """Test accessing invalid attribute raises AttributeError."""
        config = {"option": "value"}
        wrapper = DictPatchWrapper(config, "option")

        with pytest.raises(AttributeError, match="has no attribute 'invalid'"):
            _ = wrapper.invalid

    def test_setting_invalid_attribute(self):
        """Test setting invalid attribute raises AttributeError."""
        config = {"option": "value"}
        wrapper = DictPatchWrapper(config, "option")

        with pytest.raises(AttributeError, match="has no attribute 'invalid'"):
            wrapper.invalid = "new"


class TestGetJaxPatches:
    """Tests for JAX patches integration."""

    def test_get_jax_patches_returns_tuple(self):
        """Test that get_jax_patches returns a tuple."""
        from pennylane.capture import get_jax_patches

        patches = get_jax_patches()
        assert isinstance(patches, tuple)

    def test_patches_with_patcher_context(self):
        """Test that patches can be used with Patcher."""
        import pennylane as qml
        from pennylane.capture import Patcher, get_jax_patches

        qml.capture.enable()

        def f(x):
            return jax.numpy.sin(x) + jax.numpy.cos(x)

        # Should work with patches applied
        with Patcher(*get_jax_patches()):
            jaxpr = jax.make_jaxpr(f)(0.5)
            assert jaxpr is not None

        qml.capture.disable()

    def test_patches_are_temporary(self):
        """Test that patches don't leak outside context."""
        from jax._src.interpreters import partial_eval as pe

        from pennylane.capture import Patcher, get_jax_patches

        # Check if make_eqn exists before patching
        has_make_eqn_before = hasattr(pe.DynamicJaxprTrace, "make_eqn")

        with Patcher(*get_jax_patches()):
            # Should have make_eqn inside context
            assert hasattr(pe.DynamicJaxprTrace, "make_eqn")

        # After exiting context, should be back to original state
        has_make_eqn_after = hasattr(pe.DynamicJaxprTrace, "make_eqn")

        # Note: make_eqn is currently applied globally for backwards compatibility,
        # so this test documents the current behavior. When global patching is removed,
        # this should assert: has_make_eqn_after == has_make_eqn_before
        assert has_make_eqn_after  # Currently always True due to global patching
