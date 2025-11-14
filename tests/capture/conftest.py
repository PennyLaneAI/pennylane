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
Pytest configuration for capture tests.

This module provides fixtures for dynamic shape tests that need JAX patches.
"""
import pytest


@pytest.fixture(scope="function")
def apply_patches_to_dynamic_shape_tests():
    """Apply JAX patches for dynamic shape tests using Patcher context manager.

    This fixture applies patches locally to tests that need dynamic shape support.
    The patches are applied at the beginning of each test and properly removed at the end
    using the Patcher context manager.

    Usage:
        @pytest.mark.usefixtures("enable_disable_dynamic_shapes", "apply_patches_to_dynamic_shape_tests")
        def test_something_with_dynamic_shapes():
            ...
    """
    jax = pytest.importorskip("jax")

    from packaging.version import Version

    if Version(jax.__version__) >= Version("0.7.0"):
        from pennylane.capture.jax_patches import get_jax_patches
        from pennylane.capture.patching import Patcher

        # Apply patches using Patcher context manager for this test
        patches = get_jax_patches()
        patcher = Patcher(*patches)
        patcher.__enter__()

        try:
            yield
        finally:
            # Properly cleanup patches after test
            patcher.__exit__(None, None, None)
    else:
        yield
