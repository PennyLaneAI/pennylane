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

This module applies JAX patches globally for all capture tests.
The patches are applied at module import time to ensure they are active
for the entire test session.
"""

# Apply patches at module import time
try:
    import jax
    from packaging.version import Version

    if Version(jax.__version__) >= Version("0.7.0"):
        from pennylane.capture.jax_patches import apply_patches_globally

        # Apply patches globally for tests
        apply_patches_globally()
except ImportError:
    # JAX not available, skip patching
    pass
