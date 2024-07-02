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
Tests capture module imports and access.
"""
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """enable and disable capture around each test."""
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_no_attribute_available():
    """Test that if we try and access an attribute that doesn't exist, we get an attribute error."""

    with pytest.raises(AttributeError):
        _ = qml.capture.something


def test_default_use():

    @qml.capture.bind_nested_plxpr
    def repeat(qfunc, start=0, stop=4):
        def new_qfunc(*args, **kwargs):
            for _ in range(start, stop):
                qfunc(*args, **kwargs)

        return new_qfunc

    #
