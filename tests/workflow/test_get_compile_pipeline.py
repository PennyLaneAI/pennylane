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
"""Contains tests for 'get_compile_pipeline'."""

import re

import pytest

import pennylane as qml
from pennylane.workflow import get_compile_pipeline


class TestValidation:
    """Tests for validation errors."""

    @pytest.mark.parametrize(
        "unsupported_level",
        (
            [0],
            [0, 1],
            (0,),
            (0, 1),
        ),
    )
    def test_incorrect_level_type(self, unsupported_level):
        """Tests validation for incorrect level types."""

        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"'level={unsupported_level}' of type '{type(unsupported_level)}' is not supported."
            ),
        ):
            _ = get_compile_pipeline(circuit, level=unsupported_level)()


class TestUserLevel:
    """Tests 'user' level transforms."""


class TestGradientLevel:
    """Tests 'device' level transforms."""


class TestDeviceLevel:
    """Tests 'device' level transforms."""
