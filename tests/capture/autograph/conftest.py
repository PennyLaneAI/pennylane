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
"""
Pytest configuration file for AutoGraph test folder.
"""
import warnings

import pytest

from pennylane.exceptions import AutoGraphWarning


# pylint: disable=unused-import
# This is intended to suppress the *expected* warnings that arise when
# testing AutoGraph transformation functions with a `QNode` (which by default
# has AutoGraph transformations applied to it due to the `autograph` argument).
@pytest.fixture(autouse=True)
def filter_expected_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=AutoGraphWarning,
            message=r"AutoGraph will not transform the function .* as it has already been transformed\.",
        )
        yield
