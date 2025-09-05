# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Pytest configuration file for the devices test module.
"""

from os import path
from tempfile import TemporaryDirectory
from textwrap import dedent

import pytest

import pennylane as qml


@pytest.fixture(scope="function")
def create_temporary_toml_file(request) -> str:
    """Create a temporary TOML file with the given content."""
    content = request.param
    with TemporaryDirectory() as temp_dir:
        toml_file = path.join(temp_dir, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(dedent(content))
        request.node.toml_file = toml_file
        yield


@pytest.fixture(params=[False, True], ids=["graph_disabled", "graph_enabled"])
def enable_and_disable_graph_decomp(request):
    """
    A fixture that parametrizes a test to run twice: once with graph
    decomposition disabled and once with it enabled.

    It automatically handles the setup (enabling/disabling) before the
    test runs and the teardown (always disabling) after the test completes.
    """
    use_graph_decomp = request.param

    # --- Setup Phase ---
    # This code runs before the test function is executed.
    if use_graph_decomp:
        qml.decomposition.enable_graph()
    else:
        # Explicitly disable to ensure a clean state
        qml.decomposition.disable_graph()

    # Yield control to the test function
    yield use_graph_decomp

    # --- Teardown Phase ---
    # This code runs after the test function has finished,
    # regardless of whether it passed or failed.
    qml.decomposition.disable_graph()
