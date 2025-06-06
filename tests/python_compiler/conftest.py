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
"""Pytest configuration for tests for the pennylane.compiler.python_compiler submodule."""

import io

import pytest

filecheck_available = True

try:
    from filecheck.finput import FInput
    from filecheck.matcher import Matcher
    from filecheck.options import parse_argv_options
    from filecheck.parser import Parser, pattern_for_opts
except ImportError:
    filecheck_available = False


def _run_filecheck(program_str, xdsl_module):
    if not filecheck_available:
        return

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(xdsl_module)),
        Parser(opts, io.StringIO(program_str), *pattern_for_opts(opts)),
    )
    assert matcher.run() == 0


@pytest.fixture(scope="function")
def run_filecheck():
    if not filecheck_available:
        pytest.skip("Cannot run lit tests without filecheck.")

    yield _run_filecheck
