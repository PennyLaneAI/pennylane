# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Pytest configuration file for PennyLane quantum chemistry open fermion test suite.
"""
import pytest


@pytest.fixture(scope="session", name="pyscf_support")
def fixture_pyscf_support():
    """Fixture to determine whether pyscf is installed."""
    # pylint: disable=unused-import, import-outside-toplevel
    try:
        import pyscf

        pyscf_support = True
    except ModuleNotFoundError:
        pyscf_support = False

    return pyscf_support


@pytest.fixture()
def skip_if_no_pyscf_support(pyscf_support):
    """Fixture to skip a test if pyscf is not installed."""
    if not pyscf_support:
        pytest.skip("Skipped, pyscf support")
