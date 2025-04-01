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
import shutil

import numpy as np
import pytest
from packaging.version import Version

import pennylane as qml

__all__ = ["xfail_on_numpy2"]

# NumPy 2.0 compatibility detection
NUMPY_VERSION = Version(np.__version__)
IS_NUMPY_2 = NUMPY_VERSION >= Version("2.0.0")
OPENFERMION_XFAIL_INFO = (
    "This test requires numpy 1.x. Double check OpenFermion compatibility with numpy 2.x."
)
xfail_on_numpy2 = pytest.mark.xfail(
    condition=IS_NUMPY_2,
    reason=OPENFERMION_XFAIL_INFO,
)


def cmd_exists(cmd):
    """Returns True if a binary exists on
    the system path"""
    return shutil.which(cmd) is not None


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return {"rtol": 0, "atol": 1e-8}


@pytest.fixture(
    scope="module",
    params=[
        None,
        qml.wires.Wires(
            list("ab") + [-3, 42] + ["xyz", "23", "wireX"] + [f"w{i}" for i in range(20)]
        ),
        list(range(100, 120)),
        {13 - i: "abcdefghijklmn"[i] for i in range(14)},
    ],
)
def custom_wires(request):
    """Custom wire mapping for Pennylane<->OpenFermion conversion"""
    return request.param


@pytest.fixture(scope="session", name="openfermion_support")
def fixture_openfermion_support():
    """Fixture to determine whether openfermion and openfermionpyscf are installed."""
    # pylint: disable=unused-import
    try:
        import openfermion
        import openfermionpyscf

        openfermion_support = True
    except ModuleNotFoundError:
        openfermion_support = False

    return openfermion_support


@pytest.fixture()
def skip_if_no_openfermion_support(openfermion_support):
    """Fixture to skip a test if openfermion or openfermionpyscf are not installed."""
    if not openfermion_support:
        pytest.skip("Skipped, no openfermion(pyscf) support")
