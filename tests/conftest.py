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
Pytest configuration file for PennyLane test suite.
"""
# pylint: disable=unused-import
import os
import pathlib
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import DefaultGaussian

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))

# defaults
TOL = 1e-3
TF_TOL = 2e-2
TOL_STOCHASTIC = 0.05


# pylint: disable=too-few-public-methods
class DummyDevice(DefaultGaussian):
    """Dummy device to allow Kerr operations"""

    _operation_map = DefaultGaussian._operation_map.copy()
    _operation_map["Kerr"] = lambda *x, **y: np.identity(2)


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def tol_stochastic():
    """Numerical tolerance for equality tests of stochastic values."""
    return TOL_STOCHASTIC


@pytest.fixture(scope="session")
def tf_tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TF_TOL", TF_TOL))


@pytest.fixture(scope="session", params=[1, 2])
def n_layers(request):
    """Number of layers."""
    return request.param


@pytest.fixture(scope="session", params=[2, 3], name="n_subsystems")
def n_subsystems_fixture(request):
    """Number of qubits or qumodes."""
    return request.param


@pytest.fixture(scope="session")
def qubit_device(n_subsystems):
    return qml.device("default.qubit", wires=n_subsystems)


# The following 3 fixtures are for default.qutrit devices to be used
# for testing with various real and complex dtypes.


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qutrit_device_1_wire(request):
    return qml.device("default.qutrit", wires=1, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qutrit_device_2_wires(request):
    return qml.device("default.qutrit", wires=2, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qutrit_device_3_wires(request):
    return qml.device("default.qutrit", wires=3, r_dtype=request.param[0], c_dtype=request.param[1])


#######################################################################


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    """A mock instance of the abstract Device class"""

    with monkeypatch.context() as m:
        dev = qml.devices.LegacyDevice
        m.setattr(dev, "__abstractmethods__", frozenset())
        m.setattr(dev, "short_name", "mock_device")
        m.setattr(dev, "capabilities", lambda cls: {"model": "qubit"})
        m.setattr(dev, "operations", {"RX", "RY", "RZ", "CNOT", "SWAP"})
        yield qml.devices.LegacyDevice(wires=2)  # pylint:disable=abstract-class-instantiated


# pylint: disable=protected-access
@pytest.fixture
def tear_down_hermitian():
    yield None
    qml.Hermitian._eigs = {}


# pylint: disable=protected-access
@pytest.fixture
def tear_down_thermitian():
    yield None
    qml.THermitian._eigs = {}


@pytest.fixture(autouse=True)
def restore_global_seed():
    original_state = np.random.get_state()
    yield
    np.random.set_state(original_state)


@pytest.fixture
def seed(request):
    """An integer random number generator seed

    This fixture overrides the ``seed`` fixture provided by pytest-rng, adding the flexibility
    of locally getting a new seed for a test case by applying the ``local_salt`` marker. This is
    useful when the seed from pytest-rng happens to be a bad seed that causes your test to fail.

    .. code_block:: python

        @pytest.mark.local_salt(42)
        def test_something(seed):
            ...

    The value passed to ``local_salt`` needs to be an integer.

    """

    fixture_manager = request._fixturemanager  # pylint:disable=protected-access
    fixture_defs = fixture_manager.getfixturedefs("seed", request.node)
    original_fixture_def = fixture_defs[0]  # the original seed fixture provided by pytest-rng
    original_seed = original_fixture_def.func(request)
    marker = request.node.get_closest_marker("local_salt")
    if marker and marker.args:
        return original_seed + marker.args[0]
    return original_seed


@pytest.fixture(scope="function")
def enable_disable_plxpr():
    """enable and disable capture around each test."""
    qml.capture.enable()
    try:
        yield
    finally:
        qml.capture.disable()


@pytest.fixture(scope="function")
def enable_disable_dynamic_shapes():
    jax.config.update("jax_dynamic_shapes", True)
    try:
        yield
    finally:
        jax.config.update("jax_dynamic_shapes", False)


@pytest.fixture(scope="function")
def enable_graph_decomposition():
    """enable and disable graph-decomposition around each test."""
    qml.decomposition.enable_graph()
    try:
        yield
    finally:
        qml.decomposition.disable_graph()


#######################################################################

try:
    import torch
    from torch.autograd import Variable

    torch_available = True
except ImportError as e:
    torch_available = False

try:
    import jax
    import jax.numpy as jnp

    jax_available = True
except ImportError as e:
    jax_available = False


# pylint: disable=unused-argument
def pytest_generate_tests(metafunc):
    if jax_available:
        jax.config.update("jax_enable_x64", True)


@pytest.fixture(
    params=[
        pytest.param("autograd", marks=pytest.mark.autograd),
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("jax-jit", marks=pytest.mark.jax),
        pytest.param("torch", marks=pytest.mark.torch),
    ],
    scope="function",
)
def interface(request):
    """Automatically parametrize over all interfaces."""
    yield request.param


def pytest_collection_modifyitems(items, config):
    rootdir = pathlib.Path(config.rootdir)
    for item in items:
        rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
        if "qchem" in rel_path.parts:
            mark = getattr(pytest.mark, "qchem")
            item.add_marker(mark)
        if "finite_diff" in rel_path.parts:
            mark = getattr(pytest.mark, "finite-diff")
            item.add_marker(mark)
        if "parameter_shift" in rel_path.parts:
            mark = getattr(pytest.mark, "param-shift")
            item.add_marker(mark)
        if "data" in rel_path.parts:
            mark = getattr(pytest.mark, "data")
            item.add_marker(mark)

    # Tests that do not have a specific suite marker are marked `core`
    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        if (
            not any(
                elem
                in [
                    "autograd",
                    "data",
                    "torch",
                    "jax",
                    "qchem",
                    "qcut",
                    "all_interfaces",
                    "finite-diff",
                    "param-shift",
                    "external",
                    "capture",
                ]
                for elem in markers
            )
            or not markers
        ):
            item.add_marker(pytest.mark.core)
        if "capture" in markers:
            item.fixturenames.append("enable_disable_plxpr")
            if "jax" not in markers:
                item.add_marker(pytest.mark.jax)


def pytest_runtest_setup(item):
    """Automatically skip tests if interfaces are not installed"""
    # Autograd is assumed to be installed
    interfaces = {"torch", "jax"}
    available_interfaces = {
        "torch": torch_available,
        "jax": jax_available,
    }

    allowed_interfaces = [
        allowed_interface
        for allowed_interface in interfaces
        if available_interfaces[allowed_interface] is True
    ]

    # load the marker specifying what the interface is
    all_interfaces = {"tf", "torch", "jax", "all_interfaces"}
    marks = {mark.name for mark in item.iter_markers() if mark.name in all_interfaces}

    for b in marks:
        if b == "all_interfaces":
            required_interfaces = {"torch", "jax"}
            for _interface in required_interfaces:
                if _interface not in allowed_interfaces:
                    pytest.skip(
                        f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                    )
        else:
            if b not in allowed_interfaces:
                pytest.skip(
                    f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                )


@pytest.fixture(params=[False, True], ids=["graph_disabled", "graph_enabled"])
def enable_and_disable_graph_decomp(request):
    """
    A fixture that parametrizes a test to run twice: once with graph
    decomposition disabled and once with it enabled.

    It automatically handles the setup (enabling/disabling) before the
    test runs and the teardown (always disabling) after the test completes.
    """
    try:
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

    finally:
        # --- Teardown Phase ---
        # This code runs after the test function has finished,
        # regardless of whether it passed or failed.
        qml.decomposition.disable_graph()
