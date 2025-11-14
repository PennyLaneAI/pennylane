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
"""Unit test module for the mlir_specs function in the Python Compiler inspection module."""


import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")
pytest.importorskip("jax")

# pylint: disable=wrong-import-position
import jax.numpy as jnp

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.inspection import ResourcesResult, mlir_specs

# pylint: disable=implicit-str-concat, unnecessary-lambda


def resources_equal(
    a: ResourcesResult, b: ResourcesResult, raise_on_mismatch: bool = False
) -> bool:
    if not (
        # a.device_name == b.device_name TODO: Don't worry about this one for now
        a.num_wires == b.num_wires
        and len(a.quantum_operations) == len(b.quantum_operations)
        and len(a.quantum_measurements) == len(b.quantum_measurements)
        and len(a.ppm_operations) == len(b.ppm_operations)
        and len(a.resource_sizes) == len(b.resource_sizes)
    ):
        if raise_on_mismatch:
            raise AssertionError("ResourceResult metadata mismatch")
        return False

    for name, count in a.quantum_operations.items():
        if name not in b.quantum_operations or b.quantum_operations[name] != count:
            if raise_on_mismatch:
                raise AssertionError(
                    f"Quantum operation count mismatch for {name}: {count} != {b.quantum_operations.get(name)}"
                )
            return False

    for name, count in a.quantum_measurements.items():
        if name not in b.quantum_measurements or b.quantum_measurements[name] != count:
            if raise_on_mismatch:
                raise AssertionError(
                    f"Quantum measurement count mismatch for {name}: {count} != {b.quantum_measurements.get(name)}"
                )
            return False

    for name, count in a.ppm_operations.items():
        if name not in b.ppm_operations or b.ppm_operations[name] != count:
            if raise_on_mismatch:
                raise AssertionError(
                    f"PPM operation count mismatch for {name}: {count} != {b.ppm_operations.get(name)}"
                )
            return False

    for size, count in a.resource_sizes.items():
        if size not in b.resource_sizes or b.resource_sizes[size] != count:
            if raise_on_mismatch:
                raise AssertionError(
                    f"Resource size count mismatch for {size}: {count} != {b.resource_sizes.get(size)}"
                )
            return False

    return True


def make_static_resources(
    quantum_operations: dict[str, int] | None = None,
    quantum_measurements: dict[str, int] | None = None,
    ppm_operations: dict[str, int] | None = None,
    resource_sizes: dict[int, int] | None = None,
    device_name: str | None = None,
    num_wires: int = 0,
) -> ResourcesResult:
    res = ResourcesResult()
    res.quantum_operations = quantum_operations or {}
    res.quantum_measurements = quantum_measurements or {}
    res.ppm_operations = ppm_operations or {}
    res.resource_sizes = resource_sizes or {}
    res.device_name = device_name
    res.num_wires = num_wires
    return res


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMLIRSpecs:
    """Unit tests for the mlir_specs function in the Python Compiler inspection module."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            qml.RX(1, 0)
            qml.RX(2.0, 0)
            qml.RZ(3.0, 1)
            qml.RZ(4.0, 1)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([0, 1])
            return qml.probs()

        return circ

    @pytest.mark.parametrize("level", [3.14, "invalid"])
    def test_invalid_level_type(self, simple_circuit, level):
        """Test that requesting an invalid level type raises an error."""

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)

        with pytest.raises(
            ValueError, match="The `level` argument must be an int, a tuple/list of ints, or 'all'."
        ):
            mlir_specs(simple_circuit, level=level)

    @pytest.mark.parametrize("level", [10, -1])
    def test_invalid_int_level(self, simple_circuit, level):
        """Test that requesting an invalid level raises an error."""

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)

        with pytest.raises(
            ValueError, match=f"Requested specs level {level} not found in MLIR pass list."
        ):
            mlir_specs(simple_circuit, level=level)

    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                make_static_resources(
                    quantum_operations={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 6, 2: 2},
                    num_wires=2,
                ),
            ),
        ],
    )
    def test_no_passes(self, simple_circuit, level, expected):
        """Test that if no passes are applied, the circuit resources are the original amount."""

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected, raise_on_mismatch=True)

    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                make_static_resources(
                    quantum_operations={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 6, 2: 2},
                    num_wires=2,
                ),
            ),
            (
                1,
                make_static_resources(
                    quantum_operations={"RX": 2, "RZ": 2},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 4},
                    num_wires=2,
                ),
            ),
            (
                2,
                make_static_resources(
                    quantum_operations={"RX": 1, "RZ": 1},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 2},
                    num_wires=2,
                ),
            ),
        ],
    )
    def test_basic_passes(self, simple_circuit, level, expected):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected, raise_on_mismatch=True)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters, autograph",
        [
            (True, 5, False),
            (False, 2, False),
            (True, 3, False),
            (False, 10, True),
        ],
    )
    def test_fixed_loop(self, pl_ctrl_flow, iters, autograph):
        """Test that loop resources are counted correctly."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ():
                @qml.for_loop(iters)
                def loop_body(i):
                    qml.X(i % 2)

                loop_body()

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ():
                for i in range(iters):
                    qml.X(i % 2)
                return qml.state()

        expected = make_static_resources(
            quantum_operations={"PauliX": iters},
            quantum_measurements={"state": 1},
            resource_sizes={1: iters},
            num_wires=2,
        )

        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=autograph)(circ)
        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected, raise_on_mismatch=True)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters",
        [
            (True, 5),
            (False, 2),
        ],
    )
    def test_dynamic_for_loop(self, pl_ctrl_flow, iters):
        """Test that dynamic for loops emit a warning."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                @qml.for_loop(n)
                def loop_body(i):
                    qml.X(i % 2)

                loop_body()

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                for i in range(n):
                    qml.X(i % 2)
                return qml.state()

        expected = make_static_resources(
            quantum_operations={"PauliX": 1},
            quantum_measurements={"state": 1},
            resource_sizes={1: 1},
            num_wires=2,
        )

        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected, raise_on_mismatch=True)


# TODO: May want to separate some of these concerns into testing specs_collect directly

if __name__ == "__main__":
    pytest.main(["-x", __file__])
