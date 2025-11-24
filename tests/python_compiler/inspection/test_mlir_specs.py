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
import catalyst

# pylint: disable=wrong-import-position
import jax.numpy as jnp
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.inspection import ResourcesResult, mlir_specs

# pylint: disable=implicit-str-concat, unnecessary-lambda


def resources_equal(
    actual: ResourcesResult, expected: ResourcesResult, return_only: bool = False
) -> bool:
    try:

        # actual.device_name == expected.device_name TODO: Don't worry about this one for now
        assert actual.num_allocs == expected.num_allocs
        assert len(actual.quantum_operations) == len(expected.quantum_operations)
        assert len(actual.quantum_measurements) == len(expected.quantum_measurements)
        assert len(actual.qec_operations) == len(expected.qec_operations)
        assert len(actual.resource_sizes) == len(expected.resource_sizes)

        for name, count in expected.quantum_operations.items():
            assert name in actual.quantum_operations
            assert actual.quantum_operations[name] == count

        for name, count in expected.quantum_measurements.items():
            assert name in actual.quantum_measurements
            assert actual.quantum_measurements[name] == count

        for name, count in expected.qec_operations.items():
            assert name in actual.qec_operations
            assert actual.qec_operations[name] == count

        for size, count in expected.resource_sizes.items():
            assert size in actual.resource_sizes
            assert actual.resource_sizes[size] == count

        for name, count in expected.function_calls.items():
            assert name in actual.function_calls
            assert actual.function_calls[name] == count

    except AssertionError:
        if return_only:
            return False
        raise

    return True


def make_static_resources(
    quantum_operations: dict[str, int] | None = None,
    quantum_measurements: dict[str, int] | None = None,
    qec_operations: dict[str, int] | None = None,
    resource_sizes: dict[int, int] | None = None,
    function_calls: dict[str, int] | None = None,
    device_name: str | None = None,
    num_allocs: int = 0,
) -> ResourcesResult:
    res = ResourcesResult()
    res.quantum_operations = quantum_operations or {}
    res.quantum_measurements = quantum_measurements or {}
    res.qec_operations = qec_operations or {}
    res.resource_sizes = resource_sizes or {}
    res.function_calls = function_calls or {}
    res.device_name = device_name
    res.num_allocs = num_allocs
    return res


class TestMLIRSpecs:
    """Unit tests for the mlir_specs function in the Python Compiler inspection module."""

    use_plxpr = False

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
                    num_allocs=2,
                ),
            ),
        ],
    )
    def test_no_passes(self, simple_circuit, level, expected):
        """Test that if no passes are applied, the circuit resources are the original amount."""

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                make_static_resources(
                    quantum_operations={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 6, 2: 2},
                    num_allocs=2,
                ),
            ),
            (
                1,
                make_static_resources(
                    quantum_operations={"RX": 2, "RZ": 2},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 4},
                    num_allocs=2,
                ),
            ),
            (
                2,
                make_static_resources(
                    quantum_operations={"RX": 1, "RZ": 1},
                    quantum_measurements={"probs(0 wires)": 1},
                    resource_sizes={1: 2},
                    num_allocs=2,
                ),
            ),
        ],
    )
    def test_basic_passes(self, simple_circuit, level, expected):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if self.use_plxpr:
            simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
            simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        else:
            simple_circuit = catalyst.passes.cancel_inverses(simple_circuit)
            simple_circuit = catalyst.passes.merge_rotations(simple_circuit)

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)
        res = mlir_specs(simple_circuit, level=level)
        assert resources_equal(res, expected)

    def test_basic_passes_level_all(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if self.use_plxpr:
            simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
            simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        else:
            simple_circuit = catalyst.passes.cancel_inverses(simple_circuit)
            simple_circuit = catalyst.passes.merge_rotations(simple_circuit)

        expected = {
            "Before MLIR Passes (MLIR-0)": make_static_resources(
                quantum_operations={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                quantum_measurements={"probs(0 wires)": 1},
                resource_sizes={1: 6, 2: 2},
                num_allocs=2,
            ),
            "cancel-inverses (MLIR-1)": make_static_resources(
                quantum_operations={"RX": 2, "RZ": 2},
                quantum_measurements={"probs(0 wires)": 1},
                resource_sizes={1: 4},
                num_allocs=2,
            ),
            "merge-rotations (MLIR-2)": make_static_resources(
                quantum_operations={"RX": 1, "RZ": 1},
                quantum_measurements={"probs(0 wires)": 1},
                resource_sizes={1: 2},
                num_allocs=2,
            ),
        }

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)
        res = mlir_specs(simple_circuit, level="all")

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert resources_equal(res[lvl], expected_res)

    def test_basic_passes_multi_level(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        if self.use_plxpr:
            simple_circuit = qml.transforms.cancel_inverses(simple_circuit)
            simple_circuit = qml.transforms.merge_rotations(simple_circuit)
        else:
            simple_circuit = catalyst.passes.cancel_inverses(simple_circuit)
            simple_circuit = catalyst.passes.merge_rotations(simple_circuit)

        expected = {
            "Before MLIR Passes (MLIR-0)": make_static_resources(
                quantum_operations={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                quantum_measurements={"probs(0 wires)": 1},
                resource_sizes={1: 6, 2: 2},
                num_allocs=2,
            ),
            "merge-rotations (MLIR-2)": make_static_resources(
                quantum_operations={"RX": 1, "RZ": 1},
                quantum_measurements={"probs(0 wires)": 1},
                resource_sizes={1: 2},
                num_allocs=2,
            ),
        }

        simple_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(simple_circuit)
        res = mlir_specs(simple_circuit, level=[0, 2])

        assert isinstance(res, dict)
        assert len(res) == len(expected)

        for lvl, expected_res in expected.items():
            assert lvl in res.keys()
            assert resources_equal(res[lvl], expected_res)

        with pytest.raises(
            ValueError, match="Requested specs levels 3 not found in MLIR pass list."
        ):
            mlir_specs(simple_circuit, level=[0, 3])

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
            quantum_measurements={"state(0 wires)": 1},
            resource_sizes={1: iters},
            num_allocs=2,
        )

        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=autograph)(circ)
        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)

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
            quantum_measurements={"state(0 wires)": 1},
            resource_sizes={1: 1},
            num_allocs=2,
        )

        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow, iters",
        [
            (True, 5),
            (False, 2),
        ],
    )
    def test_while_loop(self, pl_ctrl_flow, iters):
        """Test that dynamic while loops emit a warning."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                def loop_cond(i):
                    return i < n

                @qml.while_loop(loop_cond)
                def loop_body(i):
                    qml.X(i % 2)
                    return i + 1

                loop_body(0)

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                i = 0
                while i < n:
                    qml.X(i % 2)
                    i += 1
                return qml.state()

        expected = make_static_resources(
            quantum_operations={"PauliX": 1},
            quantum_measurements={"state(0 wires)": 1},
            resource_sizes={1: 1},
            num_allocs=2,
        )

        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the number of loop iterations. "
            "The results will assume the loop runs only once. "
            "This may be fixed in some cases by inlining dynamic arguments.",
        ):
            res = mlir_specs(circ, 0, iters)
            assert resources_equal(res, expected)

    @pytest.mark.parametrize(
        "pl_ctrl_flow",
        [
            (True),
            (False),
        ],
    )
    def test_cond(self, pl_ctrl_flow):
        """Test that conditions emit a warning."""

        if pl_ctrl_flow:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                qml.cond(n > 0, qml.X, qml.Z)(0)

                return qml.state()

        else:

            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def circ(n):
                if n > 0:
                    qml.X(0)
                else:
                    qml.Z(0)
                return qml.state()

        expected = make_static_resources(
            quantum_operations={"PauliX": 1, "PauliZ": 1},
            quantum_measurements={"state(0 wires)": 1},
            resource_sizes={1: 2},
            num_allocs=2,
        )

        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)(circ)

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the branch of a conditional or switch statement. "
            "The results will take the maximum resources across all possible branches.",
        ):
            n = 3  # Arbitrary value for n
            res = mlir_specs(circ, 0, n)
            assert resources_equal(res, expected)

    def test_tape_transforms(self):
        """Test that tape transforms are handled correctly."""
        if qml.capture.enabled():
            pytest.xfail("Currently broken with plxpr enabled.")

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circ():
            qml.GlobalPhase(0.5)
            qml.GlobalPhase(1.0)
            return qml.expval(qml.PauliZ(0))

        circ = qml.transforms.combine_global_phases(circ)
        circ = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(circ)

        expected = make_static_resources(
            quantum_operations={"GlobalPhase": 1},
            quantum_measurements={"expval(PauliZ)": 1},
            resource_sizes={0: 1},
            num_allocs=2,
        )

        res = mlir_specs(circ, level=0)
        assert resources_equal(res, expected)


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMLIRSpecsWithPLXPR(TestMLIRSpecs):
    """Unit tests for the mlir_specs function in the Python Compiler inspection module with plxpr enabled."""

    use_plxpr = True


# TODO: May want to separate some of these concerns into testing specs_collect directly

if __name__ == "__main__":
    pytest.main(["-x", __file__])
