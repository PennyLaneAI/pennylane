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
"""Unit tests for the estimate transform"""

import copy
import importlib
import warnings
from functools import partial

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.shots import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst

TARGET = {"Hadamard", "CNOT", "RX"}


@qp.register_resources({qp.Z: 2, qp.RY: 2, qp.CNOT: 1}, name="cz_with_ry")
def cz_with_ry(wires, **_):
    """Decomposition rule for CZ that writes each Hadamard as Z followed by RY(pi/2)."""
    qp.Z(wires[1])
    qp.RY(np.pi / 2, wires[1])
    qp.CNOT(wires)
    qp.Z(wires[1])
    qp.RY(np.pi / 2, wires[1])


# A pass that is never run, since estimate skips it, so it does not need to exist in Catalyst.
custom_transform = qp.transform(pass_name="custom-pass")


@qp.register_resources({qp.RY: 1, qp.X: 1}, name="h_with_x")
def h_with_x(wires, **_):
    """Decomposition rule for Hadamard as RY(pi/2) followed by X."""
    qp.RY(np.pi / 2, wires)
    qp.X(wires)


@pytest.mark.capture
class TestEstimate:  # pylint: disable=too-many-public-methods
    """Test qp.estimate()"""

    @pytest.fixture
    def qnode(self):
        """Fixture for a QNode with gates that cancel and a gate outside the target gate set."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CZ([0, 1])
            qp.RX(x, wires=1)
            return qp.probs()

        return circuit

    @pytest.fixture
    def circuit(self, qnode):
        """Fixture for the qjit'd QNode with a user transform."""
        return qp.qjit(qp.transforms.cancel_inverses(qnode))

    @pytest.fixture
    def dev_without_cz(self):
        """Fixture for a device that does not support CZ when compiled with Catalyst."""
        dev = qp.device("lightning.qubit", wires=2)
        capabilities = copy.deepcopy(dev.capabilities)
        del capabilities.operations["CZ"]
        dev.qjit_capabilities = capabilities
        return dev

    def test_decomposition_clifford_t(self):
        """Test the resources of a circuit with many Clifford and T-based operators and several
        measurements projected onto Clifford+T, and compare with a real decompose call."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.Toffoli([0, 1, 2])
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CCZ([0, 1, 2])
            qp.SWAP([0, 1])
            qp.CZ([1, 2])
            qp.CY([0, 2])
            qp.adjoint(qp.T(0))
            qp.adjoint(qp.S(1))
            qp.S(2)
            qp.T(2)
            qp.CNOT([0, 1])
            qp.PauliX(1)
            return qp.expval(qp.Z(0)), qp.probs(wires=[0, 1]), qp.var(qp.X(2))

        resources = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources

        assert resources == SpecsResources(
            counts={
                "Adjoint(S)": 1,
                "Adjoint(T)": 7,
                "CNOT": 13,
                "CY": 1,
                "CZ": 1,
                "Hadamard": 6,
                "PauliX": 1,
                "S": 1,
                "SWAP": 1,
                "T": 9,
            },
            measurement_processes={"expval(PauliZ)": 1, "probs(2 wires)": 1, "var(PauliX)": 1},
            num_wires=3,
        )

        @qp.qjit(capture=True)
        @qp.transforms.decompose(gate_set=qp.gate_sets.CLIFFORD_T)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def decomposed_circuit():
            qp.Toffoli([0, 1, 2])
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CCZ([0, 1, 2])
            qp.SWAP([0, 1])
            qp.CZ([1, 2])
            qp.CY([0, 2])
            qp.adjoint(qp.T(0))
            qp.adjoint(qp.S(1))
            qp.S(2)
            qp.T(2)
            qp.CNOT([0, 1])
            qp.PauliX(1)
            return qp.expval(qp.Z(0)), qp.probs(wires=[0, 1]), qp.var(qp.X(2))

        assert qp.analyze(decomposed_circuit, level=1)().resources == resources

    def test_decomposition_rotations(self):
        """Test estimate of ctrl(RX) into ROTATIONS_PLUS_CNOT against a real decompose call.

        estimate counts average-case X flips for zero-valued controls; analyze takes the max
        over scf.if branches, so counts may differ by one RX and one GlobalPhase.
        """
        target = qp.gate_sets.ROTATIONS_PLUS_CNOT

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.ctrl(qp.RX(0.3, 1), control=[0])
            return qp.state()

        resources = qp.estimate(circuit, target=target)().resources
        assert resources.counts == pytest.approx(
            {"CNOT": 2, "GlobalPhase": 1, "RX": 1, "RY": 2, "RZ": 2}, abs=1
        )
        assert resources.num_wires == 2

        @qp.qjit(capture=True)
        @qp.transforms.decompose(gate_set=target)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def decomposed_circuit():
            qp.ctrl(qp.RX(0.3, 1), control=[0])
            return qp.state()

        decomposed = qp.analyze(decomposed_circuit, level=1)().resources
        assert resources.counts == pytest.approx(decomposed.counts, abs=1)

    def test_specs(self, circuit):
        """Test that estimate ignores the user transforms and decomposes the circuit into the
        target gate set."""

        specs = qp.estimate(circuit, target=TARGET)(0.5)

        assert specs == CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level="top",
            resources=SpecsResources(
                counts={"Hadamard": 4, "CNOT": 1, "RX": 1},
                measurement_processes={"probs(all wires)": 1},
                num_wires=2,
            ),
        )

    @pytest.mark.parametrize("num_wires", [1, 3])
    @pytest.mark.parametrize("measurement", ["probs", "sample", "counts", "expval"])
    def test_abstract_wire_measurements(self, num_wires, measurement):
        """Estimate supports terminal measurements on abstract registers."""
        wires = qp.typing.Wire[num_wires]

        @qp.qjit(capture=True)
        @qp.set_shots(50 if measurement in ("sample", "counts") else None)
        @qp.qnode(qp.device("null.qubit", wires=num_wires))
        def circuit():
            qp.X(wires=0)
            if measurement == "expval":
                return qp.expval(qp.Z(wires=wires[0]))
            if measurement == "counts":
                return qp.counts(wires=wires, all_outcomes=True)
            return getattr(qp, measurement)(wires=wires)

        specs = qp.estimate(circuit, target={"PauliX": 1})()

        assert specs.resources.quantum_operations == {"PauliX": 1}
        measured = "PauliZ" if measurement == "expval" else f"{num_wires} wires"
        assert specs.resources.measurement_processes == {f"{measurement}({measured})": 1}

    @pytest.mark.parametrize(
        "target",
        [
            ["Hadamard", "CNOT", "RX"],
            {qp.Hadamard, qp.CNOT, qp.RX},
            {qp.Hadamard: 1, "CNOT": 2.0, qp.RX: 1},
        ],
    )
    def test_target_formats(self, circuit, target):
        """Test that estimate accepts the gate set formats supported by qp.decompose."""

        specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    def test_predefined_gate_set(self, circuit):
        """Test that estimate accepts a predefined gate set as target."""

        specs = qp.estimate(circuit, target=qp.gate_sets.ROTATIONS_PLUS_CNOT)(0.5)

        assert set(specs.resources.quantum_operations) <= {"RX", "RY", "RZ", "CNOT", "GlobalPhase"}

    def test_operations_in_target(self, circuit):
        """Test that operations already in the target gate set are kept."""

        specs = qp.estimate(circuit, target={"Hadamard", "CZ", "RX"})(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CZ": 1, "RX": 1}

    def test_fixed_decomps(self, qnode):
        """Test that the fixed_decomps of a decompose transform in the pipeline replace the
        existing decomposition rules if the target is a pipeline level."""

        qp.add_decomps(qp.CZ, cz_with_ry)
        target = TARGET | {"RY", "PauliZ"}
        with qp.decomposition.local_decomps():
            qp.add_decomps(qp.CZ, cz_with_ry)
            circuit = qp.transforms.decompose(
                qnode, gate_set=target, fixed_decomps={qp.CZ: cz_with_ry}
            )
            specs = qp.estimate(qp.qjit(circuit), target="user")(0.5)

        assert specs.resources.quantum_operations == {
            "Hadamard": 2,
            "PauliZ": 2,
            "RY": 2,
            "CNOT": 1,
            "RX": 1,
        }

    @pytest.mark.parametrize("use_names", [False, True])
    def test_exclude_decomps(self, qnode, use_names):
        """Estimate respects exclusions from a decompose transform in the pipeline if the target
        is a pipeline level."""

        target = TARGET | {"RY", "PauliZ"}
        excluded = list(qp.list_decomps(qp.CZ))
        if use_names:
            excluded = [rule.name for rule in excluded]
        qp.add_decomps(qp.CZ, cz_with_ry)
        circuit = qp.decompose(qnode, gate_set=target, exclude_decomps={qp.CZ: excluded})
        specs = qp.estimate(qp.qjit(circuit), target="user")(0.5)

        assert specs.resources.quantum_operations == {
            "Hadamard": 2,
            "PauliZ": 2,
            "RY": 2,
            "CNOT": 1,
            "RX": 1,
        }

    def test_exclusions_from_first_decompose_transform(self, qnode):
        """Test that the excluded rules for an operator are those of the first decompose transform
        that decomposes it, as in the compiled pipeline."""
        target = TARGET | {"RY", "PauliZ"}
        excluded = [rule.name for rule in qp.list_decomps(qp.CZ)]
        qp.add_decomps(qp.CZ, cz_with_ry)
        circuit = qp.decompose(qnode, gate_set=target, exclude_decomps={qp.CZ: excluded})
        circuit = qp.decompose(circuit, gate_set=target, exclude_decomps={qp.CZ: ["cz_with_ry"]})
        circuit = qp.qjit(circuit)

        specs = qp.estimate(circuit, target="user")(0.5)

        assert specs.resources.quantum_operations == {
            "Hadamard": 2,
            "PauliZ": 2,
            "RY": 2,
            "CNOT": 1,
            "RX": 1,
        }
        expected = qp.analyze(circuit, level="user")(0.5).resources.quantum_operations
        assert specs.resources.quantum_operations == expected

    @pytest.mark.parametrize(
        "target, counts",
        [
            (1, {"Hadamard": 2, "PauliZ": 2, "RY": 2, "CNOT": 1, "RX": 1}),
            (2, {"PauliX": 2, "PauliZ": 2, "RY": 4, "CNOT": 1, "RX": 1}),
            ("user", {"PauliX": 2, "PauliZ": 2, "RY": 4, "CNOT": 1, "RX": 1}),
        ],
    )
    def test_decomps_from_several_decompose_transforms(self, qnode, target, counts):
        """Test that a pipeline level as target uses the gate set of the last decompose transform
        before that level, and the rules of every decompose transform before that level."""

        qp.add_decomps(qp.CZ, cz_with_ry)
        qp.add_decomps(qp.Hadamard, h_with_x)
        gate_set = {"CNOT", "RX", "RY", "PauliX", "PauliZ"}
        with qp.decomposition.local_decomps():
            qp.add_decomps(qp.CZ, cz_with_ry)
            circuit = qp.transforms.decompose(
                qnode, gate_set=gate_set | {"Hadamard"}, fixed_decomps={qp.CZ: cz_with_ry}
            )
            qp.add_decomps(qp.Hadamard, h_with_x)
            circuit = qp.transforms.decompose(
                circuit, gate_set=gate_set, fixed_decomps={qp.Hadamard: h_with_x}
            )
            specs = qp.estimate(qp.qjit(circuit), target=target)(0.5)

        assert specs.resources.quantum_operations == counts

    @pytest.mark.parametrize("target", [1, 2])
    def test_rules_of_later_decompose_transforms(self, qnode, target):
        """Test that the rules of a decompose transform are not used for an operator that an
        earlier decompose transform decomposes, as in the compiled pipeline."""

        gate_set = TARGET | {"RY", "PauliZ"}
        with qp.decomposition.local_decomps():
            qp.add_decomps(qp.CZ, cz_with_ry)
            circuit = qp.transforms.decompose(qnode, gate_set=gate_set)
            circuit = qp.transforms.decompose(
                circuit, gate_set=gate_set, fixed_decomps={qp.CZ: cz_with_ry}
            )
            circuit = qp.qjit(circuit)
            specs = qp.estimate(circuit, target=target)(0.5)
            expected = qp.analyze(circuit, level=target)(0.5).resources.quantum_operations

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}
        assert specs.resources.quantum_operations == expected

    @pytest.mark.parametrize("rules", ["fixed_decomps", "exclude_decomps"])
    def test_gate_set_target_ignores_pipeline_rules(self, qnode, rules):
        """Test that the rules of the decompose transforms in the pipeline are ignored with a
        warning if the target is a gate set."""

        target = TARGET | {"RY", "PauliZ"}
        with qp.decomposition.local_decomps():
            qp.add_decomps(qp.CZ, cz_with_ry)
            rule_kwargs = (
                {"fixed_decomps": {qp.CZ: cz_with_ry}}
                if rules == "fixed_decomps"
                else {"exclude_decomps": {qp.CZ: ["cz_with_ry"]}}
            )
            circuit = qp.transforms.decompose(qnode, gate_set=target, **rule_kwargs)
            with pytest.warns(UserWarning, match="ignores the fixed_decomps and exclude_decomps"):
                specs = qp.estimate(qp.qjit(circuit), target=target)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    def test_gate_set_target_without_pipeline_rules(self, qnode):
        """Test that a gate set target does not warn if the decompose transforms in the pipeline
        have no rules."""

        circuit = qp.qjit(qp.transforms.decompose(qnode, gate_set=TARGET | {"CZ"}))

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            specs = qp.estimate(circuit, target=TARGET)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    def test_unreachable_target(self, circuit):
        """Test that a short error naming the operator is raised if the circuit cannot be
        decomposed into the target gate set."""

        with pytest.raises(ValueError, match="Decomposition rule not found for operator RX"):
            qp.estimate(circuit, target={"Hadamard", "CNOT"})(0.5)

    def test_single_gate_type_target(self, circuit):
        """Test that a single gate type is a gate set with one gate, as in qp.decompose, and that
        the error for an unreachable one names the operator."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def swap_circuit():
            qp.SWAP([0, 1])
            return qp.probs()

        specs = qp.estimate(swap_circuit, target=qp.CNOT)()
        assert specs.resources.quantum_operations == {"CNOT": 3}

        with pytest.raises(ValueError, match="Decomposition rule not found for operator .*: it"):
            qp.estimate(circuit, target=qp.CNOT)(0.5)

    def test_other_compile_errors(self, circuit, monkeypatch):
        """Test that compile errors unrelated to a missing decomposition rule are raised as they
        are."""

        def fail(*_, **__):
            raise catalyst.utils.exceptions.CompileError("some other compile error")

        estimate_module = importlib.import_module("pennylane.resource.estimate")
        monkeypatch.setattr(estimate_module, "resources_from_analysis_pass", fail)

        with pytest.raises(catalyst.utils.exceptions.CompileError, match="some other"):
            qp.estimate(circuit, target=TARGET)(0.5)

    def test_original_circuit_unchanged(self, circuit):
        """Test that estimate does not remove the user transforms from the original circuit."""

        qp.estimate(circuit, target=TARGET)(0.5)

        specs = qp.analyze(circuit, level="user")(0.5)

        assert specs.resources.quantum_operations == {"CZ": 1, "RX": 1}

    def test_symbolic_resources(self):
        """Test that estimate counts the resources inside a loop with a number of iterations that
        is not known at compile time symbolically."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(n):
            @qp.for_loop(n)
            def loop(_):
                qp.CZ([0, 1])

            loop()
            return qp.probs()

        resources = qp.estimate(circuit, target={"Hadamard", "CNOT"})(5).resources

        assert resources.is_symbolic
        assert len(resources.vars) == 1

        concrete = resources.subs({var: 5 for var in resources.vars})
        assert not concrete.is_symbolic
        assert concrete.counts == {"Hadamard": 10, "CNOT": 5}

    def test_abstract_data_hints_scf(self):
        """Estimate supports abstract inputs together with hinted SCF."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=10))
        def circuit(n, angles, wires):
            @qp.hint({"num-iters": 4})
            @qp.for_loop(n)
            def loop(i):
                qp.CZ([wires[0], wires[1]])
                qp.RX(angles[i], wires[0])

            loop()
            return qp.expval(qp.Z(wires[0]))

        resources = qp.estimate(circuit, target={"Hadamard", "CNOT", "RX"})(
            2, qp.typing.AbstractArray((5,), float), qp.typing.Wire[2]
        ).resources

        # hint overrides the runtime bound: 4 iters of CZ -> 2 H + CNOT, plus RX
        assert resources == SpecsResources(
            counts={"Hadamard": 8, "CNOT": 4, "RX": 4},
            measurement_processes={"expval(PauliZ)": 1},
            num_wires=10,
        )

    def test_partial(self, circuit):
        """Test that estimate supports functools.partial wrappers."""

        specs = qp.estimate(partial(circuit, 0.5), target=TARGET)()

        assert specs == qp.estimate(circuit, target=TARGET)(0.5)

    def test_global_capture(self):
        """Test that estimate supports capture="global" when program capture is enabled."""

        @qp.qjit(capture="global")
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.CZ([0, 1])
            return qp.probs()

        specs = qp.estimate(circuit, target={"Hadamard", "CNOT"})()

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CNOT": 1}

    def test_level_zero(self, circuit):
        """Test that level=0 is the same as level="top"."""

        specs = qp.estimate(circuit, level=0, target=TARGET)(0.5)

        assert specs == qp.estimate(circuit, level="top", target=TARGET)(0.5)

    def test_marker_levels(self, qnode):
        """Test that a marker before every pass is the same as level="top", and that a marker
        after a pass is not supported."""

        circuit = qp.marker(qnode, "start")
        circuit = qp.transforms.cancel_inverses(circuit)
        circuit = qp.qjit(qp.marker(circuit, "end"))

        specs = qp.estimate(circuit, level="start", target=TARGET)(0.5)
        assert specs == qp.estimate(circuit, level="top", target=TARGET)(0.5)

        with pytest.raises(NotImplementedError, match="qp.estimate only supports level='top'"):
            qp.estimate(circuit, level="end", target=TARGET)(0.5)

    @pytest.mark.parametrize("level", [slice(0, 0), slice(1, 1)])
    def test_levels_equivalent_to_top(self, circuit, level):
        """Test that a slice that selects no transforms is the same as level="top", with a
        warning."""

        with pytest.warns(UserWarning, match="does not select any transforms"):
            specs = qp.estimate(circuit, level=level, target=TARGET)(0.5)

        assert specs == qp.estimate(circuit, level="top", target=TARGET)(0.5)

    @pytest.mark.parametrize(
        "level",
        [
            1,
            "user",
            "all",
            [0],
            ("top",),
            ["device"],
            [1],
            ["user"],
            (0, 1),
            slice(0, 1),
            slice(1, 2),
            slice(0, None),
        ],
    )
    def test_unsupported_level(self, circuit, level):
        """Test that an error is raised for levels other than "top"."""

        with pytest.raises(NotImplementedError, match="qp.estimate only supports level='top'"):
            qp.estimate(circuit, level=level, target=TARGET)(0.5)

    def test_unsorted_sequence_does_not_warn(self, circuit):
        """Test that an unsupported unsorted Sequence of levels raises an error without warning
        about sorting it."""

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The 'level' argument has been sorted")
            with pytest.raises(NotImplementedError, match="qp.estimate only supports"):
                qp.estimate(circuit, level=[1, 0], target=TARGET)(0.5)

    @pytest.mark.parametrize("level", ["all-user", ["top", 0]])
    def test_multiple_levels_without_transforms(self, qnode, level):
        """Test that several levels are not supported, even if they all are level 0 because the
        QNode has no transforms."""

        with pytest.raises(NotImplementedError, match="qp.estimate only supports level='top'"):
            qp.estimate(qp.qjit(qnode), level=level, target=TARGET)(0.5)

    def test_device_level(self, circuit):
        """Test that an error is raised for level="device"."""

        with pytest.raises(NotImplementedError, match="does not support level='device' yet"):
            qp.estimate(circuit, level="device", target=TARGET)(0.5)

    @pytest.mark.parametrize(
        "level",
        [0.0, False, np.array([0]), np.zeros((1, 2), dtype=int), [], [0.5], slice(-1, None)],
    )
    def test_invalid_level(self, circuit, level):
        """Test that invalid levels are rejected, even if they are of an unsupported type such as
        a Sequence or a slice."""

        with pytest.raises(ValueError, match="Invalid level"):
            qp.estimate(circuit, level=level, target=TARGET)(0.5)

    def test_tape_transforms(self, qnode):
        """Test that tape transforms are allowed, since the user transforms are not applied at
        level="top"."""

        @qp.transform
        def dummy_transform(tape):
            """Returns a tape-only transform that can be used for testing"""
            return (tape,), lambda res: res[0]

        specs = qp.estimate(qp.qjit(dummy_transform(qnode)), target=TARGET)(0.5)

        assert specs.resources.counts == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    def test_target_user_is_default(self, qnode):
        """Test that the default target is the "user" level, as the default level of
        qp.analyze."""

        circuit = qp.qjit(qp.transforms.decompose(qnode, gate_set=TARGET))

        specs = qp.estimate(circuit)(0.5)

        assert specs == qp.estimate(circuit, target="user")(0.5)
        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    @pytest.mark.parametrize("target", [2, np.int64(2)])
    def test_device_level_target(self, circuit, target):
        """Test that the device level as an integer target is the same as target="device"."""

        specs = qp.estimate(circuit, target=target)(0.5)

        assert specs == qp.estimate(circuit, target="device")(0.5)

    def test_target_device_capabilities(self, dev_without_cz):
        """Test that target="device" decomposes the operations that the device does not
        support."""

        @qp.qjit
        @qp.qnode(dev_without_cz)
        def circuit(x):
            qp.CZ([0, 1])
            qp.RX(x, wires=1)
            return qp.probs()

        specs = qp.estimate(circuit, target="device")(0.5)

        assert specs.resources.quantum_operations == {"ControlledPhaseShift": 1, "RX": 1}

    def test_target_device_uses_pipeline_rules(self, dev_without_cz):
        """Test that target="device" uses the rules of the decompose transforms in the
        pipeline."""

        with qp.decomposition.local_decomps():
            qp.add_decomps(qp.CZ, cz_with_ry)

            @qp.qjit
            @qp.decompose(gate_set=TARGET | {"RY", "PauliZ"}, fixed_decomps={qp.CZ: cz_with_ry})
            @qp.qnode(dev_without_cz)
            def circuit(x):
                qp.CZ([0, 1])
                qp.RX(x, wires=1)
                return qp.probs()

            specs = qp.estimate(circuit, target="device")(0.5)

        assert specs.resources.quantum_operations == {"PauliZ": 2, "RY": 2, "CNOT": 1, "RX": 1}

    def test_target_user(self, qnode):
        """Test that target="user" uses the gate set of the last decompose transform in the
        pipeline."""

        circuit = qp.transforms.decompose(qnode, gate_set=TARGET | {"CZ"})
        circuit = qp.transforms.decompose(circuit, gate_set=TARGET)

        specs = qp.estimate(qp.qjit(circuit), target="user")(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    @pytest.mark.parametrize("target", ["top", 0, "user", 1])
    def test_level_target_without_decompose(self, circuit, target):
        """Test that a level before the device level without a decompose transform before it
        gives the resources at level 0."""

        specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources == qp.analyze(circuit, level=0)(0.5).resources
        assert specs.resources.quantum_operations == {"Hadamard": 2, "CZ": 1, "RX": 1}

    @pytest.mark.parametrize("with_marker", [False, True])
    def test_default_target_without_transforms(self, qnode, with_marker):
        """Test that the default target and a marker target at level 0 give the resources at
        level 0 without any transform."""

        circuit = qp.qjit(qp.marker(qnode, "start") if with_marker else qnode)
        kwargs = {"target": "start"} if with_marker else {}

        specs = qp.estimate(circuit, **kwargs)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CZ": 1, "RX": 1}

    @pytest.mark.parametrize(
        "gate_sets",
        [
            [],
            [{"RZ", "RY", "CNOT", "GlobalPhase"}],
            # lightning.qubit does not support QFT
            [{"RZ", "RY", "CNOT", "GlobalPhase", "QFT"}],
            [{"RZ", "RY", "CNOT", "GlobalPhase", "Hadamard"}, {"RZ", "RY", "CNOT", "GlobalPhase"}],
        ],
    )
    @pytest.mark.parametrize("target", ["device", "user", 0, 1])
    def test_level_target_matches_analyze(self, target, gate_sets):
        """Test that a level as target gives the same gates as qp.analyze at that level if the
        compile pipeline has no transforms other than decompose."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.QFT(wires=[0, 1])
            qp.Hadamard(0)
            qp.CZ([0, 1])
            return qp.probs()

        for gate_set in gate_sets:
            circuit = qp.decompose(circuit, gate_set=gate_set)
        circuit = qp.qjit(circuit)

        specs = qp.estimate(circuit, target=target)()

        expected = qp.analyze(circuit, level=target)().resources.quantum_operations
        assert specs.resources.quantum_operations == expected

    def test_target_device_with_finite_shots(self):
        """Test that target="device" uses the gate set of the device with finite shots."""

        @qp.qjit
        @qp.set_shots(100)
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.QFT(wires=[0, 1])
            return qp.sample()

        specs = qp.estimate(circuit, target="device")()

        expected = qp.analyze(circuit, level="device")().resources.quantum_operations
        assert specs.resources.quantum_operations == expected

    def test_target_device_uses_decompose_gate_set(self, qnode):
        """Test that target="device" uses the gate set of the last decompose transform if the
        device supports all its gates, even if the device supports gates that it removes."""

        circuit = qp.qjit(qp.transforms.decompose(qnode, gate_set=TARGET))

        specs = qp.estimate(circuit, target="device")(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    def test_target_device_with_unsupported_decompose_gate_set(self, dev_without_cz):
        """Test that target="device" decomposes the gates of the last decompose transform into
        the gate set of the device if it does not support all of them."""

        @qp.qjit
        @qp.decompose(gate_set={"CZ", "Hadamard"})
        @qp.qnode(dev_without_cz)
        def circuit():
            qp.CNOT([0, 1])
            return qp.probs()

        specs = qp.estimate(circuit, target="device")()

        assert specs.resources.quantum_operations == {"ControlledPhaseShift": 1, "Hadamard": 2}
        expected = qp.analyze(circuit, level="device")().resources.quantum_operations
        assert specs.resources.quantum_operations == expected

    @pytest.mark.parametrize("target", ["user", 2, "device"])
    def test_gridsynth_after_decompose(self, target):
        """Test that the rotations that gridsynth would approximate are left as they are, without
        a warning."""

        @qp.qjit
        @qp.transforms.gridsynth
        @qp.decompose(gate_set={"RZ", "Hadamard", "CNOT", "GlobalPhase"})
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x):
            qp.CRX(x, [0, 1])
            return qp.probs()

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources.quantum_operations == {"CNOT": 2, "Hadamard": 2, "RZ": 2}

    def test_unreachable_target_after_gridsynth(self):
        """Test that the error for a level target that can only be reached with a skipped
        transform names the skipped transforms."""

        @qp.qjit
        @qp.decompose(gate_set={"Hadamard", "S", "T", "CNOT", "GlobalPhase"})
        @qp.transforms.cancel_inverses
        @qp.transforms.gridsynth
        @qp.transforms.cancel_inverses
        @qp.decompose(gate_set={"RZ", "Hadamard", "CNOT", "GlobalPhase"})
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x):
            qp.CRX(x, [0, 1])
            return qp.probs()

        match = "skips the transforms cancel-inverses, gridsynth, which may be needed"
        with pytest.raises(ValueError, match=match):
            qp.estimate(circuit, target="user")(0.5)

    @pytest.mark.parametrize(
        "transform",
        [
            qp.transforms.combine_global_phases,
            qp.transforms.split_to_single_terms,
            catalyst.passes.cancel_inverses,
            custom_transform,
        ],
    )
    @pytest.mark.parametrize("target", ["user", 2, "device"])
    def test_transform_after_decompose(self, qnode, transform, target):
        """Test that a transform other than gridsynth after the last decompose transform is
        skipped without a warning."""

        circuit = qp.qjit(transform(qp.transforms.decompose(qnode, gate_set=TARGET)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    @pytest.mark.parametrize(
        "with_decompose, counts",
        [
            (False, {"Hadamard": 2, "CZ": 1, "RX": 1}),
            (True, {"Hadamard": 4, "CNOT": 1, "RX": 1}),
        ],
    )
    @pytest.mark.parametrize("target", ["user", "device"])
    def test_transform_not_after_last_decompose(self, qnode, with_decompose, counts, target):
        """Test that a transform is skipped without a warning if there is no decompose transform
        before it, or if a decompose transform comes after it."""

        circuit = qnode
        if with_decompose:
            circuit = qp.transforms.decompose(circuit, gate_set=TARGET)
        circuit = custom_transform(circuit)
        if with_decompose:
            circuit = qp.transforms.decompose(circuit, gate_set=TARGET)
        circuit = qp.qjit(circuit)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources.quantum_operations == counts

    @pytest.mark.parametrize("target", ["user", 2, "device"])
    def test_tape_transform_before_level_target(self, qnode, target):
        """Test that an error is raised for a tape transform before a level target, since the
        compile pipeline cannot be compiled, as in qp.analyze."""

        circuit = qp.transforms.decompose(qnode, gate_set=TARGET)
        circuit = qp.qjit(qp.transforms.undo_swaps(circuit))

        with pytest.raises(ValueError, match="Tape transforms are no longer supported"):
            qp.estimate(circuit, target=target)(0.5)

    def test_tape_transform_after_level_target(self, qnode):
        """Test that a tape transform after a level target is not checked, since the
        transforms after that level are not compiled."""

        circuit = qp.transforms.decompose(qnode, gate_set=TARGET)
        circuit = qp.qjit(qp.transforms.undo_swaps(circuit))

        specs = qp.estimate(circuit, target=1)(0.5)
        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    @pytest.mark.parametrize("target", ["user", "device"])
    def test_catalyst_graph_decomposition(self, qnode, target):
        """Test that Catalyst's graph_decomposition pass is a decompose transform."""

        circuit = qp.qjit(catalyst.passes.graph_decomposition(qnode, gate_set=TARGET))

        specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    @pytest.mark.parametrize("transform_after", [False, True])
    def test_unreachable_level_target(self, qnode, transform_after):
        """Test that the error for a level target whose gate set cannot be reached names the
        gate set and the target, but not the transforms after the last decompose transform,
        which cannot help reach it."""

        circuit = qp.transforms.decompose(qnode, gate_set={"Hadamard", "CNOT"})
        if transform_after:
            circuit = qp.transforms.gridsynth(circuit)
        circuit = qp.qjit(circuit)

        match = r"into the gate set \['CNOT', 'Hadamard'\] at target='user'.$"
        with pytest.raises(ValueError, match=match):
            qp.estimate(circuit, target="user")(0.5)

    def test_unreachable_level_target_after_transform(self, qnode):
        """Test that the error for a level target that cannot be reached names the skipped
        transform before the last decompose transform."""

        circuit = qp.transforms.cancel_inverses(qnode)
        circuit = qp.qjit(qp.transforms.decompose(circuit, gate_set={"Hadamard", "CNOT"}))

        with pytest.raises(ValueError, match="skips the transform cancel-inverses, which may"):
            qp.estimate(circuit, target="user")(0.5)

    def test_unreachable_gate_set_before_level_target(self):
        """Test that the error for a decompose transform before a level target that cannot be
        applied names its gate set, instead of the gate set at the target."""

        @qp.qjit(capture=True)
        @qp.decompose(gate_set={"RZ", "RY", "CNOT", "GlobalPhase"})
        @qp.decompose(gate_set={"RX"})
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.CNOT([0, 1])
            return qp.probs()

        match = r"operator CNOT: .* into the gate set \['RX'\] of a qp.decompose transform before"
        with pytest.raises(ValueError, match=match):
            qp.estimate(circuit, target="user")()

    def test_device_without_decomposition(self, circuit, monkeypatch):
        """Test that an error is raised for target="device" if the device preprocessing has no
        decomposition."""

        monkeypatch.setattr(
            "catalyst.from_plxpr.device_utils.create_device_preprocessing_pipeline",
            lambda *_, **__: qp.CompilePipeline(),
        )

        with pytest.raises(NotImplementedError, match="could not find the gate set of the device"):
            qp.estimate(circuit, target="device")(0.5)

    def test_multi_controlled_gates_match_analyze(self):
        """Test that target="device" keeps the number of control wires of multi-controlled
        gates, as qp.analyze does."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit(x):
            qp.ctrl(qp.RX(x, 3), [0, 1])
            qp.ctrl(qp.RX(x, 3), [0, 1, 2])
            return qp.probs()

        specs = qp.estimate(circuit, target="device")(0.1)

        assert specs.resources.quantum_operations == {"2C(RX)": 1, "3C(RX)": 1}
        expected = qp.analyze(circuit, level="device")(0.1).resources.quantum_operations
        assert specs.resources.quantum_operations == expected

    def test_marker_target(self, qnode):
        """Test that a marker label as target is the level of the marker."""

        circuit = qp.transforms.cancel_inverses(qnode)
        circuit = qp.marker(circuit, "after_cancel")
        circuit = qp.qjit(qp.transforms.decompose(circuit, gate_set=TARGET))

        specs = qp.estimate(circuit, target="after_cancel")(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CZ": 1, "RX": 1}

    def test_marker_target_with_gate_name(self, qnode):
        """Test that a marker label as target is a level even if it is also a gate name."""

        circuit = qp.qjit(qp.transforms.decompose(qp.marker(qnode, "CNOT"), gate_set=TARGET))

        specs = qp.estimate(circuit, target="CNOT")(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CZ": 1, "RX": 1}

    @pytest.mark.parametrize("target", ["all-user", "all", slice(0, 1)])
    def test_several_levels_target(self, circuit, target):
        """Test that an error is raised for a target that selects several levels."""

        with pytest.raises(ValueError, match="selects several levels"):
            qp.estimate(circuit, target=target)(0.5)

    @pytest.mark.parametrize("target", ["gradient", "all-mlir"])
    def test_removed_level_target(self, circuit, target):
        """Test that an error is raised for a level name that is no longer supported, without
        suggesting it as a gate name."""

        with pytest.raises(ValueError, match="no longer supported.$"):
            qp.estimate(circuit, target=target)(0.5)

    @pytest.mark.parametrize("target", ["devices", "CNOT"])
    def test_unknown_level_name_target(self, circuit, target):
        """Test that an error is raised for a string target that is not a level, even if it is a
        gate name, and that it says how to use it as a gate set."""

        match = rf"Invalid target='{target}': Marker name .* pass target={{'{target}'}}.$"
        with pytest.raises(ValueError, match=match):
            qp.estimate(circuit, target=target)(0.5)

    def test_level_name_in_gate_set_target(self, circuit):
        """Test that the error for a gate set target that contains a level name says how to use
        the level as target."""

        with pytest.raises(ValueError, match="contains 'user', .* such as target='user'.$"):
            qp.estimate(circuit, target=["user"])(0.5)

    @pytest.mark.parametrize("target", [True, None])
    def test_invalid_target(self, circuit, target):
        """Test that an error is raised for a bool target, which is not an integer level, and for
        None."""

        with pytest.raises(ValueError, match=f"Invalid target={target}, expected a gate set"):
            qp.estimate(circuit, target=target)(0.5)

    @pytest.mark.parametrize("target", [-1, 3])
    def test_integer_target_out_of_bounds(self, circuit, target):
        """Test that an error is raised for an integer target that is not a level of the
        pipeline."""

        with pytest.raises(ValueError, match=f"Invalid target={target}: .* from 0 to 2"):
            qp.estimate(circuit, target=target)(0.5)

    def test_error_with_non_qjit(self):
        """Test that an error is raised if the QNode is not qjit'd."""

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            return qp.probs()

        with pytest.raises(ValueError, match="qp.estimate can only be applied to a qjit'd QNode"):
            qp.estimate(circuit, target=TARGET)()

    def test_dynamic_allocation(self):
        """Test that estimate counts the gates acting on dynamically allocated wires."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            with qp.allocate(2) as aux:
                qp.CZ(aux)
            return qp.probs(wires=0)

        resources = qp.estimate(circuit, target={"Hadamard", "CNOT"})().resources

        assert resources.quantum_operations == {"Hadamard": 2, "CNOT": 1}
        assert resources.num_wires == 4

    def test_subroutine_in_adjoint(self):
        """Test that a subroutine called both inside and outside an adjoint region has its
        operators counted once in each context."""

        @qp.capture.subroutine
        def helper(wire):
            qp.Hadamard(wire)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.adjoint(helper)(0)
            helper(0)
            return qp.state()

        resources = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources

        assert resources.quantum_operations == {"Adjoint(Hadamard)": 1, "Hadamard": 1}

    def test_nested_subroutine_in_adjoint(self):
        """Test that an adjoint applied to a subroutine also applies to the subroutines it
        calls."""

        @qp.capture.subroutine
        def inner(wire):
            qp.S(wire)

        @qp.capture.subroutine
        def outer(wire):
            inner(wire)
            qp.T(wire)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.adjoint(outer)(0)
            outer(0)
            return qp.state()

        resources = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources

        assert resources.quantum_operations == {"Adjoint(S)": 1, "Adjoint(T)": 1, "S": 1, "T": 1}

    def test_subroutine_in_ctrl(self):
        """Test that a subroutine called both inside and outside a control region has its
        operators counted once in each context."""

        @qp.capture.subroutine
        def helper(wire):
            qp.X(wire)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.ctrl(helper, control=(1, 2))(0)
            helper(0)
            helper(1)
            return qp.state()

        resources = qp.estimate(circuit, target=qp.gate_sets.CLIFFORD_T)().resources

        # The doubly-controlled X becomes a Toffoli plus two PauliX, which the MultiControlledX
        # rule counts as an average-case estimate for flipping zero-valued controls.
        assert resources.quantum_operations == {
            "Adjoint(T)": 3,
            "CNOT": 6,
            "Hadamard": 2,
            "PauliX": 4,
            "T": 4,
        }


@pytest.mark.usefixtures("disable_capture")
@pytest.mark.parametrize("capture", [False, "global"])
def test_error_without_capture(capture):
    """Test that an error is raised if the QNode is compiled without program capture."""

    @qp.qjit(capture=capture)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    with pytest.raises(ValueError, match=r"qp.estimate requires program capture"):
        qp.estimate(circuit, target=TARGET)()
