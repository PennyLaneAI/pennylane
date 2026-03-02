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

"""Tests for the transform ``qp.transform.rz_phase_gradient_decompmake_rz_to_phase_gradient_decomp``"""
# pylint: disable=no-value-for-parameter
import pytest

import pennylane as qp
from pennylane.labs.transforms.rot_to_phase_gradient import binary_repr_int
from pennylane.labs.transforms.rz_phase_gradient_decomp import make_rz_to_phase_gradient_decomp


# @pytest.mark.usefixtures("enable_graph_decomposition") # fixture doesnt exist in labs tests
@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_fixed_decomps(phi, p):
    """Test that the decomposition rule from make_rz_to_phase_gradient_decomp works as expected as a fixed decomposition and yields the correct resources"""
    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

        kwargs = {
            "angle_wires": angle_wires,
            "phase_grad_wires": phase_grad_wires,
            "work_wires": work_wires,
        }

        custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)

        @qp.transforms.decompose(
            gate_set={"SemiAdder", "CNOT", "GlobalPhase"}, fixed_decomps={qp.RZ: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].gate_types

        expected_specs = {"GlobalPhase": 1, "SemiAdder": 1}
        if (n_cnots := 2 * sum(binary_repr_int(phi * 2, p))) > 0:
            expected_specs["CNOT"] = n_cnots

        assert expected_specs == specs


@pytest.mark.parametrize("phi", [0.5, 0.3, 1 / 2 + 1 / 4 + 1 / 8, 1.0])
@pytest.mark.parametrize("p", [2, 3, 4])
def test_as_alt_decomps(phi, p):
    """Test that the decomposition rule from make_rz_to_phase_gradient_decomp works as expected as an alternative decomposition and yields the correct resources"""
    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner

        angle_wires = qp.wires.Wires([f"aux_{i}" for i in range(p)])
        phase_grad_wires = qp.wires.Wires([f"qft_{i}" for i in range(p)])
        work_wires = qp.wires.Wires([f"work_{i}" for i in range(p - 1)])

        kwargs = {
            "angle_wires": angle_wires,
            "phase_grad_wires": phase_grad_wires,
            "work_wires": work_wires,
        }

        custom_decomp = make_rz_to_phase_gradient_decomp(**kwargs)

        @qp.transforms.decompose(
            gate_set={"SemiAdder", "CNOT", "GlobalPhase"}, alt_decomps={qp.RZ: [custom_decomp]}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit():
            qp.RZ(phi, 0)
            return qp.state()

        specs = qp.specs(circuit)()["resources"].gate_types

        expected_specs = {"GlobalPhase": 1, "SemiAdder": 1}
        if (n_cnots := 2 * sum(binary_repr_int(phi * 2, p))) > 0:
            expected_specs["CNOT"] = n_cnots

        assert expected_specs == specs
