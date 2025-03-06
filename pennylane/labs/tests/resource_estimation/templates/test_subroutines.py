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
"""
Test the ResourceQFT class
"""
import pytest

import pennylane.labs.resource_estimation as re
import pennylane as qml

# pylint: disable=no-self-use


class TestQFT:
    """Test the ResourceQFT class"""

    @pytest.mark.parametrize(
        "num_wires, num_hadamard, num_swap, num_ctrl_phase_shift",
        [
            (1, 1, 0, 0),
            (2, 2, 1, 1),
            (3, 3, 1, 3),
            (4, 4, 2, 6),
        ],
    )
    def test_resources(self, num_wires, num_hadamard, num_swap, num_ctrl_phase_shift):
        """Test the resources method returns the correct dictionary"""
        hadamard = re.CompressedResourceOp(re.ResourceHadamard, {})
        swap = re.CompressedResourceOp(re.ResourceSWAP, {})
        ctrl_phase_shift = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        expected = {hadamard: num_hadamard, swap: num_swap, ctrl_phase_shift: num_ctrl_phase_shift}

        assert re.ResourceQFT.resources(num_wires) == expected

    @pytest.mark.parametrize("wires", [range(1), range(2), range(3), range(4)])
    def test_resource_params(self, wires):
        """Test that the resource params are correct"""
        op = re.ResourceQFT(wires)
        assert op.resource_params == {"num_wires": len(wires)}

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4])
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(re.ResourceQFT, {"num_wires": num_wires})
        assert re.ResourceQFT.resource_rep(num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, num_hadamard, num_swap, num_ctrl_phase_shift",
        [
            (1, 1, 0, 0),
            (2, 2, 1, 1),
            (3, 3, 1, 3),
            (4, 4, 2, 6),
        ],
    )
    def test_resources_from_rep(self, num_wires, num_hadamard, num_swap, num_ctrl_phase_shift):
        """Test that computing the resources from a compressed representation works"""

        hadamard = re.CompressedResourceOp(re.ResourceHadamard, {})
        swap = re.CompressedResourceOp(re.ResourceSWAP, {})
        ctrl_phase_shift = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        expected = {hadamard: num_hadamard, swap: num_swap, ctrl_phase_shift: num_ctrl_phase_shift}

        rep = re.ResourceQFT.resource_rep(num_wires)
        actual = rep.op_type.resources(**rep.params)

        assert actual == expected

    @pytest.mark.parametrize("num_wires", range(10))
    def test_tracking_name(self, num_wires):
        """Test that the tracking name is correct."""
        assert re.ResourceQFT.tracking_name(num_wires + 1) == f"QFT({num_wires+1})"


class TestControlledSequence:
    """Test the ResourceControlledSequence class"""

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_resources(self, base_class, base_params, num_ctrl_wires):
        """Test the resources method returns the correct dictionary"""
        resource_controlled_sequence = re.CompressedResourceOp(
            re.ResourceControlled,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )
        expected = {resource_controlled_sequence: 2**num_ctrl_wires - 1}

        assert (
            re.ResourceControlledSequence.resources(base_class, base_params, num_ctrl_wires)
            == expected
        )

    @pytest.mark.parametrize(
        "base, control",
        [(re.ResourceHadamard(3), [0, 1, 2]), (re.ResourceRX(0.25, 2), [0, 1])],
    )
    def test_resource_params(self, base, control):
        """Test that the resource params are correct"""
        op = re.ResourceControlledSequence(base=base, control=control)

        assert op.resource_params == {
            "base_class": type(base),
            "base_params": base.resource_params,
            "num_ctrl_wires": len(control),
        }

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_resource_rep(self, base_class, base_params, num_ctrl_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceControlledSequence,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )
        assert expected == re.ResourceControlledSequence.resource_rep(
            base_class, base_params, num_ctrl_wires
        )

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_resources_from_rep(self, base_class, base_params, num_ctrl_wires):
        """Test that computing the resources from a compressed representation works"""

        resource_controlled_sequence = re.CompressedResourceOp(
            re.ResourceControlled,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )

        expected = {resource_controlled_sequence: 2**num_ctrl_wires - 1}

        rep = re.ResourceControlledSequence.resource_rep(base_class, base_params, num_ctrl_wires)
        actual = rep.op_type.resources(**rep.params)

        assert actual == expected

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_tracking_name(self, base_class, base_params, num_ctrl_wires):
        """Test that the tracking name is correct."""
        base_name = base_class.tracking_name(**base_params)
        assert (
            re.ResourceControlledSequence.tracking_name(base_class, base_params, num_ctrl_wires)
            == f"ControlledSequence({base_name}, {num_ctrl_wires})"
        )


class TestPhaseAdder:
    """Test the ResourcePhaseAdder class"""

    @pytest.mark.parametrize(
        "mod, num_x_wires",
        [(8, 3), (7, 3)],
    )
    def test_resources(self, mod, num_x_wires):
        """Test the resources method returns the correct dictionary"""
        if mod == 2**num_x_wires:
            resource_phase_shift = re.CompressedResourceOp(re.ResourcePhaseShift, {})
            expected = {resource_phase_shift: num_x_wires}
            assert re.ResourcePhaseAdder.resources(mod, num_x_wires) == expected
            return

        qft = re.CompressedResourceOp(re.ResourceQFT, {"num_wires": num_x_wires})
        qft_dag = re.CompressedResourceOp(
            re.ResourceAdjoint,
            {"base_class": re.ResourceQFT, "base_params": {"num_wires": num_x_wires}},
        )

        phase_shift = re.CompressedResourceOp(re.ResourcePhaseShift, {})
        phase_shift_dag = re.CompressedResourceOp(
            re.ResourceAdjoint, {"base_class": re.ResourcePhaseShift, "base_params": {}}
        )
        ctrl_phase_shift = re.CompressedResourceOp(
            re.ResourceControlled,
            {
                "base_class": re.ResourcePhaseShift,
                "base_params": {},
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        multix = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {"num_ctrl_wires": 1, "num_ctrl_values": 0, "num_work_wires": 1},
        )

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[phase_shift] = 2 * num_x_wires
        gate_types[phase_shift_dag] = 2 * num_x_wires
        gate_types[ctrl_phase_shift] = num_x_wires
        gate_types[cnot] = 1
        gate_types[multix] = 1

        assert re.ResourcePhaseAdder.resources(mod, num_x_wires) == gate_types

    @pytest.mark.parametrize(
        "mod, x_wires",
        [(8, [0, 1, 2]), (3, [0, 1])],
    )
    def test_resource_params(self, mod, x_wires):
        """Test that the resource params are correct"""
        op = re.ResourcePhaseAdder(k=3, mod=mod, x_wires=x_wires, work_wire=[5])

        assert op.resource_params == {
            "mod": mod,
            "num_x_wires": len(x_wires),
        }

    @pytest.mark.parametrize(
        "mod, num_x_wires",
        [(8, 3), (3, 2)],
    )
    def test_resource_rep(self, mod, num_x_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourcePhaseAdder,
            {
                "mod": mod,
                "num_x_wires": num_x_wires,
            },
        )
        assert expected == re.ResourcePhaseAdder.resource_rep(mod=mod, num_x_wires=num_x_wires)

    @pytest.mark.parametrize(
        "mod, num_x_wires",
        [(8, 3), (7, 3)],
    )
    def test_resources_from_rep(self, mod, num_x_wires):
        """Test that computing the resources from a compressed representation works"""
        rep = re.ResourcePhaseAdder.resource_rep(mod=mod, num_x_wires=num_x_wires)
        actual = rep.op_type.resources(**rep.params)
        if mod == 2**num_x_wires:
            resource_phase_shift = re.CompressedResourceOp(re.ResourcePhaseShift, {})
            expected = {resource_phase_shift: num_x_wires}
            assert expected == actual
            return

        qft = re.CompressedResourceOp(re.ResourceQFT, {"num_wires": num_x_wires})
        qft_dag = re.CompressedResourceOp(
            re.ResourceAdjoint,
            {"base_class": re.ResourceQFT, "base_params": {"num_wires": num_x_wires}},
        )

        phase_shift = re.CompressedResourceOp(re.ResourcePhaseShift, {})
        phase_shift_dag = re.CompressedResourceOp(
            re.ResourceAdjoint, {"base_class": re.ResourcePhaseShift, "base_params": {}}
        )
        ctrl_phase_shift = re.CompressedResourceOp(
            re.ResourceControlled,
            {
                "base_class": re.ResourcePhaseShift,
                "base_params": {},
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        multix = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {"num_ctrl_wires": 1, "num_ctrl_values": 0, "num_work_wires": 1},
        )

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[phase_shift] = 2 * num_x_wires
        gate_types[phase_shift_dag] = 2 * num_x_wires
        gate_types[ctrl_phase_shift] = num_x_wires
        gate_types[cnot] = 1
        gate_types[multix] = 1

        assert actual == gate_types

    def test_tracking_name(self):
        """Test that the tracking name is correct."""
        assert re.ResourcePhaseAdder.tracking_name() == f"PhaseAdder"


class TestResourceMultiplier:
    """Test the ResourceMultiplier class"""

    @pytest.mark.parametrize(
        "mod, num_work_wires, num_x_wires",
        [(7, 5, 3), (8, 5, 3)],
    )
    def test_resources(self, mod, num_work_wires, num_x_wires):
        """Test the resources method returns the correct dictionary"""
        if mod == 2**num_x_wires:
            num_aux_wires = num_x_wires
            num_aux_swap = num_x_wires
        else:
            num_aux_wires = num_work_wires - 1
            num_aux_swap = num_aux_wires - 1

        qft = re.CompressedResourceOp(re.ResourceQFT, {"num_wires": num_aux_wires})
        qft_dag = re.CompressedResourceOp(
            re.ResourceAdjoint,
            {"base_class": re.ResourceQFT, "base_params": {"num_wires": num_aux_wires}},
        )

        sequence = re.CompressedResourceOp(
            re.ResourceControlledSequence,
            {
                "base_class": re.ResourcePhaseAdder,
                "base_params": {},
                "num_ctrl_wires": num_x_wires,
            },
        )

        sequence_dag = re.CompressedResourceOp(
            re.ResourceAdjoint,
            {
                "base_class": re.ResourceControlledSequence,
                "base_params": {
                    "base_class": re.ResourcePhaseAdder,
                    "base_params": {},
                    "num_ctrl_wires": num_x_wires,
                },
            },
        )

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})

        gate_types = {}
        gate_types[qft] = 2
        gate_types[qft_dag] = 2
        gate_types[sequence] = 1
        gate_types[sequence_dag] = 1
        gate_types[cnot] = min(num_x_wires, num_aux_swap)

        assert re.ResourceMultiplier.resources(mod, num_work_wires, num_x_wires) == gate_types

    @pytest.mark.parametrize(
        "base, control",
        [(re.ResourceHadamard(3), [0, 1, 2]), (re.ResourceRX(0.25, 2), [0, 1])],
    )
    def test_resource_params(self, base, control):
        """Test that the resource params are correct"""
        op = re.ResourceControlledSequence(base=base, control=control)

        assert op.resource_params == {
            "base_class": type(base),
            "base_params": base.resource_params,
            "num_ctrl_wires": len(control),
        }

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_resource_rep(self, base_class, base_params, num_ctrl_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceControlledSequence,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )
        assert expected == re.ResourceControlledSequence.resource_rep(
            base_class, base_params, num_ctrl_wires
        )

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_resources_from_rep(self, base_class, base_params, num_ctrl_wires):
        """Test that computing the resources from a compressed representation works"""

        resource_controlled_sequence = re.CompressedResourceOp(
            re.ResourceControlled,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )

        expected = {resource_controlled_sequence: 2**num_ctrl_wires - 1}

        rep = re.ResourceControlledSequence.resource_rep(base_class, base_params, num_ctrl_wires)
        actual = rep.op_type.resources(**rep.params)

        assert actual == expected

    @pytest.mark.parametrize(
        "base_class, base_params, num_ctrl_wires",
        [(re.ResourceHadamard, {}, 1), (re.ResourceRX, {}, 3)],
    )
    def test_tracking_name(self, base_class, base_params, num_ctrl_wires):
        """Test that the tracking name is correct."""
        base_name = base_class.tracking_name(**base_params)
        assert (
            re.ResourceControlledSequence.tracking_name(base_class, base_params, num_ctrl_wires)
            == f"ControlledSequence({base_name}, {num_ctrl_wires})"
        )
