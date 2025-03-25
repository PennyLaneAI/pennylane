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
import math

import numpy as np
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use, too-many-arguments, protected-access


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
        "base_class, base_params, num_ctrl_wires, num_cseq",
        [(re.ResourceHadamard, {}, 1, 1), (re.ResourceRX, {}, 3, 7)],
    )
    def test_resources(self, base_class, base_params, num_ctrl_wires, num_cseq):
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
        expected = {resource_controlled_sequence: num_cseq}

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
        "base_class, base_params, num_ctrl_wires, num_cseq",
        [(re.ResourceHadamard, {}, 1, 1), (re.ResourceRX, {}, 3, 7)],
    )
    def test_resources_from_rep(self, base_class, base_params, num_ctrl_wires, num_cseq):
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

        expected = {resource_controlled_sequence: num_cseq}

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
        ctrl_phase_shift = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        multix = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {"num_ctrl_wires": 1, "num_ctrl_values": 0, "num_work_wires": 1},
        )

        expected = {}
        expected[qft] = 2
        expected[qft_dag] = 2
        expected[phase_shift] = 2 * num_x_wires
        expected[phase_shift_dag] = 2 * num_x_wires
        expected[ctrl_phase_shift] = num_x_wires
        expected[cnot] = 1
        expected[multix] = 1

        assert re.ResourcePhaseAdder.resources(mod, num_x_wires) == expected

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
        ctrl_phase_shift = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        multix = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {"num_ctrl_wires": 1, "num_ctrl_values": 0, "num_work_wires": 1},
        )

        expected = {}
        expected[qft] = 2
        expected[qft_dag] = 2
        expected[phase_shift] = 2 * num_x_wires
        expected[phase_shift_dag] = 2 * num_x_wires
        expected[ctrl_phase_shift] = num_x_wires
        expected[cnot] = 1
        expected[multix] = 1

        assert actual == expected

    def test_tracking_name(self):
        """Test that the tracking name is correct."""
        assert re.ResourcePhaseAdder.tracking_name() == "PhaseAdder"


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

        expected = {}
        expected[qft] = 2
        expected[qft_dag] = 2
        expected[sequence] = 1
        expected[sequence_dag] = 1
        expected[cnot] = min(num_x_wires, num_aux_swap)

        assert re.ResourceMultiplier.resources(mod, num_work_wires, num_x_wires) == expected

    @pytest.mark.parametrize(
        "k, mod, work_wires, x_wires",
        [(4, 7, [3, 4, 5, 6, 7], [0, 1, 2]), (5, 8, [3, 4, 5, 6, 7], [0, 1, 2])],
    )
    def test_resource_params(self, k, mod, work_wires, x_wires):
        """Test that the resource params are correct"""
        op = re.ResourceMultiplier(
            k=k,
            x_wires=x_wires,
            mod=mod,
            work_wires=work_wires,
        )

        assert op.resource_params == {
            "mod": mod,
            "num_work_wires": len(work_wires),
            "num_x_wires": len(x_wires),
        }

    @pytest.mark.parametrize(
        "mod, num_work_wires, num_x_wires",
        [(7, 5, 3), (8, 5, 3)],
    )
    def test_resource_rep(self, mod, num_work_wires, num_x_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceMultiplier,
            {
                "mod": mod,
                "num_work_wires": num_work_wires,
                "num_x_wires": num_x_wires,
            },
        )
        assert expected == re.ResourceMultiplier.resource_rep(mod, num_work_wires, num_x_wires)

    @pytest.mark.parametrize(
        "mod, num_work_wires, num_x_wires",
        [(7, 5, 3), (8, 5, 3)],
    )
    def test_resources_from_rep(self, mod, num_work_wires, num_x_wires):
        """Test that computing the resources from a compressed representation works"""

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

        expected = {}
        expected[qft] = 2
        expected[qft_dag] = 2
        expected[sequence] = 1
        expected[sequence_dag] = 1
        expected[cnot] = min(num_x_wires, num_aux_swap)

        rep = re.ResourceMultiplier.resource_rep(mod, num_work_wires, num_x_wires)
        actual = rep.op_type.resources(**rep.params)

        assert actual == expected

    def test_tracking_name(self):
        """Test that the tracking name is correct."""
        assert re.ResourceMultiplier.tracking_name() == "Multiplier"


class TestResourceModExp:
    """Test the ResourceModExp class"""

    @pytest.mark.parametrize(
        "mod, num_output_wires, num_work_wires, num_x_wires",
        [(7, 5, 5, 3), (8, 5, 5, 3)],
    )
    def test_resources(self, mod, num_output_wires, num_work_wires, num_x_wires):
        """Test that the resources are correct"""
        mult_resources = re.ResourceMultiplier._resource_decomp(
            mod, num_work_wires, num_output_wires
        )
        expected = {}

        for comp_rep, _ in mult_resources.items():
            new_rep = re.CompressedResourceOp(
                re.ResourceControlled,
                {
                    "base_class": comp_rep.op_type,
                    "base_params": comp_rep.params,
                    "num_ctrl_wires": 1,
                    "num_ctrl_values": 0,
                    "num_work_wires": 0,
                },
            )

            if comp_rep._name in ("QFT", "Adjoint(QFT)"):
                expected[new_rep] = 1
            else:
                expected[new_rep] = mult_resources[comp_rep] * ((2**num_x_wires) - 1)

        assert (
            re.ResourceModExp.resources(mod, num_output_wires, num_work_wires, num_x_wires)
            == expected
        )

    @pytest.mark.parametrize(
        "k, mod, work_wires, x_wires",
        [(4, 7, [3, 4, 5, 6, 7], [0, 1, 2]), (5, 8, [3, 4, 5, 6, 7], [0, 1, 2])],
    )
    def test_resource_params(self, k, mod, work_wires, x_wires):
        """Test that the resource params are correct"""
        op = re.ResourceMultiplier(
            k=k,
            x_wires=x_wires,
            mod=mod,
            work_wires=work_wires,
        )

        assert op.resource_params == {
            "mod": mod,
            "num_work_wires": len(work_wires),
            "num_x_wires": len(x_wires),
        }

    @pytest.mark.parametrize(
        "mod, num_output_wires, num_work_wires, num_x_wires",
        [(7, 5, 5, 3), (8, 5, 5, 3)],
    )
    def test_resource_rep(self, mod, num_output_wires, num_work_wires, num_x_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceModExp,
            {
                "mod": mod,
                "num_output_wires": num_output_wires,
                "num_work_wires": num_work_wires,
                "num_x_wires": num_x_wires,
            },
        )
        assert expected == re.ResourceModExp.resource_rep(
            mod, num_output_wires, num_work_wires, num_x_wires
        )

    @pytest.mark.parametrize(
        "mod, num_output_wires, num_work_wires, num_x_wires",
        [(7, 5, 5, 3), (8, 5, 5, 3)],
    )
    def test_resources_from_rep(self, num_output_wires, mod, num_work_wires, num_x_wires):
        """Test that computing the resources from a compressed representation works"""
        mult_resources = re.ResourceMultiplier._resource_decomp(
            mod, num_work_wires, num_output_wires
        )
        expected = {}

        for comp_rep, _ in mult_resources.items():
            new_rep = re.CompressedResourceOp(
                re.ResourceControlled,
                {
                    "base_class": comp_rep.op_type,
                    "base_params": comp_rep.params,
                    "num_ctrl_wires": 1,
                    "num_ctrl_values": 0,
                    "num_work_wires": 0,
                },
            )

            if comp_rep._name in ("QFT", "Adjoint(QFT)"):
                expected[new_rep] = 1
            else:
                expected[new_rep] = mult_resources[comp_rep] * ((2**num_x_wires) - 1)

        rep = re.ResourceModExp.resource_rep(mod, num_output_wires, num_work_wires, num_x_wires)
        actual = rep.op_type.resources(**rep.params)

        assert actual == expected

    def test_tracking_name(self):
        """Test that the tracking name is correct."""
        assert re.ResourceModExp.tracking_name() == "ModExp"


class TestResourceQROM:
    """Test the ResourceQROM class"""

    @pytest.mark.parametrize(
        "num_bitstrings, num_bit_flips, num_control_wires, num_work_wires, size_bitstring, clean",
        [(4, 2, 2, 3, 3, True), (4, 5, 2, 3, 3, False), (4, 5, 0, 3, 3, False)],
    )
    def test_resources(
        self,
        num_bitstrings,
        num_bit_flips,
        num_control_wires,
        num_work_wires,
        size_bitstring,
        clean,
    ):
        """Test that the resources are correct"""
        expected = {}
        x = re.CompressedResourceOp(re.ResourceX, {})

        if num_control_wires == 0:
            expected[x] = num_bit_flips
            assert (
                re.ResourceQROM.resources(
                    num_bitstrings,
                    num_bit_flips,
                    num_control_wires,
                    num_work_wires,
                    size_bitstring,
                    clean,
                )
                == expected
            )
            return

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        hadamard = re.CompressedResourceOp(re.ResourceHadamard, {})

        num_parallel_computations = (num_work_wires + size_bitstring) // size_bitstring
        num_parallel_computations = min(num_parallel_computations, num_bitstrings)

        num_swap_wires = math.floor(math.log2(num_parallel_computations))
        num_select_wires = math.ceil(math.log2(math.ceil(num_bitstrings / (2**num_swap_wires))))
        assert num_swap_wires + num_select_wires <= num_control_wires

        swap_clean_prefactor = 1
        select_clean_prefactor = 1

        if clean:
            expected[hadamard] = 2 * size_bitstring
            swap_clean_prefactor = 4
            select_clean_prefactor = 2

        # SELECT cost:
        expected[cnot] = num_bit_flips  # each unitary in the select is just a CNOT

        multi_x = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": num_select_wires,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )

        num_total_ctrl_possibilities = 2**num_select_wires
        expected[multi_x] = select_clean_prefactor * (
            2 * num_total_ctrl_possibilities  # two applications targetting the aux qubit
        )
        num_zero_controls = (2 * num_total_ctrl_possibilities * num_select_wires) // 2
        expected[x] = select_clean_prefactor * (
            num_zero_controls * 2  # conjugate 0 controls on the multi-qubit x gates from above
        )
        # SWAP cost:
        ctrl_swap = re.CompressedResourceOp(re.ResourceCSWAP, {})
        expected[ctrl_swap] = swap_clean_prefactor * ((2**num_swap_wires) - 1) * size_bitstring

        assert (
            re.ResourceQROM.resources(
                num_bitstrings,
                num_bit_flips,
                num_control_wires,
                num_work_wires,
                size_bitstring,
                clean,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "bitstrings, control_wires, work_wires, target_wires, clean",
        [
            (["000", "001", "010", "100"], [0, 1], [2, 3, 4], [5, 6, 7], True),
            (["010", "111", "110", "000"], [0, 1], [2, 3, 4], [5, 6, 7], False),
        ],
    )
    def test_resource_params(self, bitstrings, control_wires, work_wires, target_wires, clean):
        """Test that the resource params are correct"""
        op = re.ResourceQROM(
            bitstrings=bitstrings,
            control_wires=control_wires,
            target_wires=work_wires,
            work_wires=target_wires,
            clean=clean,
        )

        num_bitstrings = len(bitstrings)

        num_bit_flips = 0
        for bit_string in bitstrings:
            num_bit_flips += bit_string.count("1")

        num_work_wires = len(work_wires)
        size_bitstring = len(target_wires)
        num_control_wires = len(control_wires)

        assert op.resource_params == {
            "num_bitstrings": num_bitstrings,
            "num_bit_flips": num_bit_flips,
            "num_control_wires": num_control_wires,
            "num_work_wires": num_work_wires,
            "size_bitstring": size_bitstring,
            "clean": clean,
        }

    @pytest.mark.parametrize(
        "num_bitstrings, num_bit_flips, num_control_wires, num_work_wires, size_bitstring, clean",
        [(4, 2, 2, 3, 3, True), (4, 5, 2, 3, 3, False), (4, 5, 0, 3, 3, False)],
    )
    def test_resource_rep(
        self,
        num_bitstrings,
        num_bit_flips,
        num_control_wires,
        num_work_wires,
        size_bitstring,
        clean,
    ):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceQROM,
            {
                "num_bitstrings": num_bitstrings,
                "num_bit_flips": num_bit_flips,
                "num_control_wires": num_control_wires,
                "num_work_wires": num_work_wires,
                "size_bitstring": size_bitstring,
                "clean": clean,
            },
        )
        assert expected == re.ResourceQROM.resource_rep(
            num_bitstrings, num_bit_flips, num_control_wires, num_work_wires, size_bitstring, clean
        )

    @pytest.mark.parametrize(
        "num_bitstrings, num_bit_flips, num_control_wires, num_work_wires, size_bitstring, clean",
        [(4, 2, 2, 3, 3, True), (4, 5, 2, 3, 3, False), (4, 5, 0, 3, 3, False)],
    )
    def test_resources_from_rep(
        self,
        num_bitstrings,
        num_bit_flips,
        num_control_wires,
        num_work_wires,
        size_bitstring,
        clean,
    ):
        """Test that computing the resources from a compressed representation works"""
        expected = {}
        rep = re.ResourceQROM.resource_rep(
            num_bitstrings, num_bit_flips, num_control_wires, num_work_wires, size_bitstring, clean
        )
        actual = rep.op_type.resources(**rep.params)

        x = re.CompressedResourceOp(re.ResourceX, {})

        if num_control_wires == 0:
            expected[x] = num_bit_flips
            assert actual == expected
            return

        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        hadamard = re.CompressedResourceOp(re.ResourceHadamard, {})

        num_parallel_computations = (num_work_wires + size_bitstring) // size_bitstring
        num_parallel_computations = min(num_parallel_computations, num_bitstrings)

        num_swap_wires = math.floor(math.log2(num_parallel_computations))
        num_select_wires = math.ceil(math.log2(math.ceil(num_bitstrings / (2**num_swap_wires))))
        assert num_swap_wires + num_select_wires <= num_control_wires

        swap_clean_prefactor = 1
        select_clean_prefactor = 1

        if clean:
            expected[hadamard] = 2 * size_bitstring
            swap_clean_prefactor = 4
            select_clean_prefactor = 2

        # SELECT cost:
        expected[cnot] = num_bit_flips  # each unitary in the select is just a CNOT

        multi_x = re.CompressedResourceOp(
            re.ResourceMultiControlledX,
            {
                "num_ctrl_wires": num_select_wires,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
        )

        num_total_ctrl_possibilities = 2**num_select_wires
        expected[multi_x] = select_clean_prefactor * (
            2 * num_total_ctrl_possibilities  # two applications targetting the aux qubit
        )
        num_zero_controls = (2 * num_total_ctrl_possibilities * num_select_wires) // 2
        expected[x] = select_clean_prefactor * (
            num_zero_controls * 2  # conjugate 0 controls on the multi-qubit x gates from above
        )
        # SWAP cost:
        ctrl_swap = re.CompressedResourceOp(re.ResourceCSWAP, {})
        expected[ctrl_swap] = swap_clean_prefactor * ((2**num_swap_wires) - 1) * size_bitstring

        assert actual == expected

    def test_tracking_name(self):
        """Test that the tracking name is correct."""
        assert re.ResourceQROM.tracking_name() == "QROM"


class TestResourceStatePrep:
    """Test the ResourceStatePrep class"""

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources(self, num_wires):
        """Test that the resources are correct"""
        expected = {
            re.CompressedResourceOp(
                re.ResourceMottonenStatePreparation, {"num_wires": num_wires}
            ): 1
        }

        assert re.ResourceStatePrep.resources(num_wires) == expected

    @pytest.mark.parametrize(
        "state, wires",
        [
            (
                np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]),
                range(2),
            ),
        ],
    )
    def test_resource_params(self, state, wires):
        """Test that the resource params are correct"""
        op = re.ResourceStatePrep(state=state, wires=wires, normalize=True)

        assert op.resource_params == {"num_wires": len(wires)}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceStatePrep,
            {"num_wires": num_wires},
        )
        assert expected == re.ResourceStatePrep.resource_rep(num_wires)

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources_from_rep(self, num_wires):
        """Test that computing the resources from a compressed representation works"""
        rep = re.ResourceStatePrep.resource_rep(num_wires)
        actual = rep.op_type.resources(**rep.params)

        expected = {
            re.CompressedResourceOp(
                re.ResourceMottonenStatePreparation, {"num_wires": num_wires}
            ): 1
        }

        assert actual == expected

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_tracking_name(self, num_wires):
        """Test that the tracking name is correct."""
        assert re.ResourceStatePrep.tracking_name(num_wires) == f"StatePrep({num_wires})"


class TestResourceAmplitudeAmplification:
    """Test the ResourceAmplitudeAmplification class"""

    @pytest.mark.parametrize(
        "U_op, U_params, O_op,O_params,iters,num_ref_wires,fixed_point",
        [(re.ResourceHadamard, {}, re.ResourceRX, {}, 5, 3, True)],
    )
    def test_resources(self, U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point):
        """Test that the resources are correct"""
        expected = {}
        reflection = re.ResourceReflection.resource_rep(
            base_class=U_op, base_params=U_params, num_ref_wires=num_ref_wires
        )

        if not fixed_point:
            oracles = re.CompressedResourceOp(O_op, params=O_params)
            expected[oracles] = iters
            expected[reflection] = iters

            assert (
                re.ResourceAmplitudeAmplification.resources(
                    U_op,
                    U_params,
                    O_op,
                    O_params,
                    iters,
                    num_ref_wires,
                    fixed_point,
                )
                == expected
            )
        iters = iters // 2
        ctrl = re.ResourceControlled.resource_rep(
            base_class=O_op,
            base_params=O_params,
            num_ctrl_wires=1,
            num_ctrl_values=0,
            num_work_wires=0,
        )
        phase_shift = re.ResourcePhaseShift.resource_rep()
        hadamard = re.ResourceHadamard.resource_rep()

        expected[ctrl] = iters * 2
        expected[phase_shift] = iters
        expected[hadamard] = iters * 4
        expected[reflection] = iters

        assert (
            re.ResourceAmplitudeAmplification.resources(
                U_op, U_params, O_op, O_params, 5, num_ref_wires, fixed_point
            )
            == expected
        )

    def test_resource_params(self):
        """Test that the resource params are correct"""

        @qml.prod
        def generator(wires):
            for wire in wires:
                qml.Hadamard(wires=wire)

        U = generator(wires=range(3))
        O = qml.FlipSign(2, wires=range(3))

        expected = {
            "U_op": type(U),
            "U_params": {},
            "O_op": type(O),
            "O_params": {},
            "iters": 5,
            "num_ref_wires": 3,
            "fixed_point": True,
        }

        op = re.ResourceAmplitudeAmplification(U=U, O=O, iters=5, fixed_point=True, work_wire=3)

        assert op.resource_params == expected

    @pytest.mark.parametrize(
        "U_op, U_params, O_op,O_params,iters, num_ref_wires,fixed_point",
        [(re.ResourceHadamard, {}, re.ResourceRX, {}, 5, 3, True)],
    )
    def test_resource_rep(self, U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(
            re.ResourceAmplitudeAmplification,
            {
                "U_op": U_op,
                "U_params": U_params,
                "O_op": O_op,
                "O_params": O_params,
                "iters": iters,
                "num_ref_wires": num_ref_wires,
                "fixed_point": fixed_point,
            },
        )
        assert expected == re.ResourceAmplitudeAmplification.resource_rep(
            U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point
        )

    @pytest.mark.parametrize(
        "U_op, U_params, O_op,O_params,iters,num_ref_wires,fixed_point",
        [(re.ResourceHadamard, {}, re.ResourceRX, {}, 5, 3, True)],
    )
    def test_resources_from_rep(
        self, U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point
    ):
        """Test that computing the resources from a compressed representation works"""
        rep = re.ResourceAmplitudeAmplification.resource_rep(
            U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point
        )
        actual = rep.op_type.resources(**rep.params)

        expected = {}
        reflection = re.ResourceReflection.resource_rep(
            base_class=U_op, base_params=U_params, num_ref_wires=num_ref_wires
        )

        if not fixed_point:
            oracles = re.CompressedResourceOp(O_op, params=O_params)
            expected[oracles] = iters
            expected[reflection] = iters

            assert (
                re.ResourceAmplitudeAmplification.resources(
                    U_op,
                    U_params,
                    O_op,
                    O_params,
                    iters,
                    num_ref_wires,
                    fixed_point,
                )
                == expected
            )
        iters = iters // 2
        ctrl = re.ResourceControlled.resource_rep(
            base_class=O_op,
            base_params=O_params,
            num_ctrl_wires=1,
            num_ctrl_values=0,
            num_work_wires=0,
        )
        phase_shift = re.ResourcePhaseShift.resource_rep()
        hadamard = re.ResourceHadamard.resource_rep()

        expected[ctrl] = iters * 2
        expected[phase_shift] = iters
        expected[hadamard] = iters * 4
        expected[reflection] = iters

        assert actual == expected

    @pytest.mark.parametrize(
        "U_op, U_params, O_op,O_params,iters,num_ref_wires,fixed_point",
        [(re.ResourceHadamard, {}, re.ResourceRX, {}, 5, 3, True)],
    )
    def test_tracking_name(self, U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point):
        """Test that the tracking name is correct."""
        assert (
            re.ResourceAmplitudeAmplification.tracking_name(
                U_op, U_params, O_op, O_params, iters, num_ref_wires, fixed_point
            )
            == "AmplitudeAmplification"
        )
