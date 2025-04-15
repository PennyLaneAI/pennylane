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
"""Tests for parametric multi qubit resource operators."""

import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=use-implicit-booleaness-not-comparison,no-self-use,too-many-arguments


class TestMultiRZ:
    """Test the ResourceMultiRZ class."""

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = re.ResourceMultiRZ(0.5, range(num_wires))
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceMultiRZ, {"num_wires": num_wires})
        assert re.ResourceMultiRZ.resource_rep(num_wires) == expected

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resources(self, num_wires):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCNOT.resource_rep(): 2 * (num_wires - 1),
            re.ResourceRZ.resource_rep(): 1,
        }
        assert re.ResourceMultiRZ.resources(num_wires) == expected

    @pytest.mark.parametrize("num_wires", range(1, 10))
    def test_resources_from_rep(self, num_wires):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceMultiRZ(0.5, wires=range(num_wires))
        expected = {
            re.ResourceCNOT.resource_rep(): 2 * (num_wires - 1),
            re.ResourceRZ.resource_rep(): 1,
        }

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected

    @pytest.mark.parametrize("num_wires", range(1, 5))
    def test_adjoint_decomp(self, num_wires):
        """Test that the adjoint decomposition is correct."""
        expected = {re.ResourceMultiRZ.resource_rep(num_wires=num_wires): 1}
        assert re.ResourceMultiRZ.adjoint_resource_decomp(num_wires=num_wires) == expected

        multi_rz = re.ResourceMultiRZ(0.123, wires=range(num_wires))
        multi_rz_dag = re.ResourceAdjoint(multi_rz)

        assert re.get_resources(multi_rz) == re.get_resources(multi_rz_dag)

    ctrl_data = (
        (
            [1],
            [1],
            [],
            {
                re.ResourceCNOT.resource_rep(): 4,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 1, 0, 0): 1,
            },
        ),
        (
            [1],
            [0],
            [],
            {
                re.ResourceControlled.resource_rep(
                    re.ResourceMultiRZ, {"num_wires": 3}, 1, 0, 0
                ): 1,
                re.ResourceX.resource_rep(): 2,
            },
        ),
        (
            [1, 2],
            [1, 1],
            ["w1"],
            {
                re.ResourceCNOT.resource_rep(): 4,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 2, 0, 1): 1,
            },
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            ["w1", "w2"],
            {
                re.ResourceControlled.resource_rep(
                    re.ResourceMultiRZ, {"num_wires": 3}, 3, 0, 2
                ): 1,
                re.ResourceX.resource_rep(): 4,
            },
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, work_wires, expected_res", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op = re.ResourceMultiRZ(1.24, wires=range(5, 8))
        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        if num_ctrl_values != 0:
            with pytest.raises(re.ResourcesNotDefined):
                op.controlled_resource_decomp(
                    num_ctrl_wires, num_ctrl_values, num_work_wires, **op.resource_params
                )
        else:
            assert (
                op.controlled_resource_decomp(
                    num_ctrl_wires, num_ctrl_values, num_work_wires, **op.resource_params
                )
                == expected_res
            )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {re.ResourceMultiRZ.resource_rep(num_wires=4): 1}),
        (2, {re.ResourceMultiRZ.resource_rep(num_wires=4): 1}),
        (3, {re.ResourceMultiRZ.resource_rep(num_wires=4): 1}),
        (4, {re.ResourceMultiRZ.resource_rep(num_wires=4): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = re.ResourceMultiRZ(1.23, wires=range(4))
        assert op.pow_resource_decomp(z, **op.resource_params) == expected_res

        op2 = re.ResourcePow(op, z)
        assert op2.resources(**op2.resource_params) == expected_res


class TestPauliRot:
    """Test the ResourcePauliRot class."""

    pauli_words = ("I", "XYZ", "XXX", "XIYIZIX", "III")

    @pytest.mark.parametrize("pauli_string", pauli_words)
    def test_resource_params(self, pauli_string):
        """Test that the resource params are correct."""
        op = re.ResourcePauliRot(theta=0.5, pauli_word=pauli_string, wires=range(len(pauli_string)))
        assert op.resource_params == {"pauli_string": pauli_string}

    @pytest.mark.parametrize("pauli_string", pauli_words)
    def test_resource_rep(self, pauli_string):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourcePauliRot, {"pauli_string": pauli_string})
        assert re.ResourcePauliRot.resource_rep(pauli_string) == expected

    expected_h_count = (0, 4, 6, 6, 0)
    expected_s_count = (0, 1, 0, 1, 0)
    params = zip(pauli_words, expected_h_count, expected_s_count)

    @pytest.mark.parametrize("pauli_string, expected_h_count, expected_s_count", params)
    def test_resources(self, pauli_string, expected_h_count, expected_s_count):
        """Test that the resources are correct."""
        active_wires = len(pauli_string.replace("I", ""))

        if set(pauli_string) == {"I"}:
            expected = {re.ResourceGlobalPhase.resource_rep(): 1}
        else:
            expected = {
                re.ResourceRZ.resource_rep(): 1,
                re.ResourceCNOT.resource_rep(): 2 * (active_wires - 1),
            }

            if expected_h_count:
                expected[re.ResourceHadamard.resource_rep()] = expected_h_count

            if expected_s_count:
                expected[re.ResourceS.resource_rep()] = expected_s_count
                expected[re.ResourceAdjoint.resource_rep(re.ResourceS, {})] = expected_s_count

        assert re.ResourcePauliRot.resources(pauli_string) == expected

    def test_resources_empty_pauli_string(self):
        """Test that the resources method produces the correct result for an empty pauli string."""
        expected = {re.ResourceGlobalPhase.resource_rep(): 1}
        assert re.ResourcePauliRot.resources(pauli_string="") == expected

    @pytest.mark.parametrize("pauli_string, expected_h_count, expected_s_count", params)
    def test_resources_from_rep(self, pauli_string, expected_h_count, expected_s_count):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourcePauliRot(0.5, pauli_string, wires=range(len(pauli_string)))
        active_wires = len(pauli_string.replace("I", ""))

        if (set(pauli_string) == {"I"}) or (pauli_string == ""):
            expected = {re.ResourceGlobalPhase.resource_rep(): 1}

        else:
            expected = {
                re.ResourceRZ.resource_rep(): 1,
                re.ResourceCNOT.resource_rep(): 2 * (active_wires - 1),
            }

            if expected_h_count:
                expected[re.ResourceHadamard.resource_rep()] = expected_h_count

            if expected_s_count:
                expected[re.ResourceS.resource_rep()] = expected_s_count
                expected[re.ResourceAdjoint.resource_rep(re.ResourceS, {})] = expected_s_count

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected

    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_adjoint_decomp(self, pauli_word):
        """Test that the adjoint decomposition is correct."""
        expected = {re.ResourcePauliRot.resource_rep(pauli_string=pauli_word): 1}
        assert re.ResourcePauliRot.adjoint_resource_decomp(pauli_string=pauli_word) == expected

        op = re.ResourcePauliRot(theta=0.5, pauli_word=pauli_word, wires=range(len(pauli_word)))
        op_dag = re.ResourceAdjoint(op)

        assert re.get_resources(op) == re.get_resources(op_dag)

    ctrl_data = (
        (
            "XXX",
            [1],
            [1],
            [],
            {
                re.ResourceHadamard.resource_rep(): 6,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 1, 0, 0): 1,
                re.ResourceCNOT.resource_rep(): 4,
            },
        ),
        (
            "XXX",
            [1],
            [0],
            [],
            {
                re.ResourceHadamard.resource_rep(): 6,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 1, 1, 0): 1,
                re.ResourceCNOT.resource_rep(): 4,
            },
        ),
        (
            "XXX",
            [1, 2],
            [1, 1],
            ["w1"],
            {
                re.ResourceHadamard.resource_rep(): 6,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 2, 0, 1): 1,
                re.ResourceCNOT.resource_rep(): 4,
            },
        ),
        (
            "XIYIZIX",
            [1],
            [1],
            [],
            {
                re.ResourceHadamard.resource_rep(): 6,
                re.ResourceS.resource_rep(): 1,
                re.ResourceAdjoint.resource_rep(re.ResourceS, {}): 1,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 1, 0, 0): 1,
                re.ResourceCNOT.resource_rep(): 6,
            },
        ),
        (
            "XIYIZIX",
            [1],
            [0],
            [],
            {
                re.ResourceHadamard.resource_rep(): 6,
                re.ResourceS.resource_rep(): 1,
                re.ResourceAdjoint.resource_rep(re.ResourceS, {}): 1,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 1, 1, 0): 1,
                re.ResourceCNOT.resource_rep(): 6,
            },
        ),
        (
            "XIYIZIX",
            [1, 2],
            [1, 1],
            ["w1"],
            {
                re.ResourceHadamard.resource_rep(): 6,
                re.ResourceS.resource_rep(): 1,
                re.ResourceAdjoint.resource_rep(re.ResourceS, {}): 1,
                re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 2, 0, 1): 1,
                re.ResourceCNOT.resource_rep(): 6,
            },
        ),
        (
            "III",
            [1],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceGlobalPhase, {}, 1, 0, 0): 1},
        ),
        (
            "X",
            [1],
            [0],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceRX, {}, 1, 1, 0): 1},
        ),
        (
            "Y",
            [1, 2],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceRY, {}, 2, 0, 1): 1},
        ),
    )

    @pytest.mark.parametrize(
        "pauli_word, ctrl_wires, ctrl_values, work_wires, expected_res", ctrl_data
    )
    def test_resource_controlled(
        self, ctrl_wires, ctrl_values, work_wires, pauli_word, expected_res
    ):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op = re.ResourcePauliRot(
            1.24, pauli_word, wires=list(f"wire_{i}" for i in range(len(pauli_word)))
        )
        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            op.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **op.resource_params
            )
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    @pytest.mark.parametrize("z", range(1, 5))
    @pytest.mark.parametrize("pauli_word", pauli_words)
    def test_pow_decomp(self, z, pauli_word):
        """Test that the pow decomposition is correct."""
        op = re.ResourcePauliRot(theta=0.5, pauli_word=pauli_word, wires=range(len(pauli_word)))
        expected_res = {re.ResourcePauliRot.resource_rep(pauli_string=pauli_word): 1}
        assert op.pow_resource_decomp(z, **op.resource_params) == expected_res

        op2 = re.ResourcePow(op, z)
        assert op2.resources(**op2.resource_params) == expected_res


class TestIsingXX:
    """Test the IsingXX class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingXX(0.5, wires=[0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingXX, {})
        assert re.ResourceIsingXX.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRX.resource_rep(): 1,
        }
        assert re.ResourceIsingXX.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingXX(0.5, wires=[0, 1])
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRX.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingXY:
    """Test the IsingXY class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingXY(0.5, wires=[0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingXY, {})
        assert re.ResourceIsingXY.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
            re.ResourceRX.resource_rep(): 1,
        }
        assert re.ResourceIsingXY.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingXY(0.5, wires=[0, 1])
        expected = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
            re.ResourceRX.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingYY:
    """Test the IsingYY class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingYY(0.5, wires=[0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingYY, {})
        assert re.ResourceIsingYY.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
        }
        assert re.ResourceIsingYY.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingYY(0.5, wires=[0, 1])
        expected = {
            re.ResourceCY.resource_rep(): 2,
            re.ResourceRY.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestIsingZZ:
    """Test the IsingZZ class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceIsingZZ(0.5, wires=[0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceIsingZZ, {})
        assert re.ResourceIsingZZ.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRZ.resource_rep(): 1,
        }
        assert re.ResourceIsingZZ.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourceIsingZZ(0.5, wires=[0, 1])
        expected = {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceRZ.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestPSWAP:
    """Test the PSWAP class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourcePSWAP(0.5, wires=[0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourcePSWAP, {})
        assert re.ResourcePSWAP.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceSWAP.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourcePhaseShift.resource_rep(): 1,
        }
        assert re.ResourcePSWAP.resources() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation and params."""
        op = re.ResourcePSWAP(0.5, wires=[0, 1])
        expected = {
            re.ResourceSWAP.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourcePhaseShift.resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected
