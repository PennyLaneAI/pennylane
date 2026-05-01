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
"""Tests for controlled resource operators."""

from collections import defaultdict

import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta import (
    Allocate,
    Deallocate,
    GateCount,
    mcx_many_clean_aux_resource_decomp,
    mcx_one_clean_aux_resource_decomp,
    mcx_one_dirty_aux_resource_decomp,
    resource_rep,
)

# pylint: disable= no-self-use


def _test_decomp_equal(decomp1, decomp2):
    if len(decomp1) != len(decomp2):
        return False

    for op1, op2 in zip(decomp1, decomp2):
        if isinstance(op1, (Allocate, Deallocate)):
            ops_equal = op1.equal(op2)
        else:
            ops_equal = op1 == op2

        if not ops_equal:
            return False

    return True


class TestLabsCH:
    """Test the Resource CH operation"""

    op = qre.CH(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(qre.Hadamard.resource_rep(), 2),
            GateCount(qre.T.resource_rep(), 1),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
            GateCount(qre.S.resource_rep(), 1),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
            GateCount(qre.CNOT.resource_rep(), 1),
        ]
        assert qre.ch_resource_decomp(**self.op.resource_params) == expected_resources

    def test_resources_estimate(self):
        """Test that correct resources are produced when using estimate."""
        expected_resources = qre.Resources(
            zeroed_wires=0,
            any_state_wires=0,
            algo_wires=2,
            gate_types=defaultdict(
                int,
                {
                    resource_rep(qre.CNOT): 1,
                    resource_rep(qre.T): 2,
                    resource_rep(qre.Z): 2,
                    resource_rep(qre.S): 3,
                    resource_rep(qre.Hadamard): 2,
                },
            ),
        )
        assert qre.estimate(self.op) == expected_resources

    def test_toffoli_based_resources(self):
        """Test that the resources method produces the expected resources when using a Toffoli-based decomposition."""

        expected_resources = [
            Allocate(1),
            GateCount(qre.Hadamard.resource_rep(), 5),
            GateCount(qre.S.resource_rep(), 2),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
            GateCount(resource_rep(qre.Toffoli), 1),
            GateCount(resource_rep(qre.CNOT), 5),
            GateCount(resource_rep(qre.CZ), 1),
            GateCount(resource_rep(qre.X), 4),
            Deallocate(1),
        ]
        result = qre.ch_toffoli_based_resource_decomp(**self.op.resource_params)
        assert _test_decomp_equal(result, expected_resources)


class TestLabsMultiControlledX:
    """Test the resource decompositions for the MultiControlledX operation"""

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, allocated_register, base_decomp",
        (
            (0, 0, None, [GateCount(qre.X.resource_rep())]),
            (0, 1, None, []),
            (1, 0, None, [GateCount(qre.CNOT.resource_rep())]),
            (1, 1, None, [GateCount(qre.CNOT.resource_rep())]),
            (2, 0, None, [GateCount(qre.Toffoli.resource_rep())]),
            (2, 1, None, [GateCount(qre.Toffoli.resource_rep())]),
            (2, 2, None, [GateCount(qre.Toffoli.resource_rep())]),
            (
                5,
                0,
                Allocate(3, state="zero", restored=True),
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 3),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 3),
                    GateCount(qre.Toffoli.resource_rep()),
                ],
            ),
            (
                6,
                3,
                Allocate(4, state="zero", restored=True),
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 4),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 4),
                    GateCount(qre.Toffoli.resource_rep()),
                ],
            ),
        ),
    )
    def test_mcx_many_clean_aux_resource_decomp(
        self, num_ctrl_wires, num_zero_ctrl, allocated_register, base_decomp
    ):
        """Test the MCX decomposition using many clean auxiliary qubits works
        as expected"""
        expected_decomp = []
        if num_ctrl_wires > 0 and num_zero_ctrl > 0:
            expected_decomp.append(GateCount(qre.X.resource_rep(), num_zero_ctrl * 2))

        if allocated_register is None:
            expected_decomp.extend(base_decomp)
        else:
            expected_decomp.append(allocated_register)
            expected_decomp.extend(base_decomp)
            expected_decomp.append(Deallocate(allocated_register=allocated_register))

        actual_decomp = mcx_many_clean_aux_resource_decomp(num_ctrl_wires, num_zero_ctrl)
        assert _test_decomp_equal(actual_decomp, expected_decomp)

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, base_decomp",
        (
            (0, 0, [GateCount(qre.X.resource_rep())]),
            (0, 1, []),
            (1, 0, [GateCount(qre.CNOT.resource_rep())]),
            (1, 1, [GateCount(qre.CNOT.resource_rep())]),
            (2, 0, [GateCount(qre.Toffoli.resource_rep())]),
            (2, 1, [GateCount(qre.Toffoli.resource_rep())]),
            (2, 2, [GateCount(qre.Toffoli.resource_rep())]),
            (5, 0, [GateCount(qre.Toffoli.resource_rep(), 7)]),
            (6, 3, [GateCount(qre.Toffoli.resource_rep(), 9)]),
        ),
    )
    def test_mcx_one_clean_aux_resource_decomp(self, num_ctrl_wires, num_zero_ctrl, base_decomp):
        """Test the MCX decomposition using one clean auxiliary qubit works
        as expected"""
        expected_decomp = []
        if num_ctrl_wires > 0 and num_zero_ctrl > 0:
            expected_decomp.append(GateCount(qre.X.resource_rep(), num_zero_ctrl * 2))

        if num_ctrl_wires <= 2:
            expected_decomp.extend(base_decomp)

        else:
            allocated_register = Allocate(1, state="zero", restored=True)
            expected_decomp.append(allocated_register)
            expected_decomp.extend(base_decomp)
            expected_decomp.append(Deallocate(allocated_register=allocated_register))

        actual_decomp = mcx_one_clean_aux_resource_decomp(num_ctrl_wires, num_zero_ctrl)
        assert _test_decomp_equal(actual_decomp, expected_decomp)

    def test_default_resource_decomp(self):
        """Test the default resource decomp is the decomposition using one clean
        auxiliary qubit"""
        res = qre.estimate(qre.MultiControlledX(6, 3))
        expected_res = qre.Resources(
            zeroed_wires=1,
            algo_wires=7,
            gate_types={
                qre.Toffoli.resource_rep(): 9,  # 2*n - 3
                qre.X.resource_rep(): 6,
            },
        )
        assert res == expected_res

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, base_decomp",
        (
            (0, 0, [GateCount(qre.X.resource_rep())]),
            (0, 1, []),
            (1, 0, [GateCount(qre.CNOT.resource_rep())]),
            (1, 1, [GateCount(qre.CNOT.resource_rep())]),
            (2, 0, [GateCount(qre.Toffoli.resource_rep())]),
            (2, 1, [GateCount(qre.Toffoli.resource_rep())]),
            (2, 2, [GateCount(qre.Toffoli.resource_rep())]),
            (5, 0, [GateCount(qre.Toffoli.resource_rep(), 12)]),
            (6, 3, [GateCount(qre.Toffoli.resource_rep(), 16)]),
        ),
    )
    def test_mcx_one_dirty_aux_resource_decomp(self, num_ctrl_wires, num_zero_ctrl, base_decomp):
        """Test the MCX decomposition using one borrowed auxiliary qubit works
        as expected"""
        expected_decomp = []
        if num_ctrl_wires > 0 and num_zero_ctrl > 0:
            expected_decomp.append(GateCount(qre.X.resource_rep(), num_zero_ctrl * 2))

        if num_ctrl_wires <= 2:
            expected_decomp.extend(base_decomp)

        else:
            allocated_register = Allocate(1, state="any", restored=True)
            expected_decomp.append(allocated_register)
            expected_decomp.extend(base_decomp)
            expected_decomp.append(Deallocate(allocated_register=allocated_register))

        actual_decomp = mcx_one_dirty_aux_resource_decomp(num_ctrl_wires, num_zero_ctrl)
        assert _test_decomp_equal(actual_decomp, expected_decomp)
