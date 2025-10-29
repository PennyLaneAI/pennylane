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
Tests for the state preparation subroutines resource operators.
"""
import pytest

import pennylane.estimator as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.templates.stateprep import AliasSampling, MPSPrep, QROMStatePreparation
from pennylane.estimator.wires_manager import Allocate, Deallocate

# pylint: disable=no-self-use,too-many-arguments


class TestUniformStatePrep:
    """Test the ResourceUniformStatePrep class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            qre.UniformStatePrep(num_states=4, wires=[0, 1, 2])

    @pytest.mark.parametrize(
        "num_states",
        (10, 6, 4),
    )
    def test_resource_params(self, num_states):
        """Test that the resource params are correct."""
        op = qre.UniformStatePrep(num_states)
        assert op.resource_params == {"num_states": num_states}

    @pytest.mark.parametrize(
        "num_states, num_wires",
        (
            (10, 4),
            (6, 3),
            (4, 2),
        ),
    )
    def test_resource_rep(self, num_states, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.UniformStatePrep, num_wires, {"num_states": num_states}
        )
        assert qre.UniformStatePrep.resource_rep(num_states) == expected

    @pytest.mark.parametrize(
        "num_states, expected_res",
        (
            (
                10,
                [
                    GateCount(resource_rep(qre.Hadamard), 10),
                    GateCount(
                        resource_rep(qre.IntegerComparator, {"value": 5, "register_size": 3}),
                        1,
                    ),
                    GateCount(resource_rep(qre.RZ), 2),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    qre.IntegerComparator, {"value": 5, "register_size": 3}
                                )
                            },
                        ),
                        1,
                    ),
                ],
            ),
            (
                6,
                [
                    GateCount(resource_rep(qre.Hadamard), 7),
                    GateCount(
                        resource_rep(qre.IntegerComparator, {"value": 3, "register_size": 2}),
                        1,
                    ),
                    GateCount(resource_rep(qre.RZ), 2),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    qre.IntegerComparator, {"value": 3, "register_size": 2}
                                )
                            },
                        ),
                        1,
                    ),
                ],
            ),
            (
                4,
                [
                    GateCount(resource_rep(qre.Hadamard), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_states, expected_res):
        """Test that the resources are correct."""
        assert qre.UniformStatePrep.resource_decomp(num_states) == expected_res


class TestAliasSampling:
    """Test the ResourceAliasSampling class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 3"):
            qre.AliasSampling(num_coeffs=10, wires=[0, 1, 2])

    @pytest.mark.parametrize(
        "num_coeffs, precision",
        (
            (10, None),
            (6, None),
            (4, 1e-6),
        ),
    )
    def test_resource_params(self, num_coeffs, precision):
        """Test that the resource params are correct."""
        op = qre.AliasSampling(num_coeffs, precision=precision)
        assert op.resource_params == {"num_coeffs": num_coeffs, "precision": precision}

    @pytest.mark.parametrize(
        "num_coeffs, precision, num_wires",
        (
            (10, None, 4),
            (6, None, 3),
            (4, 1e-6, 2),
        ),
    )
    def test_resource_rep(self, num_coeffs, precision, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.AliasSampling,
            num_wires,
            {"num_coeffs": num_coeffs, "precision": precision},
        )
        assert qre.AliasSampling.resource_rep(num_coeffs, precision) == expected

    @pytest.mark.parametrize(
        "num_coeffs, precision, expected_res",
        (
            (
                10,
                None,
                [
                    Allocate(65),
                    GateCount(resource_rep(qre.UniformStatePrep, {"num_states": 10})),
                    GateCount(resource_rep(qre.Hadamard), 30),
                    GateCount(
                        resource_rep(qre.QROM, {"num_bitstrings": 10, "size_bitstring": 34}),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.RegisterComparator,
                            {"first_register": 30, "second_register": 30},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.CSWAP), 4),
                ],
            ),
            (
                6,
                None,
                [
                    Allocate(64),
                    GateCount(resource_rep(qre.UniformStatePrep, {"num_states": 6})),
                    GateCount(resource_rep(qre.Hadamard), 30),
                    GateCount(
                        resource_rep(qre.QROM, {"num_bitstrings": 6, "size_bitstring": 33}),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.RegisterComparator,
                            {"first_register": 30, "second_register": 30},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.CSWAP), 3),
                ],
            ),
            (
                4,
                1e-6,
                [
                    Allocate(43),
                    GateCount(resource_rep(qre.UniformStatePrep, {"num_states": 4})),
                    GateCount(resource_rep(qre.Hadamard), 20),
                    GateCount(
                        resource_rep(qre.QROM, {"num_bitstrings": 4, "size_bitstring": 22}),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.RegisterComparator,
                            {"first_register": 20, "second_register": 20},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.CSWAP), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_coeffs, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[AliasSampling]
            assert qre.AliasSampling.resource_decomp(num_coeffs, **kwargs) == expected_res
        else:
            assert qre.AliasSampling.resource_decomp(num_coeffs, precision) == expected_res


class TestMPSPrep:
    """Tests for the ResourceMPSPrep template"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 10 wires, got 3"):
            qre.MPSPrep(num_mps_matrices=10, max_bond_dim=2, wires=[0, 1, 2])

    @pytest.mark.parametrize(
        "num_mps, bond_dim, precision",
        (
            (10, 2, None),
            (8, 5, 1e-3),
            (12, 12, 1e-4),
        ),
    )
    def test_resource_params(self, num_mps, bond_dim, precision):
        """Test that the resource params are correct."""
        if precision is None:
            op = qre.MPSPrep(num_mps, bond_dim)
        else:
            op = qre.MPSPrep(num_mps, bond_dim, precision)

        assert op.resource_params == {
            "num_mps_matrices": num_mps,
            "max_bond_dim": bond_dim,
            "precision": precision,
        }

    @pytest.mark.parametrize(
        "num_mps, bond_dim, precision",
        (
            (10, 2, None),
            (8, 5, 1e-3),
            (12, 12, 1e-4),
        ),
    )
    def test_resource_rep(self, num_mps, bond_dim, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.MPSPrep,
            num_mps,
            {
                "num_mps_matrices": num_mps,
                "max_bond_dim": bond_dim,
                "precision": precision,
            },
        )

        assert (
            qre.MPSPrep.resource_rep(
                num_mps_matrices=num_mps, max_bond_dim=bond_dim, precision=precision
            )
            == expected
        )

    @pytest.mark.parametrize(
        "num_mps, bond_dim, precision, expected_res",
        (
            (
                6,
                4,
                None,
                [
                    Allocate(2),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=2, precision=1e-9)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=2, precision=1e-9)),
                    Deallocate(2),
                ],
            ),
            (
                8,
                5,
                1e-3,
                [
                    Allocate(3),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=2, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-3)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=2, precision=1e-3)),
                    Deallocate(3),
                ],
            ),
            (
                10,
                32,
                1e-4,
                [
                    Allocate(5),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=2, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=4, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=5, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=6, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=6, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=5, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=4, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=3, precision=1e-4)),
                    GateCount(qre.QubitUnitary.resource_rep(num_wires=2, precision=1e-4)),
                    Deallocate(5),
                ],
            ),
        ),
    )
    def test_resources(self, num_mps, bond_dim, precision, expected_res):
        "Test that the resources are correct."
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[MPSPrep]
            actual = qre.MPSPrep.resource_decomp(
                num_mps_matrices=num_mps, max_bond_dim=bond_dim, **kwargs
            )
            assert actual == expected_res

        else:
            actual = qre.MPSPrep.resource_decomp(
                num_mps_matrices=num_mps,
                max_bond_dim=bond_dim,
                precision=precision,
            )
            assert actual == expected_res

        assert qre.MPSPrep

    @pytest.mark.parametrize(
        "num_mps, bond_dim, precision, expected_name",
        (
            (10, 2, None, "MPSPrep(10, 2, None)"),
            (8, 5, 1e-3, "MPSPrep(8, 5, 0.001)"),
            (12, 12, 1e-4, "MPSPrep(12, 12, 0.0001)"),
        ),
    )
    def test_tracking_name(self, num_mps, bond_dim, precision, expected_name):
        """Test that the tracking name is as expected."""
        assert qre.MPSPrep.tracking_name(num_mps, bond_dim, precision) == expected_name


class TestQROMStatePrep:
    """Tests for the ResourceQROMStateprep template"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 10 wires, got 3"):
            qre.QROMStatePreparation(num_state_qubits=10, wires=[0, 1, 2])

    @pytest.mark.parametrize(
        "num_state_qubits, precision, positive_and_real, selswap_depths",
        (
            (10, None, None, None),
            (7, 1e-5, False, 1),
            (5, 1e-4, False, 2),
            (5, 1e-4, True, [1, 2, 2, 1, 4]),
            (5, 1e-4, False, [1, 2, 2, 1, 4, 4]),
            (8, 1e-5, True, [1, None, 2, 4, None, None, 2, 8]),
        ),
    )
    def test_resource_params(self, num_state_qubits, precision, positive_and_real, selswap_depths):
        """Test that the resource params are as expected"""
        if all((_ is None for _ in (precision, positive_and_real, selswap_depths))):
            op = qre.QROMStatePreparation(num_state_qubits)  # check default values
            expected_params = {
                "num_state_qubits": num_state_qubits,
                "precision": None,
                "positive_and_real": False,
                "selswap_depths": 1,
            }
        else:
            op = qre.QROMStatePreparation(
                num_state_qubits, precision, positive_and_real, selswap_depths
            )
            expected_params = {
                "num_state_qubits": num_state_qubits,
                "precision": precision,
                "positive_and_real": positive_and_real,
                "selswap_depths": selswap_depths,
            }

        assert op.resource_params == expected_params

    @pytest.mark.parametrize(
        "num_state_qubits, precision, positive_and_real, selswap_depths",
        (
            (10, None, None, None),
            (7, 1e-5, False, 1),
            (5, 1e-4, False, 2),
            (5, 1e-4, True, [1, 2, 2, 1, 4]),
            (5, 1e-4, False, [1, 2, 2, 1, 4, 4]),
            (8, 1e-5, True, [1, None, 2, 4, None, None, 2, 8]),
        ),
    )
    def test_resource_rep(self, num_state_qubits, precision, positive_and_real, selswap_depths):
        """Test that the resource rep is constructed as expected"""
        if all((_ is None for _ in (precision, positive_and_real, selswap_depths))):
            actual_resource_rep = qre.QROMStatePreparation.resource_rep(num_state_qubits)
            expected = qre.CompressedResourceOp(
                qre.QROMStatePreparation,
                num_state_qubits,
                {
                    "num_state_qubits": num_state_qubits,
                    "precision": None,
                    "positive_and_real": False,
                    "selswap_depths": 1,
                },
            )
        else:
            actual_resource_rep = qre.QROMStatePreparation.resource_rep(
                num_state_qubits,
                precision,
                positive_and_real,
                selswap_depths,
            )
            expected = qre.CompressedResourceOp(
                qre.QROMStatePreparation,
                num_state_qubits,
                {
                    "num_state_qubits": num_state_qubits,
                    "precision": precision,
                    "positive_and_real": positive_and_real,
                    "selswap_depths": selswap_depths,
                },
            )

        assert actual_resource_rep == expected

    def test_error_selswap_depths(self):
        """Test that an error is raised if incompatible inputs are provided for selswap_depths"""

        # test incorrect type
        sel_swapdepths = "my selswap depth"

        with pytest.raises(
            TypeError, match="`select_swap_depths` must be an integer, None or iterable"
        ):
            qre.QROMStatePreparation(10, select_swap_depths=sel_swapdepths)

        with pytest.raises(
            TypeError, match="`selswap_depths` must be an integer, None or iterable"
        ):
            qre.QROMStatePreparation.resource_rep(10, selswap_depths=sel_swapdepths)

        # test incorrect size
        sel_swapdepths = [1]
        with pytest.raises(
            ValueError, match="Expected the length of `select_swap_depths` to be 11, got 1"
        ):
            qre.QROMStatePreparation(10, select_swap_depths=sel_swapdepths)

        with pytest.raises(
            ValueError, match="Expected the length of `selswap_depths` to be 11, got 1"
        ):
            qre.QROMStatePreparation.resource_rep(10, selswap_depths=sel_swapdepths)

    # Computed resources by hand:
    @pytest.mark.parametrize(
        "num_state_qubits, precision, positive_and_real, selswap_depths, expected_res",
        (
            (
                5,
                None,
                False,
                1,
                [
                    Allocate(32),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=32,
                            num_bit_flips=16,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=32,
                                num_bit_flips=16,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=32,
                            num_bit_flips=32,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=32,
                                num_bit_flips=32,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=32,
                            num_bit_flips=64,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=32,
                                num_bit_flips=64,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=32,
                            num_bit_flips=128,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=32,
                                num_bit_flips=128,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=32,
                            num_bit_flips=256,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=32,
                                num_bit_flips=256,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(32),
                            1,
                            0,
                        ),
                        5,
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 64),
                    GateCount(qre.S.resource_rep(), 32),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        32,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=32,
                            size_bitstring=32,
                            num_bit_flips=512,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=32,
                                size_bitstring=32,
                                num_bit_flips=512,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(32),
                            1,
                            0,
                        ),
                    ),
                    Deallocate(32),
                ],
            ),
            (
                4,
                1e-5,
                False,
                1,
                [
                    Allocate(19),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=19,
                            num_bit_flips=9,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=19,
                                num_bit_flips=9,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=19,
                            num_bit_flips=19,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=19,
                                num_bit_flips=19,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=19,
                            num_bit_flips=38,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=19,
                                num_bit_flips=38,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=19,
                            num_bit_flips=76,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=19,
                                num_bit_flips=76,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(19),
                            1,
                            0,
                        ),
                        4,
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 38),
                    GateCount(qre.S.resource_rep(), 19),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        19,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=19,
                            num_bit_flips=152,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=19,
                                num_bit_flips=152,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(19),
                            1,
                            0,
                        ),
                    ),
                    Deallocate(19),
                ],
            ),
            (
                3,
                1e-4,
                False,
                2,
                [
                    Allocate(15),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                        3,
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 30),
                    GateCount(qre.S.resource_rep(), 15),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        15,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                    ),
                    Deallocate(15),
                ],
            ),
            (
                3,
                1e-4,
                True,
                [1, 2, 2],
                [
                    Allocate(15),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                        3,
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 30),
                    GateCount(qre.S.resource_rep(), 15),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        15,
                    ),
                    Deallocate(15),
                ],
            ),
            (
                3,
                1e-4,
                False,
                [None, 1, None, 4],
                [
                    Allocate(15),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            restored=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                restored=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            restored=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                restored=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                        3,
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 30),
                    GateCount(qre.S.resource_rep(), 15),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        15,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            restored=False,
                            select_swap_depth=4,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                restored=False,
                                select_swap_depth=4,
                            )
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                    ),
                    Deallocate(15),
                ],
            ),
        ),
    )
    def test_default_resources(
        self, num_state_qubits, precision, positive_and_real, selswap_depths, expected_res
    ):
        """Test that the resources are as expected for the default decomposition"""

        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[QROMStatePreparation]
            actual_resources = qre.QROMStatePreparation.resource_decomp(
                num_state_qubits=num_state_qubits,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
                **kwargs,
            )
        else:
            actual_resources = qre.QROMStatePreparation.resource_decomp(
                num_state_qubits=num_state_qubits,
                precision=precision,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
            )

        assert actual_resources == expected_res

    @pytest.mark.parametrize(
        "num_state_qubits, precision, positive_and_real, selswap_depths, expected_res",
        (
            (
                5,
                None,
                False,
                1,
                [
                    Allocate(32),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=32,
                            num_bit_flips=16,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=32,
                                num_bit_flips=16,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=32,
                            num_bit_flips=32,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=32,
                                num_bit_flips=32,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=32,
                            num_bit_flips=64,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=32,
                                num_bit_flips=64,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=32,
                            num_bit_flips=128,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=32,
                                num_bit_flips=128,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=32,
                            num_bit_flips=256,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=32,
                                num_bit_flips=256,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.CRY.resource_rep(),
                        160,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=32,
                            size_bitstring=32,
                            num_bit_flips=512,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=32,
                                size_bitstring=32,
                                num_bit_flips=512,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.PhaseShift.resource_rep(),
                        32,
                    ),
                    Deallocate(32),
                ],
            ),
            (
                4,
                1e-5,
                False,
                1,
                [
                    Allocate(19),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=19,
                            num_bit_flips=9,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=19,
                                num_bit_flips=9,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=19,
                            num_bit_flips=19,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=19,
                                num_bit_flips=19,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=19,
                            num_bit_flips=38,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=19,
                                num_bit_flips=38,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=19,
                            num_bit_flips=76,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=19,
                                num_bit_flips=76,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.CRY.resource_rep(),
                        76,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=19,
                            num_bit_flips=152,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=19,
                                num_bit_flips=152,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.PhaseShift.resource_rep(),
                        19,
                    ),
                    Deallocate(19),
                ],
            ),
            (
                3,
                1e-4,
                False,
                2,
                [
                    Allocate(15),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.CRY.resource_rep(),
                        45,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.PhaseShift.resource_rep(),
                        15,
                    ),
                    Deallocate(15),
                ],
            ),
            (
                3,
                1e-4,
                True,
                [1, 2, 2],
                [
                    Allocate(15),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            restored=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                restored=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        qre.CRY.resource_rep(),
                        45,
                    ),
                    Deallocate(15),
                ],
            ),
            (
                3,
                1e-4,
                False,
                [None, 1, None, 4],
                [
                    Allocate(15),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            restored=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                restored=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            restored=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                restored=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            restored=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                restored=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        qre.CRY.resource_rep(),
                        45,
                    ),
                    GateCount(
                        qre.QROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            restored=False,
                            select_swap_depth=4,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                restored=False,
                                select_swap_depth=4,
                            )
                        )
                    ),
                    GateCount(
                        qre.PhaseShift.resource_rep(),
                        15,
                    ),
                    Deallocate(15),
                ],
            ),
        ),
    )
    def test_control_ry_resources(
        self, num_state_qubits, precision, positive_and_real, selswap_depths, expected_res
    ):
        """Test that the resources are as expected for the controlled-RY decomposition"""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[QROMStatePreparation]
            actual_resources = qre.QROMStatePreparation.controlled_ry_resource_decomp(
                num_state_qubits=num_state_qubits,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
                **kwargs,
            )
        else:
            actual_resources = qre.QROMStatePreparation.controlled_ry_resource_decomp(
                num_state_qubits=num_state_qubits,
                precision=precision,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
            )

        assert actual_resources == expected_res


class TestPrepTHC:
    """Test the PrepTHC class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        ch = qre.THCHamiltonian(4, 10)
        with pytest.raises(ValueError, match="Expected 8 wires, got 3"):
            qre.PrepTHC(ch, wires=[0, 1, 2])

    @pytest.mark.parametrize(
        "thc_ham, coeff_prec, selswap_depth",
        (
            (qre.THCHamiltonian(58, 160), 13, 1),
            (qre.THCHamiltonian(10, 50), None, None),
            (qre.THCHamiltonian(4, 20), None, 2),
        ),
    )
    def test_resource_params(self, thc_ham, coeff_prec, selswap_depth):
        """Test that the resource params for PrepTHC are correct."""
        op = qre.PrepTHC(thc_ham, coeff_prec, selswap_depth)
        assert op.resource_params == {
            "thc_ham": thc_ham,
            "coeff_precision": coeff_prec,
            "select_swap_depth": selswap_depth,
        }

    @pytest.mark.parametrize(
        "thc_ham, coeff_prec, selswap_depth, num_wires",
        (
            (qre.THCHamiltonian(58, 160), 13, 1, 16),
            (qre.THCHamiltonian(10, 50), None, None, 12),
            (qre.THCHamiltonian(4, 20), None, 2, 10),
        ),
    )
    def test_resource_rep(self, thc_ham, coeff_prec, selswap_depth, num_wires):
        """Test that the compressed representation of PrepTHC is correct."""
        expected = qre.CompressedResourceOp(
            qre.PrepTHC,
            num_wires,
            {
                "thc_ham": thc_ham,
                "coeff_precision": coeff_prec,
                "select_swap_depth": selswap_depth,
            },
        )
        assert qre.PrepTHC.resource_rep(thc_ham, coeff_prec, selswap_depth) == expected

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 33 in https://arxiv.org/abs/2011.03494.
    # The numbers were adjusted slightly to account for a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, coeff_prec, selswap_depth, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                13,
                1,
                {"algo_wires": 16, "auxiliary_wires": 86, "toffoli_gates": 13156},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                {"algo_wires": 12, "auxiliary_wires": 174, "toffoli_gates": 579},
            ),
            (
                qre.THCHamiltonian(4, 20),
                None,
                2,
                {"algo_wires": 10, "auxiliary_wires": 109, "toffoli_gates": 279},
            ),
        ),
    )
    def test_resources(self, thc_ham, coeff_prec, selswap_depth, expected_res):
        """Test that the resources for PrepTHC are correct."""

        prep_cost = qre.estimate(
            qre.PrepTHC(thc_ham, coeff_precision=coeff_prec, select_swap_depth=selswap_depth)
        )
        assert prep_cost.algo_wires == expected_res["algo_wires"]
        assert prep_cost.zeroed_wires + prep_cost.any_state_wires == expected_res["auxiliary_wires"]
        assert prep_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    def test_incompatible_hamiltonian(self):
        """Test that an error is raised for incompatible Hamiltonians."""
        with pytest.raises(TypeError, match="Unsupported Hamiltonian representation for PrepTHC."):
            qre.PrepTHC(qre.CDFHamiltonian(58, 160))

        with pytest.raises(TypeError, match="Unsupported Hamiltonian representation for PrepTHC."):
            qre.PrepTHC.resource_rep(qre.CDFHamiltonian(58, 160))

    def test_type_error_precision(self):
        "Test that an error is raised when wrong type is provided for precision."
        with pytest.raises(
            TypeError,
            match=f"`coeff_precision` must be an integer, but type {type(2.5)} was provided.",
        ):
            qre.PrepTHC(qre.THCHamiltonian(58, 160), coeff_precision=2.5)

        with pytest.raises(
            TypeError,
            match=f"`coeff_precision` must be an integer, but type {type(2.5)} was provided.",
        ):
            qre.PrepTHC.resource_rep(qre.THCHamiltonian(58, 160), coeff_precision=2.5)
