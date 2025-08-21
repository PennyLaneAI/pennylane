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

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation import AllocWires, FreeWires, GateCount, resource_rep

# pylint: disable=no-self-use,too-many-arguments


class TestUniformStatePrep:
    """Test the ResourceUniformStatePrep class."""

    @pytest.mark.parametrize(
        "num_states",
        (10, 6, 4),
    )
    def test_resource_params(self, num_states):
        """Test that the resource params are correct."""
        op = plre.ResourceUniformStatePrep(num_states)
        assert op.resource_params == {"num_states": num_states}

    @pytest.mark.parametrize(
        "num_states",
        (10, 6, 4),
    )
    def test_resource_rep(self, num_states):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceUniformStatePrep, {"num_states": num_states}
        )
        assert plre.ResourceUniformStatePrep.resource_rep(num_states) == expected

    @pytest.mark.parametrize(
        "num_states, expected_res",
        (
            (
                10,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 10),
                    GateCount(
                        resource_rep(
                            plre.ResourceIntegerComparator, {"value": 5, "register_size": 3}
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceRZ), 2),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceIntegerComparator, {"value": 5, "register_size": 3}
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
                    GateCount(resource_rep(plre.ResourceHadamard), 7),
                    GateCount(
                        resource_rep(
                            plre.ResourceIntegerComparator, {"value": 3, "register_size": 2}
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceRZ), 2),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceIntegerComparator, {"value": 3, "register_size": 2}
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
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_states, expected_res):
        """Test that the resources are correct."""
        print(plre.ResourceUniformStatePrep.resource_decomp(num_states))
        assert plre.ResourceUniformStatePrep.resource_decomp(num_states) == expected_res


class TestAliasSampling:
    """Test the ResourceAliasSampling class."""

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
        op = plre.ResourceAliasSampling(num_coeffs, precision=precision)
        assert op.resource_params == {"num_coeffs": num_coeffs, "precision": precision}

    @pytest.mark.parametrize(
        "num_coeffs, precision",
        (
            (10, None),
            (6, None),
            (4, 1e-6),
        ),
    )
    def test_resource_rep(self, num_coeffs, precision):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceAliasSampling, {"num_coeffs": num_coeffs, "precision": precision}
        )
        assert plre.ResourceAliasSampling.resource_rep(num_coeffs, precision) == expected

    @pytest.mark.parametrize(
        "num_coeffs, precision, expected_res",
        (
            (
                10,
                None,
                [
                    AllocWires(65),
                    GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": 10})),
                    GateCount(resource_rep(plre.ResourceHadamard), 30),
                    GateCount(
                        resource_rep(
                            plre.ResourceQROM, {"num_bitstrings": 10, "size_bitstring": 34}
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceRegisterComparator,
                            {"first_register": 30, "second_register": 30},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceCSWAP), 4),
                ],
            ),
            (
                6,
                None,
                [
                    AllocWires(64),
                    GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": 6})),
                    GateCount(resource_rep(plre.ResourceHadamard), 30),
                    GateCount(
                        resource_rep(
                            plre.ResourceQROM, {"num_bitstrings": 6, "size_bitstring": 33}
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceRegisterComparator,
                            {"first_register": 30, "second_register": 30},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceCSWAP), 3),
                ],
            ),
            (
                4,
                1e-6,
                [
                    AllocWires(43),
                    GateCount(resource_rep(plre.ResourceUniformStatePrep, {"num_states": 4})),
                    GateCount(resource_rep(plre.ResourceHadamard), 20),
                    GateCount(
                        resource_rep(
                            plre.ResourceQROM, {"num_bitstrings": 4, "size_bitstring": 22}
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceRegisterComparator,
                            {"first_register": 20, "second_register": 20},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceCSWAP), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_coeffs, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = {"precision_alias_sampling": 1e-9}
            assert (
                plre.ResourceAliasSampling.resource_decomp(num_coeffs, precision, config=config)
                == expected_res
            )
        else:
            assert plre.ResourceAliasSampling.resource_decomp(num_coeffs, precision) == expected_res


class TestMPSPrep:
    """Tests for the ResourceMPSPrep template"""

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
            op = plre.ResourceMPSPrep(num_mps, bond_dim)
        else:
            op = plre.ResourceMPSPrep(num_mps, bond_dim, precision)

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
        expected = plre.CompressedResourceOp(
            plre.ResourceMPSPrep,
            {
                "num_mps_matrices": num_mps,
                "max_bond_dim": bond_dim,
                "precision": precision,
            },
        )

        assert (
            plre.ResourceMPSPrep.resource_rep(
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
                    AllocWires(2),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=2, precision=1e-9)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-9)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=2, precision=1e-9)),
                    FreeWires(2),
                ],
            ),
            (
                8,
                5,
                1e-3,
                [
                    AllocWires(3),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=2, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=4, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-3)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=2, precision=1e-3)),
                    FreeWires(3),
                ],
            ),
            (
                10,
                32,
                1e-4,
                [
                    AllocWires(5),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=2, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=4, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=5, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=6, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=6, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=5, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=4, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=3, precision=1e-4)),
                    GateCount(plre.ResourceQubitUnitary.resource_rep(num_wires=2, precision=1e-4)),
                    FreeWires(5),
                ],
            ),
        ),
    )
    def test_resources(self, num_mps, bond_dim, precision, expected_res):
        "Test that the resources are correct."
        if precision is None:
            config = {"precision_mps_prep": 1e-9}
            actual = plre.ResourceMPSPrep.resource_decomp(
                num_mps_matrices=num_mps,
                max_bond_dim=bond_dim,
                config=config,
            )

            print(actual)
            assert actual == expected_res

        else:
            actual = plre.ResourceMPSPrep.resource_decomp(
                num_mps_matrices=num_mps,
                max_bond_dim=bond_dim,
                precision=precision,
            )
            for op in actual:
                if isinstance(op, GateCount):
                    print(op.gate)
                else:
                    print(op)

            print(actual)
            assert actual == expected_res

        assert plre.ResourceMPSPrep

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
        assert plre.ResourceMPSPrep.tracking_name(num_mps, bond_dim, precision) == expected_name


class TestQROMStatePrep:
    """Tests for the ResourceQROMStateprep template"""

    # {"num_state_qubits", "precision", "positive_and_real", "selswap_depths"}

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
            op = plre.ResourceQROMStatePreparation(num_state_qubits)  # check default values
            expected_params = {
                "num_state_qubits": num_state_qubits,
                "precision": None,
                "positive_and_real": False,
                "selswap_depths": 1,
            }
        else:
            op = plre.ResourceQROMStatePreparation(
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
            actual_resource_rep = plre.ResourceQROMStatePreparation.resource_rep(num_state_qubits)
            expected = plre.CompressedResourceOp(
                plre.ResourceQROMStatePreparation,
                {
                    "num_state_qubits": num_state_qubits,
                    "precision": None,
                    "positive_and_real": False,
                    "selswap_depths": 1,
                },
            )
        else:
            actual_resource_rep = plre.ResourceQROMStatePreparation.resource_rep(
                num_state_qubits,
                precision,
                positive_and_real,
                selswap_depths,
            )
            expected = plre.CompressedResourceOp(
                plre.ResourceQROMStatePreparation,
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
            plre.ResourceQROMStatePreparation(10, select_swap_depths=sel_swapdepths)

        with pytest.raises(
            TypeError, match="`selswap_depths` must be an integer, None or iterable"
        ):
            plre.ResourceQROMStatePreparation.resource_rep(10, selswap_depths=sel_swapdepths)

        # test incorrect size
        sel_swapdepths = [1]
        with pytest.raises(
            ValueError, match="Expected the length of `select_swap_depths` to be 11, got 1"
        ):
            plre.ResourceQROMStatePreparation(10, select_swap_depths=sel_swapdepths)

        with pytest.raises(
            ValueError, match="Expected the length of `selswap_depths` to be 11, got 1"
        ):
            plre.ResourceQROMStatePreparation.resource_rep(10, selswap_depths=sel_swapdepths)

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
                    AllocWires(32),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=32,
                            num_bit_flips=16,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=32,
                                num_bit_flips=16,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=32,
                            num_bit_flips=32,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=32,
                                num_bit_flips=32,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=32,
                            num_bit_flips=64,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=32,
                                num_bit_flips=64,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=32,
                            num_bit_flips=128,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=32,
                                num_bit_flips=128,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=32,
                            num_bit_flips=256,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=32,
                                num_bit_flips=256,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(32),
                            1,
                            0,
                        ),
                        5,
                    ),
                    GateCount(plre.ResourceHadamard.resource_rep(), 64),
                    GateCount(plre.ResourceS.resource_rep(), 32),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceS.resource_rep()),
                        32,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=32,
                            size_bitstring=32,
                            num_bit_flips=512,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=32,
                                size_bitstring=32,
                                num_bit_flips=512,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(32),
                            1,
                            0,
                        ),
                    ),
                    FreeWires(32),
                ],
            ),
            (
                4,
                1e-5,
                False,
                1,
                [
                    AllocWires(19),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=19,
                            num_bit_flips=9,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=19,
                                num_bit_flips=9,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=19,
                            num_bit_flips=19,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=19,
                                num_bit_flips=19,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=19,
                            num_bit_flips=38,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=19,
                                num_bit_flips=38,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=19,
                            num_bit_flips=76,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=19,
                                num_bit_flips=76,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(19),
                            1,
                            0,
                        ),
                        4,
                    ),
                    GateCount(plre.ResourceHadamard.resource_rep(), 38),
                    GateCount(plre.ResourceS.resource_rep(), 19),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceS.resource_rep()),
                        19,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=19,
                            num_bit_flips=152,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=19,
                                num_bit_flips=152,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(19),
                            1,
                            0,
                        ),
                    ),
                    FreeWires(19),
                ],
            ),
            (
                3,
                1e-4,
                False,
                2,
                [
                    AllocWires(15),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                        3,
                    ),
                    GateCount(plre.ResourceHadamard.resource_rep(), 30),
                    GateCount(plre.ResourceS.resource_rep(), 15),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceS.resource_rep()),
                        15,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                    ),
                    FreeWires(15),
                ],
            ),
            (
                3,
                1e-4,
                True,
                [1, 2, 2],
                [
                    AllocWires(15),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                        3,
                    ),
                    GateCount(plre.ResourceHadamard.resource_rep(), 30),
                    GateCount(plre.ResourceS.resource_rep(), 15),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceS.resource_rep()),
                        15,
                    ),
                    FreeWires(15),
                ],
            ),
            (
                3,
                1e-4,
                False,
                [None, 1, None, 4],
                [
                    AllocWires(15),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            clean=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                clean=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            clean=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                clean=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                        3,
                    ),
                    GateCount(plre.ResourceHadamard.resource_rep(), 30),
                    GateCount(plre.ResourceS.resource_rep(), 15),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceS.resource_rep()),
                        15,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            clean=False,
                            select_swap_depth=4,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                clean=False,
                                select_swap_depth=4,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(15),
                            1,
                            0,
                        ),
                    ),
                    FreeWires(15),
                ],
            ),
        ),
    )
    def test_default_resources(
        self, num_state_qubits, precision, positive_and_real, selswap_depths, expected_res
    ):
        """Test that the resources are as expected for the default decomposition"""
        config = {"precision_qrom_state_prep": 1e-9}
        actual_resources = plre.ResourceQROMStatePreparation.resource_decomp(
            num_state_qubits=num_state_qubits,
            precision=precision,
            positive_and_real=positive_and_real,
            selswap_depths=selswap_depths,
            config=config,
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
                    AllocWires(32),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=32,
                            num_bit_flips=16,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=32,
                                num_bit_flips=16,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=32,
                            num_bit_flips=32,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=32,
                                num_bit_flips=32,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=32,
                            num_bit_flips=64,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=32,
                                num_bit_flips=64,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=32,
                            num_bit_flips=128,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=32,
                                num_bit_flips=128,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=32,
                            num_bit_flips=256,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=32,
                                num_bit_flips=256,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceCRY.resource_rep(),
                        160,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=32,
                            size_bitstring=32,
                            num_bit_flips=512,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=32,
                                size_bitstring=32,
                                num_bit_flips=512,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourcePhaseShift.resource_rep(),
                        32,
                    ),
                    FreeWires(32),
                ],
            ),
            (
                4,
                1e-5,
                False,
                1,
                [
                    AllocWires(19),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=19,
                            num_bit_flips=9,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=19,
                                num_bit_flips=9,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=19,
                            num_bit_flips=19,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=19,
                                num_bit_flips=19,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=19,
                            num_bit_flips=38,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=19,
                                num_bit_flips=38,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=19,
                            num_bit_flips=76,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=19,
                                num_bit_flips=76,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceCRY.resource_rep(),
                        76,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=16,
                            size_bitstring=19,
                            num_bit_flips=152,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=19,
                                num_bit_flips=152,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourcePhaseShift.resource_rep(),
                        19,
                    ),
                    FreeWires(19),
                ],
            ),
            (
                3,
                1e-4,
                False,
                2,
                [
                    AllocWires(15),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceCRY.resource_rep(),
                        45,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourcePhaseShift.resource_rep(),
                        15,
                    ),
                    FreeWires(15),
                ],
            ),
            (
                3,
                1e-4,
                True,
                [1, 2, 2],
                [
                    AllocWires(15),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            clean=False,
                            select_swap_depth=2,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                clean=False,
                                select_swap_depth=2,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceCRY.resource_rep(),
                        45,
                    ),
                    FreeWires(15),
                ],
            ),
            (
                3,
                1e-4,
                False,
                [None, 1, None, 4],
                [
                    AllocWires(15),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=1,
                            size_bitstring=15,
                            num_bit_flips=7,
                            clean=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                clean=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=2,
                            size_bitstring=15,
                            num_bit_flips=15,
                            clean=False,
                            select_swap_depth=1,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                clean=False,
                                select_swap_depth=1,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=4,
                            size_bitstring=15,
                            num_bit_flips=30,
                            clean=False,
                            select_swap_depth=None,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                clean=False,
                                select_swap_depth=None,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourceCRY.resource_rep(),
                        45,
                    ),
                    GateCount(
                        plre.ResourceQROM.resource_rep(
                            num_bitstrings=8,
                            size_bitstring=15,
                            num_bit_flips=60,
                            clean=False,
                            select_swap_depth=4,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceQROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                clean=False,
                                select_swap_depth=4,
                            )
                        )
                    ),
                    GateCount(
                        plre.ResourcePhaseShift.resource_rep(),
                        15,
                    ),
                    FreeWires(15),
                ],
            ),
        ),
    )
    def test_control_ry_resources(
        self, num_state_qubits, precision, positive_and_real, selswap_depths, expected_res
    ):
        """Test that the resources are as expected for the controlled-RY decomposition"""
        config = {"precision_qrom_state_prep": 1e-9}
        actual_resources = plre.ResourceQROMStatePreparation.controlled_ry_resource_decomp(
            num_state_qubits=num_state_qubits,
            precision=precision,
            positive_and_real=positive_and_real,
            selswap_depths=selswap_depths,
            config=config,
        )

        assert actual_resources == expected_res
