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
Tests for quantum algorithmic subroutines resource operators.
"""
import math

import numpy as np
import pytest

import pennylane as qml
import pennylane.estimator as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.templates import HybridQRAM, SelectOnlyQRAM
from pennylane.wires import Wires

# pylint: disable=no-self-use,too-many-arguments


class TestResourceOutOfPlaceSquare:
    """Test the OutOfPlaceSquare class."""

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = qre.OutOfPlaceSquare(register_size)
        assert op.resource_params == {"register_size": register_size}

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.OutOfPlaceSquare, 3 * register_size, {"register_size": register_size}
        )
        assert qre.OutOfPlaceSquare.resource_rep(register_size=register_size) == expected

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resources(self, register_size):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(qre.Toffoli), (register_size - 1) ** 2),
            GateCount(resource_rep(qre.CNOT), register_size),
        ]
        assert qre.OutOfPlaceSquare.resource_decomp(register_size=register_size) == expected


class TestIQP:
    """Test the IQP class."""

    @pytest.mark.parametrize(
        ("num_wires", "pattern", "spin_sym"),
        [
            (4, [[[0]], [[1]], [[2]], [[3]]], False),
            (
                6,
                [[[0]], [[4]], [[3]], [[2]], [[1]], [[5]]],
                True,
            ),
        ],
    )
    def test_resource_params(self, num_wires, pattern, spin_sym):
        """Test that the resource params are correct."""
        op = qre.IQP(num_wires, pattern, spin_sym)
        assert op.resource_params == {
            "spin_sym": spin_sym,
            "pattern": pattern,
            "num_wires": num_wires,
        }

    @pytest.mark.parametrize(
        ("num_wires", "pattern", "spin_sym"),
        [
            (4, [[[0]], [[1]], [[2]], [[3]]], False),
            (
                6,
                [[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]],
                True,
            ),
        ],
    )
    def test_resource_rep(self, num_wires, pattern, spin_sym):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.IQP,
            num_wires,
            {
                "num_wires": num_wires,
                "pattern": pattern,
                "spin_sym": spin_sym,
            },
        )
        assert (
            qre.IQP.resource_rep(num_wires=num_wires, pattern=pattern, spin_sym=spin_sym)
            == expected
        )

    @pytest.mark.parametrize(
        ("num_wires", "pattern", "spin_sym", "expected_res"),
        [
            (
                4,
                [[[0]], [[1]], [[2]], [[3]]],
                False,
                [
                    GateCount(resource_rep(qre.Hadamard), 8),
                    GateCount(resource_rep(qre.MultiRZ, {"num_wires": 1}), 4),
                ],
            ),
            (
                6,
                [[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]],
                True,
                [
                    GateCount(resource_rep(qre.Hadamard), 12),
                    GateCount(resource_rep(qre.PauliRot, {"pauli_string": "YXXXXX"}), 1),
                    GateCount(resource_rep(qre.MultiRZ, {"num_wires": 1}), 6),
                ],
            ),
        ],
    )
    def test_resources(self, num_wires, pattern, spin_sym, expected_res):
        """Test that the resources are correct."""
        assert qre.IQP.resource_decomp(num_wires, pattern, spin_sym) == expected_res

    @pytest.mark.parametrize(
        ("num_wires", "pattern", "spin_sym", "expected"),
        [
            (4, [[[0]], [[1]], [[2]], [[3]]], False, "IQP(4, [[[0]], [[1]], [[2]], [[3]]], False)"),
            (
                6,
                [[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]],
                True,
                "IQP(6, [[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]], True)",
            ),
        ],
    )
    def test_tracking_name(self, num_wires, pattern, spin_sym, expected):
        """Test that the tracking name is correct."""
        assert qre.IQP.tracking_name(num_wires, pattern, spin_sym) == expected


class TestHybridQRAM:
    """Test the HybridQRAM class."""

    def test_raises_with_wrong_wire_num(self):
        with pytest.raises(ValueError, match="Expected 10 wires, got 4."):
            qre.HybridQRAM(
                ["000", "010", "101", "111"],
                10,
                2,
                2,
                control_wires=(
                    0,
                    1,
                ),
                target_wires=(2,),
                work_wires=(3,),
            )

    @pytest.mark.parametrize(
        (
            "num_wires",
            "data",
            "num_select_wires",
            "num_control_wires",
            "control_wires",
            "target_wires",
            "work_wires",
        ),
        [
            (
                15,
                np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]),
                1,
                2,
                (0, 1),
                (2, 3, 4),
                (5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            ),
        ],
    )
    def test_resource_params(
        self,
        num_wires,
        data,
        num_select_wires,
        num_control_wires,
        control_wires,
        target_wires,
        work_wires,
    ):
        """Test that the resource params are correct."""
        op = qre.HybridQRAM(
            data,
            num_wires,
            num_select_wires,
            num_control_wires,
            control_wires,
            target_wires,
            work_wires,
        )
        assert np.allclose(data, op.resource_params["data"])
        assert op.resource_params["num_wires"] == num_wires
        assert op.resource_params["num_select_wires"] == num_select_wires
        assert op.resource_params["num_tree_control_wires"] == num_control_wires - num_select_wires

    @pytest.mark.parametrize(
        ("num_wires", "data", "num_select_wires", "num_control_wires"),
        [(15, ["000", "010", "101", "111"], 5, 10)],
    )
    def test_resource_rep(self, num_wires, data, num_select_wires, num_control_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.HybridQRAM,
            num_wires,
            {
                "data": data,
                "num_wires": num_wires,
                "num_select_wires": num_select_wires,
                "num_tree_control_wires": num_control_wires - num_select_wires,
            },
        )
        assert (
            qre.HybridQRAM.resource_rep(
                data=data,
                num_wires=num_wires,
                num_select_wires=num_select_wires,
                num_tree_control_wires=num_control_wires - num_select_wires,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "data, num_wires, num_select_wires, num_control_wires, expected_res",
        (
            (
                np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]),
                10,
                1,
                2,
                [
                    GateCount(resource_rep(qre.CSWAP), 20),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Controlled.resource_rep(resource_rep(qre.SWAP), 1, 1), 1, 0
                        ),
                        12,
                    ),
                    GateCount(qre.Controlled.resource_rep(resource_rep(qre.CSWAP), 1, 0), 12),
                    GateCount(qre.Controlled.resource_rep(resource_rep(qre.Hadamard), 1, 0), 12),
                    GateCount(qre.Controlled.resource_rep(resource_rep(qre.Z), 1, 0), 6),
                    GateCount(
                        qre.Controlled.resource_rep(resource_rep(qre.X), 1, num_zero_ctrl=1), 2
                    ),
                    GateCount(resource_rep(qre.CNOT), 2),
                ],
            ),
            (
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                        [1, 1, 0],
                        [0, 0, 0],
                    ]
                ),
                16,
                0,
                2,
                [
                    GateCount(resource_rep(qre.CSWAP), 16),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Controlled.resource_rep(resource_rep(qre.SWAP), 1, 1), 1, 0
                        ),
                        20,
                    ),
                    GateCount(qre.Controlled.resource_rep(resource_rep(qre.CSWAP), 1, 0), 20),
                    GateCount(qre.Controlled.resource_rep(resource_rep(qre.Hadamard), 1, 0), 6),
                    GateCount(qre.Controlled.resource_rep(resource_rep(qre.Z), 1, 0), 6),
                    GateCount(resource_rep(qre.X), 2),
                ],
            ),
        ),
    )
    def test_resources(self, data, num_wires, num_select_wires, num_control_wires, expected_res):
        """Test that the resources are correct."""
        decomp = qre.HybridQRAM.resource_decomp(
            data, num_wires, num_select_wires, num_control_wires - num_select_wires
        )
        assert decomp == expected_res

        dev = qml.device("default.qubit")

        @qml.transforms.decompose(max_expansion=1)
        @qml.qnode(dev)
        def circuit():
            HybridQRAM(
                data=data,
                control_wires=range(num_control_wires),
                target_wires=range(num_control_wires, num_control_wires + data.shape[1]),
                work_wires=range(
                    num_control_wires + data.shape[1],
                    num_control_wires
                    + data.shape[1]
                    + num_wires
                    - (num_control_wires + data.shape[1]),
                ),
                k=num_select_wires,
            )

        specs = qml.specs(circuit)()

        def _match_controlled(name, op):
            if (
                # pylint: disable=too-many-boolean-expressions
                name == "MultiControlledX"
                and op.name.startswith("C(X")
                or name == "CH"
                and op.name.startswith("C(H")
                or name == "2C(SWAP)"
                and (op.name.startswith("C(C(SWAP") or op.name.startswith("C(CSWAP"))
                or name == "CZ"
                and op.name.startswith("C(Z")
            ):
                return True
            return name == op.name

        for ty, count in specs.resources.gate_types.items():
            found = False
            i = 0
            total = 0
            while i < len(expected_res):
                if expected_res[i].gate.op_type.__name__ == ty.replace(
                    "Pauli", ""
                ) or _match_controlled(ty, expected_res[i].gate):
                    total += expected_res[i].count
                    found = True
                i += 1
            assert found
            assert total == count

    @pytest.mark.parametrize(
        ("data", "num_wires", "num_select_wires", "num_control_wires"),
        [(["000", "101", "010", "111"], 15, 1, 2)],
    )
    def test_tracking_name(self, data, num_wires, num_select_wires, num_control_wires):
        """Tests that the tracking name is correct."""
        assert (
            qre.HybridQRAM.tracking_name(
                data, num_wires, num_select_wires, num_control_wires - num_select_wires
            )
            == f"HybridQRAM({data}, {num_wires}, {num_select_wires}, {num_control_wires -  num_select_wires})"
        )


class TestSelectOnlyQRAM:
    """Test the SelectOnlyQRAM class."""

    def test_raises_with_wrong_wire_num(self):
        with pytest.raises(ValueError, match="Expected 7 wires, got 4."):
            qre.SelectOnlyQRAM(
                [
                    [1],
                    [1],
                    [1],
                    [1],
                ],
                7,
                2,
                2,
                control_wires=(
                    0,
                    1,
                ),
                target_wires=(2,),
                select_wires=(3,),
                select_value=0,
            )

    @pytest.mark.parametrize(
        (
            "data",
            "num_wires",
            "num_control_wires",
            "num_select_wires",
            "control_wires",
            "target_wires",
            "select_wires",
            "select_value",
        ),
        [
            ([[1], [0], [1], [0]], 7, 2, 3, (0, 1), (2, 3, 4), (5, 6), 0),
        ],
    )
    def test_resource_params(
        self,
        data,
        num_wires,
        num_control_wires,
        num_select_wires,
        control_wires,
        target_wires,
        select_wires,
        select_value,
    ):
        """Test that the resource params are correct."""
        op = qre.SelectOnlyQRAM(
            data,
            num_wires,
            num_control_wires,
            num_select_wires,
            control_wires,
            target_wires,
            select_wires,
            select_value,
        )
        assert np.allclose(op.resource_params["data"], data)
        assert op.resource_params["num_wires"] == num_wires
        assert op.resource_params["num_control_wires"] == num_control_wires
        assert op.resource_params["num_select_wires"] == num_select_wires
        assert op.resource_params["select_value"] == select_value

    @pytest.mark.parametrize(
        (
            "data",
            "num_wires",
            "select_value",
            "num_select_wires",
            "num_control_wires",
        ),
        [([[1], [0], [1], [0]], 7, 0, 2, 2)],
    )
    def test_resource_rep(self, data, num_wires, select_value, num_select_wires, num_control_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.SelectOnlyQRAM,
            num_wires,
            {
                "data": data,
                "num_wires": num_wires,
                "select_value": select_value,
                "num_select_wires": num_select_wires,
                "num_control_wires": num_control_wires,
            },
        )
        assert (
            qre.SelectOnlyQRAM.resource_rep(
                data=data,
                num_wires=num_wires,
                select_value=select_value,
                num_select_wires=num_select_wires,
                num_control_wires=num_control_wires,
            )
            == expected
        )

    @pytest.mark.parametrize(
        (
            "data",
            "num_wires",
            "select_value",
            "num_select_wires",
            "num_control_wires",
            "expected",
        ),
        (
            (
                np.array(
                    [[1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0]]
                ),
                7,
                0,
                2,
                2,
                [
                    GateCount(resource_rep(qre.X), 24),
                    GateCount(
                        qre.Controlled.resource_rep(
                            resource_rep(qre.X), num_ctrl_wires=4, num_zero_ctrl=0
                        ),
                        2,
                    ),
                    GateCount(resource_rep(qre.BasisEmbedding, {"num_wires": 2}), 1),
                ],
            ),
        ),
    )
    def test_resources(
        self,
        data,
        num_wires,
        select_value,
        num_select_wires,
        num_control_wires,
        expected,
    ):
        """Test that the resources are correct."""
        assert (
            qre.SelectOnlyQRAM.resource_decomp(
                data,
                num_wires,
                select_value,
                num_select_wires,
                num_control_wires,
            )
            == expected
        )

        dev = qml.device("default.qubit")

        @qml.transforms.decompose
        @qml.qnode(dev)
        def circuit():
            SelectOnlyQRAM(
                data=data,
                control_wires=range(num_control_wires),
                target_wires=range(num_control_wires, num_control_wires + data.shape[1]),
                select_wires=range(
                    num_control_wires + data.shape[1],
                    num_control_wires + data.shape[1] + num_select_wires,
                ),
                select_value=select_value,
            )

        specs = qml.specs(circuit)()

        for ty, count in specs.resources.gate_types.items():
            found = False
            i = 0
            while not found and i < len(expected):
                if (
                    expected[i].gate.op_type.__name__ == ty.replace("Pauli", "")
                    or ty == "MultiControlledX"
                    and "Controlled" == expected[i].gate.op_type.__name__
                ):
                    assert expected[i].count == count
                    found = True
                i += 1
            assert found

    @pytest.mark.parametrize(
        (
            "data",
            "num_wires",
            "select_value",
            "num_select_wires",
            "num_control_wires",
        ),
        [([[1], [0], [1], [0]], 7, 0, 2, 2)],
    )
    def test_tracking_name(
        self, data, num_wires, select_value, num_select_wires, num_control_wires
    ):
        """Tests that the tracking name is correct."""
        assert (
            qre.SelectOnlyQRAM.tracking_name(
                data,
                num_wires,
                select_value,
                num_select_wires,
                num_control_wires,
            )
            == f"SelectOnlyQRAM({data}, {num_wires}, {select_value}, {num_select_wires}, {num_control_wires})"
        )


class TestBBQRAM:
    """Test the BBQRAM class."""

    def test_raises_with_wrong_wire_num(self):
        with pytest.raises(ValueError, match="Expected 4 wires, got 3."):
            qre.BBQRAM(
                4,
                3,
                4,
                6,
                control_wires=(1,),
                target_wires=(2,),
                work_wires=(3,),
            )

    @pytest.mark.parametrize(
        (
            "num_wires",
            "num_bitstrings",
            "size_bitstring",
            "num_bit_flips",
            "control_wires",
            "target_wires",
            "work_wires",
        ),
        [
            (
                15,
                4,
                3,
                6,
                (0, 1),
                (2, 3, 4),
                (5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            ),
            (
                15,
                4,
                3,
                None,
                (0, 1),
                (2, 3, 4),
                (5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            ),
        ],
    )
    def test_resource_params(
        self,
        num_wires,
        num_bitstrings,
        size_bitstring,
        num_bit_flips,
        control_wires,
        target_wires,
        work_wires,
    ):
        """Test that the resource params are correct."""
        op = qre.BBQRAM(
            num_bitstrings,
            size_bitstring,
            num_wires,
            num_bit_flips,
            control_wires,
            target_wires,
            work_wires,
        )
        assert op.resource_params == {
            "num_bitstrings": num_bitstrings,
            "size_bitstring": size_bitstring,
            "num_bit_flips": (
                num_bit_flips if num_bit_flips is not None else num_bitstrings * size_bitstring // 2
            ),
            "num_wires": num_wires,
        }

    @pytest.mark.parametrize(
        ("num_wires", "num_bitstrings", "size_bitstring", "num_bit_flips"), [(15, 4, 3, 6)]
    )
    def test_resource_rep(self, num_wires, num_bitstrings, size_bitstring, num_bit_flips):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.BBQRAM,
            num_wires,
            {
                "num_bitstrings": num_bitstrings,
                "size_bitstring": size_bitstring,
                "num_bit_flips": num_bit_flips,
                "num_wires": num_wires,
            },
        )
        assert (
            qre.BBQRAM.resource_rep(
                num_bitstrings=num_bitstrings,
                size_bitstring=size_bitstring,
                num_bit_flips=num_bit_flips,
                num_wires=num_wires,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "num_bitstrings, size_bitstring, num_bit_flips, num_wires, expected_res",
        (
            (
                4,
                3,
                6,
                15,
                [
                    GateCount(resource_rep(qre.SWAP), 16),
                    GateCount(resource_rep(qre.Hadamard), 6),
                    GateCount(resource_rep(qre.CSWAP), 40),
                    GateCount(resource_rep(qre.Z), 6),
                ],
            ),
        ),
    )
    def test_resources(
        self, num_bitstrings, size_bitstring, num_bit_flips, num_wires, expected_res
    ):
        """Test that the resources are correct."""
        assert (
            qre.BBQRAM.resource_decomp(num_bitstrings, size_bitstring, num_bit_flips, num_wires)
            == expected_res
        )

    @pytest.mark.parametrize(
        ("num_bitstrings", "size_bitstring", "num_bit_flips", "num_wires"), [(4, 3, 6, 15)]
    )
    def test_tracking_name(self, num_bitstrings, size_bitstring, num_bit_flips, num_wires):
        """Tests that the tracking name is correct."""
        assert (
            qre.BBQRAM.tracking_name(num_bitstrings, size_bitstring, num_bit_flips, num_wires)
            == f"BBQRAM({num_bitstrings}, {size_bitstring}, {num_bit_flips}, {num_wires})"
        )


class TestResourcePhaseGradient:
    """Test the PhaseGradient class."""

    def test_init_no_num_wires(self):
        """Test that we can instantiate the operator without providing num_wires"""
        op = qre.PhaseGradient(wires=range(3))
        assert op.resource_params == {"num_wires": 3}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and num_wires are both not provided"""
        with pytest.raises(ValueError, match="Must provide at least one of"):
            qre.PhaseGradient()

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = qre.PhaseGradient(num_wires)
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.PhaseGradient, num_wires, {"num_wires": num_wires})
        assert qre.PhaseGradient.resource_rep(num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [
                    GateCount(qre.Hadamard.resource_rep()),
                    GateCount(qre.Z.resource_rep()),
                ],
            ),
            (
                2,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.S.resource_rep()),
                ],
            ),
            (
                3,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.S.resource_rep()),
                    GateCount(qre.T.resource_rep()),
                ],
            ),
            (
                5,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.S.resource_rep()),
                    GateCount(qre.T.resource_rep()),
                    GateCount(qre.RZ.resource_rep(), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, expected_res):
        """Test that the resources are correct."""
        assert qre.PhaseGradient.resource_decomp(num_wires=num_wires) == expected_res


class TestResourceOutMultiplier:
    """Test the OutMultiplier class."""

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_params(self, a_register_size, b_register_size):
        """Test that the resource params are correct."""
        op = qre.OutMultiplier(a_register_size, b_register_size)
        assert op.resource_params == {
            "a_num_wires": a_register_size,
            "b_num_wires": b_register_size,
        }

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_rep(self, a_register_size, b_register_size):
        """Test that the compressed representation is correct."""
        expected_num_wires = a_register_size + 3 * b_register_size
        expected = qre.CompressedResourceOp(
            qre.OutMultiplier,
            expected_num_wires,
            {"a_num_wires": a_register_size, "b_num_wires": b_register_size},
        )
        assert qre.OutMultiplier.resource_rep(a_register_size, b_register_size) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        a_register_size = 5
        b_register_size = 3

        toff = resource_rep(qre.Toffoli)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        num_elbows = 12
        num_toff = 1

        expected = [
            GateCount(l_elbow, num_elbows),
            GateCount(r_elbow, num_elbows),
            GateCount(toff, num_toff),
        ]
        assert qre.OutMultiplier.resource_decomp(a_register_size, b_register_size) == expected


class TestResourceSemiAdder:
    """Test the ResourceSemiAdder class."""

    @pytest.mark.parametrize("register_size", (1, 2, 3, 4))
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = qre.SemiAdder(register_size)
        assert op.resource_params == {"max_register_size": register_size}

    @pytest.mark.parametrize("register_size", (1, 2, 3, 4))
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.SemiAdder, 2 * register_size, {"max_register_size": register_size}
        )
        assert qre.SemiAdder.resource_rep(max_register_size=register_size) == expected

    @pytest.mark.parametrize(
        "register_size, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(qre.CNOT))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(qre.CNOT), 2),
                    GateCount(resource_rep(qre.X), 2),
                    GateCount(resource_rep(qre.Toffoli)),
                ],
            ),
            (
                3,
                [
                    qre.Allocate(2),
                    GateCount(resource_rep(qre.CNOT), 9),
                    GateCount(resource_rep(qre.TemporaryAND), 2),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.TemporaryAND)},
                        ),
                        2,
                    ),
                    qre.Deallocate(2),
                ],
            ),
        ),
    )
    def test_resources(self, register_size, expected_res):
        """Test that the resources are correct."""
        assert qre.SemiAdder.resource_decomp(register_size) == expected_res

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, max_register_size, expected_res",
        (
            (
                1,
                1,
                1,
                [
                    GateCount(resource_rep(qre.X), 2),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.CNOT),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                ],
            ),
            (
                1,
                0,
                5,
                [
                    qre.Allocate(4),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.TemporaryAND), 8),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        8,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                2,
                1,
                5,
                [
                    qre.Allocate(1),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {
                                "num_ctrl_wires": 2,
                                "num_zero_ctrl": 1,
                            },
                        ),
                        2,
                    ),
                    qre.Allocate(4),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.TemporaryAND), 8),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        8,
                    ),
                    qre.Deallocate(4),
                    qre.Deallocate(1),
                ],
            ),
            (
                1,
                1,
                5,
                [
                    qre.Allocate(4),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.TemporaryAND), 8),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        8,
                    ),
                    qre.Deallocate(4),
                    GateCount(resource_rep(qre.X), 2),
                ],
            ),
        ),
    )
    def test_resources_controlled(
        self, num_ctrl_wires, num_zero_ctrl, max_register_size, expected_res
    ):
        """Test that the special case controlled resources are correct."""
        op = qre.Controlled(
            qre.SemiAdder(max_register_size=max_register_size),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceControlledSequence:
    """Test the ResourceControlledSequence class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 10 wires, got 3"):
            qre.ControlledSequence(
                base=qre.QFT(5, [0, 1, 2, 3, 4]), num_control_wires=5, wires=[0, 1, 2]
            )

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires",
        (
            (qre.QFT(5), 5),
            (qre.RZ(precision=1e-3), 10),
            (
                qre.MultiRZ(
                    3,
                    1e-5,
                ),
                3,
            ),
        ),
    )
    def test_resource_params(self, base_op, num_ctrl_wires):
        """Test the resource params"""
        op = qre.ControlledSequence(base_op, num_ctrl_wires)
        expected_params = {
            "base_cmpr_op": base_op.resource_rep_from_op(),
            "num_ctrl_wires": num_ctrl_wires,
        }

        assert op.resource_params == expected_params

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires",
        (
            (qre.QFT(5), 5),
            (qre.RZ(precision=1e-3), 10),
            (
                qre.MultiRZ(
                    3,
                    1e-5,
                ),
                3,
            ),
        ),
    )
    def test_resource_rep(self, base_op, num_ctrl_wires):
        """Test the resource rep method"""
        base_cmpr_op = base_op.resource_rep_from_op()
        expected = qre.CompressedResourceOp(
            qre.ControlledSequence,
            base_cmpr_op.num_wires + num_ctrl_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )

        assert qre.ControlledSequence.resource_rep(base_cmpr_op, num_ctrl_wires) == expected

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires, expected_res",
        (
            (
                qre.QFT(5),
                5,
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                16,
                            ),
                            1,
                            0,
                        )
                    ),
                ],
            ),
            (
                qre.RZ(precision=1e-3),
                3,
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-3),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-3),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-3),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                ],
            ),
            (
                qre.ChangeOpBasis(
                    compute_op=qre.AQFT(3, 5),
                    target_op=qre.RZ(),
                ),
                3,
                [
                    GateCount(qre.AQFT.resource_rep(3, 5)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 5))),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_ctrl_wires, expected_res):
        """Test resources"""
        op = qre.ControlledSequence(base_op, num_ctrl_wires)
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceQPE:
    """Test the ResourceQPE class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 3"):
            qre.QPE(base=qre.X(0), num_estimation_wires=3, adj_qft_op=qre.QFT(3), wires=[0, 1, 2])

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        op = qre.QPE(base=qre.X(), num_estimation_wires=3, adj_qft_op=qre.QFT(3))
        assert (
            op.tracking_name(resource_rep(qre.X), 3, resource_rep(qre.QFT, {"num_wires": 3}))
            == "QPE(X, 3, adj_qft=QFT(3))"
        )

    @pytest.mark.parametrize(
        "base_op, num_est_wires, adj_qft_op",
        (
            (qre.RX(precision=1e-5), 5, None),
            (qre.X(), 3, qre.QFT(3)),
            (qre.RZ(), 4, qre.Adjoint(qre.AQFT(3, 4))),
        ),
    )
    def test_resource_params(self, base_op, num_est_wires, adj_qft_op):
        """Test the resource_params method"""
        base_cmpr_op = base_op.resource_rep_from_op()

        if adj_qft_op is None:
            op = qre.QPE(base_op, num_est_wires)
            adj_qft_cmpr_op = None
        else:
            op = qre.QPE(base_op, num_est_wires, adj_qft_op)
            adj_qft_cmpr_op = adj_qft_op.resource_rep_from_op()

        assert op.resource_params == {
            "base_cmpr_op": base_cmpr_op,
            "num_estimation_wires": num_est_wires,
            "adj_qft_cmpr_op": adj_qft_cmpr_op,
        }

    @pytest.mark.parametrize(
        "base_cmpr_op, num_est_wires, adj_qft_cmpr_op",
        (
            (qre.RX.resource_rep(precision=1e-5), 5, None),
            (qre.X.resource_rep(), 3, qre.QFT.resource_rep(3)),
            (
                qre.RZ.resource_rep(),
                4,
                qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 4)),
            ),
        ),
    )
    def test_resource_rep(self, base_cmpr_op, num_est_wires, adj_qft_cmpr_op):
        """Test the resource_rep method"""
        if adj_qft_cmpr_op is None:
            adj_qft_cmpr_op = qre.Adjoint.resource_rep(qre.QFT.resource_rep(num_est_wires))

        expected = qre.CompressedResourceOp(
            qre.QPE,
            base_cmpr_op.num_wires + num_est_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_estimation_wires": num_est_wires,
                "adj_qft_cmpr_op": adj_qft_cmpr_op,
            },
        )

        assert qre.QPE.resource_rep(base_cmpr_op, num_est_wires, adj_qft_cmpr_op) == expected

    @pytest.mark.parametrize(
        "base_op, num_est_wires, adj_qft_op, expected_res",
        (
            (
                qre.RX(precision=1e-5),
                5,
                None,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5),
                    GateCount(
                        qre.ControlledSequence.resource_rep(
                            qre.RX.resource_rep(precision=1e-5),
                            5,
                        ),
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.QFT.resource_rep(5))),
                ],
            ),
            (
                qre.X(),
                3,
                qre.QFT(3),
                [
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    GateCount(
                        qre.ControlledSequence.resource_rep(
                            qre.X.resource_rep(),
                            3,
                        ),
                    ),
                    GateCount(qre.QFT.resource_rep(3)),
                ],
            ),
            (
                qre.RZ(),
                4,
                qre.Adjoint(qre.AQFT(3, 4)),
                [
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(
                        qre.ControlledSequence.resource_rep(
                            qre.RZ.resource_rep(),
                            4,
                        ),
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 4)),
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_est_wires, adj_qft_op, expected_res):
        """Test that resources method is correct"""
        op = (
            qre.QPE(base_op, num_est_wires)
            if adj_qft_op is None
            else qre.QPE(base_op, num_est_wires, adj_qft_op)
        )
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceIterativeQPE:
    """Test the ResourceIterativeQPE class."""

    @pytest.mark.parametrize(
        "base_op, num_iter",
        (
            (qre.RX(precision=1e-5), 5),
            (qre.QubitUnitary(4, 1e-5), 7),
            (
                qre.ChangeOpBasis(
                    qre.RY(precision=1e-3),
                    qre.RZ(precision=1e-5),
                ),
                3,
            ),
        ),
    )
    def test_resource_params(self, base_op, num_iter):
        """Test the resource_params method"""
        op = qre.IterativeQPE(base_op, num_iter)
        expected = {
            "base_cmpr_op": base_op.resource_rep_from_op(),
            "num_iter": num_iter,
        }
        assert op.resource_params == expected

    @pytest.mark.parametrize(
        "base_op, num_iter",
        (
            (qre.RX(precision=1e-5), 5),
            (qre.QubitUnitary(4, 1e-5), 7),
            (
                qre.ChangeOpBasis(
                    qre.RY(precision=1e-3),
                    qre.RZ(precision=1e-5),
                ),
                3,
            ),
        ),
    )
    def test_resource_rep(self, base_op, num_iter):
        """Test the resource_rep method"""
        base_cmpr_op = base_op.resource_rep_from_op()
        expected = qre.CompressedResourceOp(
            qre.IterativeQPE,
            base_cmpr_op.num_wires,
            {"base_cmpr_op": base_cmpr_op, "num_iter": num_iter},
        )
        assert qre.IterativeQPE.resource_rep(base_cmpr_op, num_iter) == expected

    @pytest.mark.parametrize(
        "base_op, num_iter, expected_res",
        (
            (
                qre.RX(precision=1e-5),
                5,
                [
                    GateCount(qre.Hadamard.resource_rep(), 10),
                    Allocate(1),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                16,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(qre.PhaseShift.resource_rep(), 10),
                    Deallocate(1),
                ],
            ),
            (
                qre.QubitUnitary(7, 1e-5),
                4,
                [
                    GateCount(qre.Hadamard.resource_rep(), 8),
                    Allocate(1),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(qre.PhaseShift.resource_rep(), 6),
                    Deallocate(1),
                ],
            ),
            (
                qre.ChangeOpBasis(
                    qre.RY(precision=1e-3),
                    qre.RZ(precision=1e-5),
                ),
                3,
                [
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    Allocate(1),
                    GateCount(qre.RY.resource_rep(precision=1e-3)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.RY.resource_rep(precision=1e-3)),
                    ),
                    GateCount(qre.PhaseShift.resource_rep(), 3),
                    Deallocate(1),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_iter, expected_res):
        """Test the resources method"""
        op = qre.IterativeQPE(base_op, num_iter)
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceQFT:
    """Test the ResourceQFT class."""

    def test_init_no_num_wires(self):
        """Test that we can instantiate the operator without providing num_wires"""
        op = qre.QFT(wires=range(3))
        assert op.resource_params == {"num_wires": 3}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and num_wires are both not provided"""
        with pytest.raises(ValueError, match="Must provide at least one of"):
            qre.QFT()

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        assert qre.QFT(1).tracking_name(1) == "QFT(1)"

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = qre.QFT(num_wires)
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.QFT, num_wires, {"num_wires": num_wires})
        assert qre.QFT.resource_rep(num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(qre.Hadamard))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(resource_rep(qre.ControlledPhaseShift)),
                ],
            ),
            (
                3,
                [
                    GateCount(resource_rep(qre.Hadamard), 3),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(resource_rep(qre.ControlledPhaseShift), 3),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, expected_res):
        """Test that the resources are correct."""
        assert qre.QFT.resource_decomp(num_wires) == expected_res

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(qre.Hadamard))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                ],
            ),
            (
                3,
                [
                    GateCount(resource_rep(qre.Hadamard), 3),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=2),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                ],
            ),
        ),
    )
    def test_resources_phasegrad(self, num_wires, expected_res):
        """Test that the resources are correct for phase gradient method."""
        assert qre.QFT.phase_grad_resource_decomp(num_wires) == expected_res

    def test_phase_grad_resource_decomp_estimate(self):
        """Test the resource estimation with QFT.phase_grad_resource_decomp."""
        config = qre.ResourceConfig()
        config.set_decomp(qre.QFT, qre.QFT.phase_grad_resource_decomp)

        op = qre.QFT(3)
        resources = qre.estimate(op, config=config)

        expected_gates = {
            "Toffoli": 5,
            "CNOT": 6,
            "Hadamard": 6,
        }
        assert resources.gate_counts == expected_gates
        assert resources.algo_wires == 3
        assert resources.any_state_wires == 0
        assert resources.zeroed_wires == 1


class TestResourceAQFT:
    """Test the ResourceAQFT class."""

    @pytest.mark.parametrize("order", (-1, 0))
    def test_aqft_order_errors(self, order):
        """Test that the correct error is raised when invalid values of order are provided."""
        with pytest.raises(ValueError, match="Order must be a positive integer greater than 0."):
            qre.AQFT(order, 3)

    def test_init_no_num_wires(self):
        """Test that we can instantiate the operator without providing num_wires"""
        op = qre.AQFT(order=2, wires=range(3))
        assert op.resource_params == {"order": 2, "num_wires": 3}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and num_wires are both not provided"""
        with pytest.raises(ValueError, match="Must provide at least one of"):
            qre.AQFT(order=2)

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        assert qre.AQFT(3, 2).tracking_name(3, 2) == "AQFT(3, 2)"

    @pytest.mark.parametrize(
        "num_wires, order",
        (
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (5, 5),
        ),
    )
    def test_resource_params(self, num_wires, order):
        """Test that the resource params are correct."""
        op = qre.AQFT(order, num_wires)
        assert op.resource_params == {"order": order, "num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires, order",
        (
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (5, 5),
        ),
    )
    def test_resource_rep(self, order, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.AQFT, num_wires, {"order": order, "num_wires": num_wires}
        )
        assert qre.AQFT.resource_rep(order=order, num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, order, expected_res",
        (
            (
                5,
                1,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                ],
            ),
            (
                5,
                3,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=resource_rep(qre.S),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    Allocate(1),
                    GateCount(resource_rep(qre.TemporaryAND), 1),
                    GateCount(qre.SemiAdder.resource_rep(1)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        1,
                    ),
                    Deallocate(1),
                    Allocate(2),
                    GateCount(resource_rep(qre.TemporaryAND), 2 * 2),
                    GateCount(qre.SemiAdder.resource_rep(2), 2),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        2 * 2,
                    ),
                    Deallocate(2),
                    GateCount(resource_rep(qre.SWAP), 2),
                ],
            ),
            (
                5,
                5,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=resource_rep(qre.S),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    Allocate(1),
                    GateCount(resource_rep(qre.TemporaryAND), 1),
                    GateCount(qre.SemiAdder.resource_rep(1)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        1,
                    ),
                    Deallocate(1),
                    Allocate(2),
                    GateCount(resource_rep(qre.TemporaryAND), 2),
                    GateCount(qre.SemiAdder.resource_rep(2)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        2,
                    ),
                    Deallocate(2),
                    Allocate(3),
                    GateCount(resource_rep(qre.TemporaryAND), 3),
                    GateCount(qre.SemiAdder.resource_rep(3)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        3,
                    ),
                    Deallocate(3),
                    GateCount(resource_rep(qre.SWAP), 2),
                ],
            ),
        ),
    )
    def test_resources(self, order, num_wires, expected_res):
        """Test that the resources are correct."""
        assert qre.AQFT.resource_decomp(order, num_wires) == expected_res


class TestResourceBasisRotation:
    """Test the BasisRotation class."""

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        assert qre.BasisRotation(1).tracking_name(1) == "BasisRotation(1)"

    def test_init_no_dim(self):
        """Test that we can instantiate the operator without providing dim"""
        op = qre.BasisRotation(wires=range(3))
        assert op.resource_params == {"dim": 3}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and dim are both not provided"""
        with pytest.raises(ValueError, match="Must provide at least one of"):
            qre.BasisRotation()

    @pytest.mark.parametrize("dim", (1, 2, 3))
    def test_resource_params(self, dim):
        """Test that the resource params are correct."""
        op = qre.BasisRotation(dim)
        assert op.resource_params == {"dim": dim}

    @pytest.mark.parametrize("dim", (1, 2, 3))
    def test_resource_rep(self, dim):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.BasisRotation, dim, {"dim": dim})
        assert qre.BasisRotation.resource_rep(dim=dim) == expected

    @pytest.mark.parametrize("dim", (1, 2, 3))
    def test_resources(self, dim):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(qre.PhaseShift), dim + (dim * (dim - 1) // 2)),
            GateCount(resource_rep(qre.SingleExcitation), dim * (dim - 1) // 2),
        ]
        assert qre.BasisRotation.resource_decomp(dim) == expected


class TestResourceSelect:
    """Test the Select class."""

    def test_select_factor_errors(self):
        """Test that the correct error is raised when invalid ops are provided."""
        with pytest.raises(ValueError, match="All factors of the Select must be instances of"):
            qre.Select(ops=[qml.X(0), qre.Y(), qre.Z()])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected at least 4 wires"):
            qre.Select(ops=[qre.RX(), qre.Z(), qre.CNOT()], wires=[0])

    def test_wire_init(self):
        """Test that the number of wires is correctly computed from the provided wires."""
        wires = [0, 1, 2, 3]
        op = qre.Select(ops=[qre.RX(), qre.Z(), qre.CNOT()], wires=wires)
        assert op.num_wires == len(wires)

    def test_resource_params(self):
        """Test that the resource params are correct."""
        ops = [qre.RX(), qre.Z(), qre.CNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        op = qre.Select(ops)
        assert op.resource_params == {"cmpr_ops": cmpr_ops, "num_wires": 4}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        ops = [qre.RX(wires=0), qre.Z(wires=1), qre.CNOT(wires=[1, 2])]
        num_wires = 3 + 2  # 3 op wires + 2 control wires
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = qre.CompressedResourceOp(
            qre.Select, num_wires, {"cmpr_ops": cmpr_ops, "num_wires": num_wires}
        )
        print(expected)
        print(qre.Select.resource_rep(cmpr_ops, num_wires))
        assert qre.Select.resource_rep(cmpr_ops, num_wires) == expected

        op = qre.Select(ops)
        print(op.resource_rep(**op.resource_params))
        assert op.resource_rep(**op.resource_params) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        ops = [qre.RX(), qre.Z(), qre.CNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = [
            qre.Allocate(1),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.RX.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.Z.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.CNOT.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(qre.X.resource_rep(), 4),
            GateCount(qre.CNOT.resource_rep(), 2),
            GateCount(qre.TemporaryAND.resource_rep(), 2),
            GateCount(
                qre.Adjoint.resource_rep(
                    qre.TemporaryAND.resource_rep(),
                ),
                2,
            ),
            qre.Deallocate(1),
        ]
        assert qre.Select.resource_decomp(cmpr_ops, num_wires=4) == expected

    def test_textbook_resources(self):
        """Test that the textbook resources are correct."""
        ops = [qre.X()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        from collections import defaultdict

        expected = defaultdict(int)
        expected[qre.CompressedResourceOp(qre.X, 1)] = 0
        expected[
            qre.CompressedResourceOp(
                qre.Controlled,
                num_wires=1,
                params={
                    "base_cmpr_op": qre.CompressedResourceOp(qre.X, num_wires=1),
                    "num_ctrl_wires": 0,
                    "num_zero_ctrl": 0,
                },
            )
        ] = 1

        assert qre.Select.textbook_resources(cmpr_ops) == expected


class TestResourceQROM:
    """Test the resource QROM class."""

    def test_select_swap_depth_errors(self):
        """Test that the correct error is raised when invalid values of
        select_swap_depth are provided.
        """
        select_swap_depth = "Not A Valid Input"
        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            qre.QROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            qre.QROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

        select_swap_depth = 3
        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            qre.QROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            qre.QROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_params(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored
    ):
        """Test that the resource params are correct."""
        if depth is None:
            op = qre.QROM(num_data_points, size_data_points)
        else:
            op = qre.QROM(num_data_points, size_data_points, num_bit_flips, restored, depth)

        assert op.resource_params == {
            "num_bitstrings": num_data_points,
            "size_bitstring": size_data_points,
            "num_bit_flips": num_bit_flips,
            "select_swap_depth": depth,
            "restored": restored,
        }

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_rep(self, num_data_points, size_data_points, num_bit_flips, depth, restored):
        """Test that the compressed representation is correct."""
        expected_num_wires = size_data_points + qml.math.ceil_log2(num_data_points)
        expected = qre.CompressedResourceOp(
            qre.QROM,
            expected_num_wires,
            {
                "num_bitstrings": num_data_points,
                "size_bitstring": size_data_points,
                "num_bit_flips": num_bit_flips,
                "select_swap_depth": depth,
                "restored": restored,
            },
        )
        assert (
            qre.QROM.resource_rep(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                restored=restored,
                select_swap_depth=depth,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    qre.Allocate(5),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 14),
                    GateCount(qre.CNOT.resource_rep(), 36),
                    GateCount(qre.TemporaryAND.resource_rep(), 6),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        6,
                    ),
                    qre.Deallocate(2),
                    GateCount(qre.CSWAP.resource_rep(), 12),
                    qre.Deallocate(3),
                ],
            ),
            (
                100,
                5,
                50,
                2,
                False,
                [
                    qre.Allocate(10),
                    GateCount(qre.X.resource_rep(), 97),
                    GateCount(qre.CNOT.resource_rep(), 98),
                    GateCount(qre.TemporaryAND.resource_rep(), 48),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        48,
                    ),
                    qre.Deallocate(5),
                    GateCount(qre.CSWAP.resource_rep(), 5),
                    GateCount(qre.X.resource_rep(), 5),
                    qre.Deallocate(5),
                ],
            ),
            (
                12,
                2,
                5,
                1,
                True,
                [
                    qre.Allocate(3),
                    GateCount(qre.X.resource_rep(), 21),
                    GateCount(qre.CNOT.resource_rep(), 15),
                    GateCount(qre.TemporaryAND.resource_rep(), 10),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        10,
                    ),
                    qre.Deallocate(3),
                ],
            ),
            (
                12,
                2,
                5,
                128,  # This will get truncated to 16 as the max depth
                False,
                [
                    qre.Allocate(30),
                    GateCount(qre.X.resource_rep(), 5),
                    GateCount(qre.CSWAP.resource_rep(), 30),
                    GateCount(qre.X.resource_rep(), 30),
                    qre.Deallocate(30),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    qre.Allocate(30),
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.X.resource_rep(), 10),
                    GateCount(qre.CSWAP.resource_rep(), 120),
                    qre.Deallocate(30),
                ],
            ),
        ),
    )
    def test_resources(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res
    ):
        """Test that the resources are correct."""
        assert (
            qre.QROM.resource_decomp(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                restored=restored,
                select_swap_depth=depth,
            )
            == expected_res
        )

    # pylint: disable=protected-access
    def test_t_select_swap_width(self):
        """Test that the private function doesn't give negative or
        fractional values for the depth"""
        num_bitstrings = 8
        size_bitstring = 17

        opt_width = qre.QROM._t_optimized_select_swap_width(
            num_bitstrings,
            size_bitstring,
        )
        assert opt_width == 1

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    qre.Allocate(6),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 38),
                    GateCount(qre.TemporaryAND.resource_rep(), 8),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        8,
                    ),
                    qre.Deallocate(3),
                    qre.Allocate(1),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.CSWAP.resource_rep(), 12),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                    qre.Deallocate(3),
                ],
            ),
            (
                10,
                3,
                15,
                1,
                True,
                [
                    qre.Allocate(4),
                    GateCount(qre.X.resource_rep(), 18),
                    GateCount(qre.CNOT.resource_rep(), 24),
                    GateCount(qre.TemporaryAND.resource_rep(), 9),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        9,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    qre.Allocate(30),
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.X.resource_rep(), 10),
                    qre.Allocate(1),
                    GateCount(qre.TemporaryAND.resource_rep(), 4),
                    GateCount(qre.CSWAP.resource_rep(), 120),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        4,
                    ),
                    qre.Deallocate(1),
                    qre.Deallocate(30),
                ],
            ),
        ),
    )
    def test_single_controlled_res_decomp(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res
    ):
        """Test that the resources computed by single_controlled_res_decomp are correct."""
        assert (
            qre.QROM.single_controlled_res_decomp(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                restored=restored,
                select_swap_depth=depth,
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                1,
                0,
                10,
                3,
                15,
                None,
                True,
                [
                    qre.Allocate(6),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 38),
                    GateCount(qre.TemporaryAND.resource_rep(), 8),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        8,
                    ),
                    qre.Deallocate(3),
                    qre.Allocate(1),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.CSWAP.resource_rep(), 12),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                    qre.Deallocate(3),
                ],
            ),
            (
                2,
                1,
                10,
                3,
                15,
                1,
                True,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    qre.Allocate(1),
                    GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
                    qre.Allocate(4),
                    GateCount(qre.X.resource_rep(), 18),
                    GateCount(qre.CNOT.resource_rep(), 24),
                    GateCount(qre.TemporaryAND.resource_rep(), 9),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        9,
                    ),
                    qre.Deallocate(4),
                    GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
                    qre.Deallocate(1),
                ],
            ),
        ),
    )
    def test_controlled_res_decomp(
        self,
        num_ctrl_wires,
        num_zero_ctrl,
        num_data_points,
        size_data_points,
        num_bit_flips,
        depth,
        restored,
        expected_res,
    ):
        """Test that the resources computed by single_controlled_res_decomp are correct."""
        assert (
            qre.QROM.controlled_resource_decomp(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                target_resource_params={
                    "num_bitstrings": num_data_points,
                    "size_bitstring": size_data_points,
                    "num_bit_flips": num_bit_flips,
                    "restored": restored,
                    "select_swap_depth": depth,
                },
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    qre.Allocate(4),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CZ.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep(), 2),
                    GateCount(qre.X.resource_rep(), 14),
                    GateCount(qre.CNOT.resource_rep(), 16),
                    GateCount(qre.TemporaryAND.resource_rep(), 6),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        6,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                100,
                5,
                50,
                2,
                False,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5),
                    qre.Allocate(7),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.CSWAP.resource_rep(), 1),
                    GateCount(qre.Hadamard.resource_rep(), 1),
                    GateCount(qre.CZ.resource_rep(), 1),
                    GateCount(qre.CNOT.resource_rep(), 1),
                    GateCount(qre.X.resource_rep(), 97),
                    GateCount(qre.CNOT.resource_rep(), 98),
                    GateCount(qre.TemporaryAND.resource_rep(), 48),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        48,
                    ),
                    qre.Deallocate(7),
                ],
            ),
            (
                12,
                2,
                5,
                1,
                True,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    qre.Allocate(4),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 0),
                    GateCount(qre.Hadamard.resource_rep(), 0),
                    GateCount(qre.CZ.resource_rep(), 0),
                    GateCount(qre.CNOT.resource_rep(), 0),
                    GateCount(qre.X.resource_rep(), 21),
                    GateCount(qre.CNOT.resource_rep(), 16),
                    GateCount(qre.TemporaryAND.resource_rep(), 10),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        10,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                12,
                2,
                5,
                128,  # This will get truncated to 16 as the max depth
                False,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    qre.Allocate(16),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 32),
                    GateCount(qre.CSWAP.resource_rep(), 15),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(qre.CZ.resource_rep(), 15),
                    GateCount(qre.CNOT.resource_rep(), 15),
                    GateCount(qre.X.resource_rep(), 8),
                    qre.Deallocate(16),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    qre.Allocate(16),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 30),
                    GateCount(qre.Hadamard.resource_rep(), 30),
                    GateCount(qre.CZ.resource_rep(), 30),
                    GateCount(qre.CNOT.resource_rep(), 30),
                    GateCount(qre.X.resource_rep(), 16),
                    qre.Deallocate(16),
                ],
            ),
        ),
    )
    def test_adjoint_resources(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res
    ):
        """Test that the resources are correct."""

        assert (
            qre.QROM.adjoint_resource_decomp(
                {
                    "num_bitstrings": num_data_points,
                    "size_bitstring": size_data_points,
                    "num_bit_flips": num_bit_flips,
                    "restored": restored,
                    "select_swap_depth": depth,
                }
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_data_points, output_size, restored, depth",
        (
            (100, 10, False, 2),
            (100, 2, False, 4),
            (12, 1, False, 1),
            (12, 3, True, 1),
            (160, 8, True, 2),
        ),
    )
    def test_toffoli_counts(self, num_data_points, output_size, restored, depth):
        """Test that the Toffoli counts are correct compared to arXiv:1092.02134."""

        qrom = qre.Adjoint(
            qre.QROM(
                num_bitstrings=num_data_points,
                size_bitstring=output_size,
                restored=restored,
                select_swap_depth=depth,
            )
        )
        resources = qre.estimate(qrom)

        toffoli_count = int(math.ceil(num_data_points / depth)) + depth - 3
        if restored and depth > 1:
            toffoli_count *= 2

        assert resources.gate_counts["Toffoli"] == toffoli_count


class TestResourceSelectPauliRot:
    """Test the ResourceSelectPauliRot template"""

    def test_rot_axis_errors(self):
        """Test that the correct error is raised when invalid rotation axis argument is provided."""
        with pytest.raises(ValueError, match="The `rot_axis` argument must be one of"):
            qre.SelectPauliRot(rot_axis="A", num_ctrl_wires=1, precision=1e-3)

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("rot_axis", ("X", "Y", "Z"))
    @pytest.mark.parametrize("num_ctrl_wires", (1, 2, 3, 4, 5))
    def test_resource_params(self, num_ctrl_wires, rot_axis, precision):
        """Test that the resource params are correct."""
        op = (
            qre.SelectPauliRot(rot_axis, num_ctrl_wires, precision)
            if precision
            else qre.SelectPauliRot(rot_axis, num_ctrl_wires)
        )
        assert op.resource_params == {
            "rot_axis": rot_axis,
            "num_ctrl_wires": num_ctrl_wires,
            "precision": precision,
        }

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("rot_axis", ("X", "Y", "Z"))
    @pytest.mark.parametrize("num_ctrl_wires", (1, 2, 3, 4, 5))
    def test_resource_rep(self, num_ctrl_wires, rot_axis, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.SelectPauliRot,
            num_ctrl_wires + 1,
            {
                "rot_axis": rot_axis,
                "num_ctrl_wires": num_ctrl_wires,
                "precision": precision,
            },
        )
        assert qre.SelectPauliRot.resource_rep(num_ctrl_wires, rot_axis, precision) == expected

    @pytest.mark.parametrize(
        "num_ctrl_wires, rot_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    GateCount(resource_rep(qre.RX, {"precision": 1e-9}), 2),
                    GateCount(resource_rep(qre.CNOT), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    GateCount(resource_rep(qre.RY, {"precision": 1e-3}), 2**2),
                    GateCount(resource_rep(qre.CNOT), 2**2),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-5}), 2**5),
                    GateCount(resource_rep(qre.CNOT), 2**5),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_ctrl_wires, rot_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qre.SelectPauliRot]
            assert (
                qre.SelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rot_axis=rot_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                qre.SelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rot_axis=rot_axis,
                    precision=precision,
                )
                == expected_res
            )

    @pytest.mark.parametrize(
        "num_ctrl_wires, rot_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    Allocate(33),
                    GateCount(qre.QROM.resource_rep(2, 33, 33, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(33),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(2, 33, 33, False),
                            },
                        )
                    ),
                    Deallocate(33),
                    GateCount(resource_rep(qre.Hadamard), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    Allocate(13),
                    GateCount(qre.QROM.resource_rep(4, 13, 26, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(13),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(4, 13, 26, False),
                            },
                        )
                    ),
                    Deallocate(13),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.S)),
                    GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)})),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    Allocate(20),
                    GateCount(qre.QROM.resource_rep(32, 20, 320, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(20),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(32, 20, 320, False),
                            },
                        )
                    ),
                    Deallocate(20),
                ],
            ),
        ),
    )
    def test_phase_gradient_resources(self, num_ctrl_wires, rot_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qre.SelectPauliRot]
            assert (
                qre.SelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rot_axis=rot_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                qre.SelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rot_axis=rot_axis,
                    precision=precision,
                )
                == expected_res
            )


class TestResourceUnaryIterationQPE:
    """Test the UnaryIterationQPE class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        walk_op = qre.QubitizeTHC(thc_ham=qre.THCHamiltonian(num_orbitals=20, tensor_rank=40))
        with pytest.raises(ValueError, match="Expected 101 wires, got 3"):
            qre.UnaryIterationQPE(walk_op=walk_op, num_iterations=8, wires=[0, 1, 2])

    def test_tracking_name(self):
        """Test that the name of the operator is tracked correctly."""
        walk_op = qre.QubitizeTHC(thc_ham=qre.THCHamiltonian(num_orbitals=20, tensor_rank=40))
        walk_op_name = walk_op.resource_rep_from_op().name
        res_params = walk_op.resource_params

        op = qre.UnaryIterationQPE(walk_op=walk_op, num_iterations=8, adj_qft_op=qre.QFT(3))
        assert (
            op.tracking_name(
                resource_rep(qre.QubitizeTHC, res_params),
                8,
                resource_rep(qre.QFT, {"num_wires": 3}),
            )
            == f"UnaryIterationQPE({walk_op_name}, 8, adj_qft=QFT(3))"
        )

    @pytest.mark.parametrize(
        "walk_op, adj_qft, input_wires, expected_wires",
        (
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                ),
                qre.Adjoint(qre.QFT(4)),
                None,
                None,
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                ),
                qre.Adjoint(qre.QFT(4)),
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                    wires=[1, 2, 3, 4],
                ),
                qre.Adjoint(qre.QFT(4, ["c1", "c2", "c3", "c4"])),
                None,
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                    wires=[1, 2, 3, 4],
                ),
                qre.Adjoint(qre.QFT(4)),
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                ),
                qre.Adjoint(qre.QFT(4, wires=["c1", "c2", "c3", "c4"])),
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
                Wires([1, 2, 3, 4, "c1", "c2", "c3", "c4"]),
            ),
        ),
    )
    def test_wires_init(self, walk_op, adj_qft, input_wires, expected_wires):
        """Test that we can correctly initialize the wires of the operator"""
        op = qre.UnaryIterationQPE(
            walk_op=walk_op,
            num_iterations=11,
            adj_qft_op=adj_qft,
            wires=input_wires,
        )
        assert op.wires == expected_wires

    @pytest.mark.parametrize(
        "walk_op, n_iter, error_message",
        (
            (
                qre.QubitizeTHC(qre.THCHamiltonian(40, 10)),
                0,
                "Expected 'num_iterations' to be an integer greater than zero,",
            ),
            (
                qre.QubitizeTHC(qre.THCHamiltonian(40, 10)),
                3.5,
                "Expected 'num_iterations' to be an integer greater than zero,",
            ),
            (
                qre.QubitizeTHC(qre.THCHamiltonian(40, 10)),
                -2,
                "Expected 'num_iterations' to be an integer greater than zero,",
            ),
            (qre.RZ(), 4, "Expected the 'walk_op' to be a qubitization type operator "),
        ),
    )
    def test_init_errors(self, walk_op, n_iter, error_message):
        """Test that Value errors are raised when incompatible inputs are provided."""
        with pytest.raises(ValueError, match=error_message):
            _ = qre.UnaryIterationQPE(walk_op, n_iter)

    @pytest.mark.parametrize(
        "walk_operator, n_iter, adj_qft",
        (
            (qre.QubitizeTHC(thc_ham=qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)), 5, None),
            (
                qre.QubitizeTHC(thc_ham=qre.THCHamiltonian(num_orbitals=10, tensor_rank=15)),
                3,
                qre.QFT(2),
            ),
            (
                qre.Qubitization(qre.UniformStatePrep(3), qre.Select([qre.X(), qre.Y(), qre.Z()])),
                4,
                qre.Adjoint(qre.AQFT(3, 2)),
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                ),
                4,
                qre.Adjoint(qre.AQFT(3, 2)),
            ),
        ),
    )
    def test_resource_params(self, walk_operator, n_iter, adj_qft):
        """Test the resource_params method"""
        walk_operator_cmpr = walk_operator.resource_rep_from_op()

        if adj_qft is None:
            op = qre.UnaryIterationQPE(walk_operator, n_iter)
            adj_qft_cmpr = None
        else:
            op = qre.UnaryIterationQPE(walk_operator, n_iter, adj_qft)
            adj_qft_cmpr = adj_qft.resource_rep_from_op()

        assert op.resource_params == {
            "cmpr_walk_op": walk_operator_cmpr,
            "num_iterations": n_iter,
            "adj_qft_cmpr_op": adj_qft_cmpr,
        }

    @pytest.mark.parametrize(
        "walk_operator_cmpr, n_iter, adj_qft_cmpr",
        (
            (
                qre.QubitizeTHC(
                    thc_ham=qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
                ).resource_rep_from_op(),
                5,
                None,
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3), qre.Select([qre.X(), qre.Y(), qre.Z()])
                ).resource_rep_from_op(),
                3,
                qre.QFT.resource_rep(2),
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                ).resource_rep_from_op(),
                4,
                qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 2)),
            ),
        ),
    )
    def test_resource_rep(self, walk_operator_cmpr, n_iter, adj_qft_cmpr):
        """Test the resource_rep method"""
        num_estimation_wires = qml.math.ceil_log2(n_iter + 1)
        expected_num_wires = walk_operator_cmpr.num_wires + num_estimation_wires

        expected = qre.CompressedResourceOp(
            qre.UnaryIterationQPE,
            expected_num_wires,
            {
                "cmpr_walk_op": walk_operator_cmpr,
                "num_iterations": n_iter,
                "adj_qft_cmpr_op": adj_qft_cmpr,
            },
        )

        assert (
            qre.UnaryIterationQPE.resource_rep(walk_operator_cmpr, n_iter, adj_qft_cmpr) == expected
        )

    @pytest.mark.parametrize(
        "walk_operator, n_iter, adj_qft_op, expected_res",
        (
            (
                qre.QubitizeTHC(thc_ham=qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)),
                5,
                None,
                [
                    qre.Allocate(2),
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    GateCount(resource_rep(qre.Toffoli, {"elbow": "left"}), 4),
                    GateCount(qre.CNOT.resource_rep(), 4),
                    GateCount(qre.X.resource_rep(), 10),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Reflection.resource_rep(
                                num_wires=qre.PrepTHC(
                                    qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
                                ).num_wires,
                                alpha=math.pi,
                                cmpr_U=qre.PrepTHC(
                                    qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
                                ).resource_rep_from_op(),
                            ),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        6,
                    ),
                    GateCount(
                        qre.SelectTHC(
                            qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
                        ).resource_rep_from_op(),
                        5,
                    ),
                    GateCount(resource_rep(qre.Toffoli, {"elbow": "right"}), 4),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.QFT.resource_rep(3)),
                    ),
                    qre.Deallocate(2),
                ],
            ),
            (
                qre.Qubitization(qre.UniformStatePrep(3), qre.Select([qre.X(), qre.Y(), qre.Z()])),
                3,
                qre.QFT(2),
                [
                    qre.Allocate(1),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(resource_rep(qre.Toffoli, {"elbow": "left"}), 2),
                    GateCount(qre.CNOT.resource_rep(), 2),
                    GateCount(qre.X.resource_rep(), 6),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Reflection.resource_rep(
                                num_wires=2,
                                alpha=math.pi,
                                cmpr_U=qre.UniformStatePrep(3).resource_rep_from_op(),
                            ),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    GateCount(qre.Select([qre.X(), qre.Y(), qre.Z()]).resource_rep_from_op(), 3),
                    GateCount(resource_rep(qre.Toffoli, {"elbow": "right"}), 2),
                    GateCount(qre.QFT.resource_rep(2)),
                    qre.Deallocate(1),
                ],
            ),
            (
                qre.Qubitization(
                    qre.UniformStatePrep(3),
                    qre.SelectPauli(qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})),
                ),
                4,
                qre.Adjoint(qre.AQFT(3, 2)),
                [
                    qre.Allocate(2),
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    GateCount(resource_rep(qre.Toffoli, {"elbow": "left"}), 3),
                    GateCount(qre.CNOT.resource_rep(), 3),
                    GateCount(qre.X.resource_rep(), 8),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Reflection.resource_rep(
                                num_wires=2,
                                alpha=math.pi,
                                cmpr_U=qre.UniformStatePrep(3).resource_rep_from_op(),
                            ),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        5,
                    ),
                    GateCount(
                        qre.SelectPauli(
                            qre.PauliHamiltonian(2, {"XX": 1, "Z": 1, "Y": 1})
                        ).resource_rep_from_op(),
                        4,
                    ),
                    GateCount(resource_rep(qre.Toffoli, {"elbow": "right"}), 3),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 2)),
                    ),
                    qre.Deallocate(2),
                ],
            ),
        ),
    )
    def test_resources(self, walk_operator, n_iter, adj_qft_op, expected_res):
        """Test that resources method is correct"""
        op = (
            qre.UnaryIterationQPE(walk_operator, n_iter)
            if adj_qft_op is None
            else qre.UnaryIterationQPE(walk_operator, n_iter, adj_qft_op)
        )
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceReflection:
    """Test the Reflection class."""

    def test_init_raises_error(self):
        """Test that an error is raised when neither num_wires nor U is provided."""
        with pytest.raises(ValueError, match="Must provide at least one of `num_wires` or `U`"):
            qre.Reflection()

        with pytest.raises(ValueError, match="Must provide atleast one of `num_wires` or `U`"):
            qre.Reflection.resource_rep()

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 3 wires, got 2"):
            qre.Reflection(num_wires=3, U=qre.QFT(3), wires=[0, 1])

    def test_init_with_U_no_num_wires(self):
        """Test that we can instantiate the operator with U but without providing num_wires."""
        U = qre.QFT(3)
        op = qre.Reflection(U=U)
        assert op.num_wires == 3
        assert op.cmpr_U == U.resource_rep_from_op()

    def test_init_with_num_wires_no_U(self):
        """Test that we can instantiate the operator with num_wires but without providing U."""
        op = qre.Reflection(num_wires=1)
        assert op.num_wires == 1
        assert op.cmpr_U == qre.Identity.resource_rep()

    @pytest.mark.parametrize("alpha", (-1, 7))
    def test_init_alpha_error(self, alpha):
        """Test that an error is raised if the alpha is provided outside of the expected range"""
        with pytest.raises(ValueError, match="alpha must be within"):
            _ = qre.Reflection(num_wires=1, alpha=alpha)

        with pytest.raises(ValueError, match="alpha must be within"):
            _ = qre.Reflection.resource_rep(num_wires=1, alpha=alpha)

    @pytest.mark.parametrize(
        "U, alpha",
        (
            (qre.QFT(3), math.pi),
            (qre.AQFT(2, 5), math.pi / 2),
            (qre.Hadamard(), 0),
            (qre.Identity(), math.pi),
        ),
    )
    def test_resource_params(self, U, alpha):
        """Test that the resource params are correct."""
        op = qre.Reflection(U=U, alpha=alpha)
        cmpr_U = U.resource_rep_from_op()

        assert op.resource_params == {
            "alpha": alpha,
            "num_wires": cmpr_U.num_wires,
            "cmpr_U": cmpr_U,
        }

    @pytest.mark.parametrize(
        "num_wires, cmpr_U, alpha",
        (
            (3, qre.QFT.resource_rep(3), math.pi),
            (5, qre.AQFT.resource_rep(2, 5), math.pi / 2),
            (2, qre.Hadamard.resource_rep(), 0),
            (4, qre.Identity.resource_rep(), math.pi),
        ),
    )
    def test_resource_rep(self, num_wires, cmpr_U, alpha):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.Reflection,
            num_wires,
            {"alpha": alpha, "num_wires": num_wires, "cmpr_U": cmpr_U},
        )
        assert qre.Reflection.resource_rep(num_wires, alpha, cmpr_U) == expected

    @pytest.mark.parametrize(
        "num_wires, cmpr_U, alpha, expected_res",
        (
            # alpha = 0 case: just global phase
            (
                3,
                qre.QFT.resource_rep(3),
                0,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                ],
            ),
            # alpha = 2*pi case: just global phase
            (
                3,
                qre.QFT.resource_rep(3),
                2 * math.pi,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                ],
            ),
            # alpha = pi, num_wires > 1 case
            (
                3,
                qre.QFT.resource_rep(3),
                math.pi,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.QFT.resource_rep(3)),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Z.resource_rep(),
                            num_ctrl_wires=2,
                            num_zero_ctrl=2,
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.QFT.resource_rep(3))),
                ],
            ),
            # alpha = pi, num_wires = 1 case
            (
                1,
                qre.Hadamard.resource_rep(),
                math.pi,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep()),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.Adjoint.resource_rep(qre.Hadamard.resource_rep())),
                ],
            ),
            # alpha != pi case (uses PhaseShift)
            (
                2,
                qre.CNOT.resource_rep(),
                math.pi / 2,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep()),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.PhaseShift.resource_rep(),
                            num_ctrl_wires=1,
                            num_zero_ctrl=1,
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.CNOT.resource_rep())),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, cmpr_U, alpha, expected_res):
        """Test that the resources are correct."""
        assert (
            qre.Reflection.resource_decomp(num_wires=num_wires, alpha=alpha, cmpr_U=cmpr_U)
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_wires, cmpr_U, alpha",
        (
            (3, qre.QFT.resource_rep(3), math.pi),
            (2, qre.Hadamard.resource_rep(), math.pi / 2),
        ),
    )
    def test_adjoint_resources(self, num_wires, cmpr_U, alpha):
        """Test that the adjoint resources are correct (reflection is self-adjoint)."""
        target_params = {"num_wires": num_wires, "cmpr_U": cmpr_U, "alpha": alpha}
        expected = [GateCount(qre.Reflection.resource_rep(num_wires, alpha, cmpr_U))]
        assert qre.Reflection.adjoint_resource_decomp(target_params) == expected

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_wires, cmpr_U, alpha, expected_res",
        (
            # alpha = 0 case: just controlled global phase
            (
                1,
                0,
                3,
                qre.QFT.resource_rep(3),
                0,
                [
                    GateCount(qre.MultiControlledX.resource_rep(1, 0), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                ],
            ),
            # alpha = pi, all zero controls
            (
                2,
                2,
                3,
                qre.QFT.resource_rep(3),
                math.pi,
                [
                    GateCount(qre.MultiControlledX.resource_rep(2, 2), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.QFT.resource_rep(3)),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Z.resource_rep(),
                            num_ctrl_wires=4,  # num_wires - 1 + num_ctrl_wires
                            num_zero_ctrl=4,  # num_wires - 1 + num_zero_ctrl
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.QFT.resource_rep(3))),
                ],
            ),
            # alpha = pi, not all zero controls (absorbs X into control)
            (
                2,
                1,
                3,
                qre.QFT.resource_rep(3),
                math.pi,
                [
                    GateCount(qre.MultiControlledX.resource_rep(2, 1), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.QFT.resource_rep(3)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Z.resource_rep(),
                            num_ctrl_wires=4,  # num_wires - 1 + num_ctrl_wires
                            num_zero_ctrl=4,  # num_wires - 1 + num_zero_ctrl + 1
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.QFT.resource_rep(3))),
                ],
            ),
            # alpha != pi case (uses PhaseShift)
            (
                1,
                1,
                2,
                qre.CNOT.resource_rep(),
                math.pi / 2,
                [
                    GateCount(qre.MultiControlledX.resource_rep(1, 1), 2),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep()),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.PhaseShift.resource_rep(),
                            num_ctrl_wires=2,
                            num_zero_ctrl=2,
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.CNOT.resource_rep())),
                ],
            ),
        ),
    )
    def test_controlled_resources(
        self, num_ctrl_wires, num_zero_ctrl, num_wires, cmpr_U, alpha, expected_res
    ):
        """Test that the controlled resources are correct."""
        target_params = {"num_wires": num_wires, "cmpr_U": cmpr_U, "alpha": alpha}
        assert (
            qre.Reflection.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, target_params)
            == expected_res
        )


class TestResourceQubitization:
    """Test the Qubitization class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        prep = qre.QFT(3)
        sel = qre.Select(
            [qre.X(), qre.Y(), qre.Z()]
        )  # has 4 wires (3 ops + 2 ctrl, but ops share wires)
        with pytest.raises(ValueError, match="Expected .* wires, got"):
            qre.Qubitization(prep, sel, wires=[0])

    def test_init_wires_inherited(self):
        """Test that wires are inherited from prep and sel when possible."""
        prep = qre.QFT(2, wires=[0, 1])
        sel = qre.Select([qre.X(wires=2), qre.Y(wires=2)], wires=[3, 2])
        op = qre.Qubitization(prep, sel)
        assert op.num_wires == sel.num_wires

    def test_init_wires_inherited_success(self):
        """Test that wires are inherited from prep and sel when they match num_wires."""
        prep = qre.Hadamard(wires=0)
        sel = qre.Select([qre.X(wires=1), qre.Z(wires=1)], wires=[0])
        op = qre.Qubitization(prep, sel)

        assert op.wires == qml.wires.Wires([0, 1])
        assert op.num_wires == 2

    @pytest.mark.parametrize(
        "prep, sel",
        (
            (qre.QFT(3), qre.Select([qre.X(), qre.Y(), qre.Z()])),
            (qre.AQFT(2, 4), qre.Select([qre.RX(), qre.RY()])),
            (qre.Hadamard(), qre.Select([qre.Z()])),
        ),
    )
    def test_resource_params(self, prep, sel):
        """Test that the resource params are correct."""
        op = qre.Qubitization(prep, sel)
        assert op.resource_params == {
            "prep_op": prep.resource_rep_from_op(),
            "select_op": sel.resource_rep_from_op(),
        }

    @pytest.mark.parametrize(
        "prep_cmpr, sel_cmpr",
        (
            (
                qre.QFT.resource_rep(3),
                qre.Select.resource_rep(
                    (qre.X.resource_rep(), qre.Y.resource_rep(), qre.Z.resource_rep()), 4
                ),
            ),
            (
                qre.AQFT.resource_rep(2, 4),
                qre.Select.resource_rep((qre.RX.resource_rep(), qre.RY.resource_rep()), 3),
            ),
            (qre.Hadamard.resource_rep(), qre.Select.resource_rep((qre.Z.resource_rep(),), 1)),
        ),
    )
    def test_resource_rep(self, prep_cmpr, sel_cmpr):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.Qubitization,
            sel_cmpr.num_wires,
            {"prep_op": prep_cmpr, "select_op": sel_cmpr},
        )
        assert qre.Qubitization.resource_rep(prep_cmpr, sel_cmpr) == expected

    @pytest.mark.parametrize(
        "prep_cmpr, sel_cmpr",
        (
            (
                qre.QFT.resource_rep(3),
                qre.Select.resource_rep(
                    (qre.X.resource_rep(), qre.Y.resource_rep(), qre.Z.resource_rep()), 4
                ),
            ),
            (
                qre.Hadamard.resource_rep(),
                qre.Select.resource_rep((qre.Z.resource_rep(),), 1),
            ),
        ),
    )
    def test_resources(self, prep_cmpr, sel_cmpr):
        """Test that the resources are correct."""
        ref_op = qre.Reflection.resource_rep(
            num_wires=prep_cmpr.num_wires,
            alpha=math.pi,
            cmpr_U=prep_cmpr,
        )
        expected_res = [
            GateCount(sel_cmpr),
            GateCount(ref_op),
        ]
        assert qre.Qubitization.resource_decomp(prep_cmpr, sel_cmpr) == expected_res

    @pytest.mark.parametrize(
        "prep_cmpr, sel_cmpr",
        (
            (
                qre.QFT.resource_rep(3),
                qre.Select.resource_rep(
                    (qre.X.resource_rep(), qre.Y.resource_rep(), qre.Z.resource_rep()), 4
                ),
            ),
            (qre.Hadamard.resource_rep(), qre.Select.resource_rep((qre.Z.resource_rep(),), 1)),
        ),
    )
    def test_adjoint_resources(self, prep_cmpr, sel_cmpr):
        """Test that the adjoint resources are correct."""
        target_params = {"prep_op": prep_cmpr, "select_op": sel_cmpr}
        ref_op = qre.Reflection.resource_rep(
            num_wires=prep_cmpr.num_wires, alpha=math.pi, cmpr_U=prep_cmpr
        )

        expected = [
            GateCount(ref_op),
            GateCount(sel_cmpr),
        ]
        assert qre.Qubitization.adjoint_resource_decomp(target_params) == expected

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, prep_cmpr, sel_cmpr",
        (
            (
                1,
                0,
                qre.QFT.resource_rep(3),
                qre.Select.resource_rep((qre.X.resource_rep(), qre.Y.resource_rep()), 3),
            ),
            (
                2,
                1,
                qre.Hadamard.resource_rep(),
                qre.Select.resource_rep((qre.Z.resource_rep(),), 1),
            ),
        ),
    )
    def test_controlled_resources(self, num_ctrl_wires, num_zero_ctrl, prep_cmpr, sel_cmpr):
        """Test that the controlled resources are correct."""
        target_params = {"prep_op": prep_cmpr, "select_op": sel_cmpr}
        ref_op = qre.Reflection.resource_rep(
            num_wires=prep_cmpr.num_wires, alpha=math.pi, cmpr_U=prep_cmpr
        )

        ctrl_sel = qre.Controlled.resource_rep(
            base_cmpr_op=sel_cmpr,
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        ctrl_ref = qre.Controlled.resource_rep(
            base_cmpr_op=ref_op,
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )

        expected = [
            GateCount(ctrl_sel),
            GateCount(ctrl_ref),
        ]
        assert (
            qre.Qubitization.controlled_resource_decomp(
                num_ctrl_wires, num_zero_ctrl, target_params
            )
            == expected
        )
