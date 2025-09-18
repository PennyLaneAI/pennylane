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
This module contains tests for the Resources container class.
"""
from collections import defaultdict
from dataclasses import dataclass

import pytest

from pennylane.labs.resource_estimation.qubit_manager import QubitManager
from pennylane.labs.resource_estimation.resources_base import (
    Resources,
    _combine_dict,
    _scale_dict,
    add_in_parallel,
    add_in_series,
    mul_in_parallel,
    mul_in_series,
)

# pylint: disable= no-self-use,too-few-public-methods,comparison-with-itself


@dataclass(frozen=True)
class DummyResOp:
    """A dummy class to populate the gate types dictionary for testing."""

    name: str


h = DummyResOp("Hadamard")
x = DummyResOp("X")
y = DummyResOp("Y")
z = DummyResOp("Z")
cnot = DummyResOp("CNOT")

gate_types_data = (
    defaultdict(
        int,
        {h: 2, x: 1, z: 1},
    ),
    defaultdict(
        int,
        {h: 467, cnot: 791},
    ),
    defaultdict(
        int,
        {x: 100, y: 120, z: 1000, cnot: 4523},
    ),
)

qm1 = QubitManager(work_wires=5)
qm2 = QubitManager(work_wires={"clean": 8753, "dirty": 2347}, algo_wires=22)
qm3 = QubitManager(work_wires={"clean": 400, "dirty": 222}, algo_wires=108)

qubit_manager_data = (qm1, qm2, qm3)


class TestResources:
    """Test the Resources class"""

    @pytest.mark.parametrize("gt", gate_types_data + (None,))
    @pytest.mark.parametrize("qm", qubit_manager_data)
    def test_init(self, qm, gt):
        """Test that the class is correctly initialized"""
        resources = Resources(qubit_manager=qm, gate_types=gt)

        expected_qm = qm
        expected_gt = defaultdict(int, {}) if gt is None else gt

        assert resources.qubit_manager == expected_qm
        assert resources.gate_types == expected_gt

    str_data = (
        (
            "--- Resources: ---\n"
            + " Total qubits: 5\n"
            + " Total gates : 4\n"
            + " Qubit breakdown:\n"
            + "  clean qubits: 5, dirty qubits: 0, algorithmic qubits: 0\n"
            + " Gate breakdown:\n"
            + "  {'Hadamard': 2, 'X': 1, 'Z': 1}"
        ),
        (
            "--- Resources: ---\n"
            + " Total qubits: 1.112E+4\n"
            + " Total gates : 1.258E+3\n"
            + " Qubit breakdown:\n"
            + "  clean qubits: 8753, dirty qubits: 2347, algorithmic qubits: 22\n"
            + " Gate breakdown:\n"
            + "  {'Hadamard': 467, 'CNOT': 791}"
        ),
        (
            "--- Resources: ---\n"
            + " Total qubits: 730\n"
            + " Total gates : 5.743E+3\n"
            + " Qubit breakdown:\n"
            + "  clean qubits: 400, dirty qubits: 222, algorithmic qubits: 108\n"
            + " Gate breakdown:\n"
            + "  {'X': 100, 'Y': 120, 'Z': 1.000E+3, 'CNOT': 4.523E+3}"
        ),
    )

    @pytest.mark.parametrize(
        "resources, expected_str",
        zip(
            tuple(Resources(qm, gt) for qm, gt in zip(qubit_manager_data, gate_types_data)),
            str_data,
        ),
    )
    def test_str_method(self, resources, expected_str):
        """Test that the str method correctly displays the information."""
        assert str(resources) == expected_str

    @pytest.mark.parametrize("gt", gate_types_data + (None,))
    @pytest.mark.parametrize("qm", qubit_manager_data)
    def test_repr_method(self, gt, qm):
        """Test that the repr method correctly represents the class."""
        resources = Resources(qubit_manager=qm, gate_types=gt)

        expected_qm = qm
        expected_gt = defaultdict(int, {}) if gt is None else gt
        assert repr(resources) == repr({"qubit_manager": expected_qm, "gate_types": expected_gt})

    def test_clean_gate_counts(self):
        """Test that this function correctly simplifies the gate types
        dictionary by grouping together gates with the same name."""

        class DummyResOp2:
            """A dummy class to populate the gate types dictionary for testing."""

            def __init__(self, name, parameter=None):
                """Initialize dummy class."""
                self.name = name
                self.parameter = parameter

            def __hash__(self):
                """Custom hash which only depends on instance name."""
                return hash((self.name, self.parameter))

        rx1 = DummyResOp2("RX", parameter=3.14)
        rx2 = DummyResOp2("RX", parameter=3.14 / 2)
        cnots = DummyResOp2("CNOT")
        ry1 = DummyResOp2("RY", parameter=3.14)
        ry2 = DummyResOp2("RY", parameter=3.14 / 4)

        gate_types = {rx1: 1, ry1: 2, cnots: 3, rx2: 4, ry2: 5}
        res = Resources(qubit_manager=qm1, gate_types=gate_types)

        expected_clean_gate_counts = {"RX": 5, "RY": 7, "CNOT": 3}
        assert res.clean_gate_counts == expected_clean_gate_counts

    def test_equality(self):
        """Test that the equality method works as expected."""
        gt1, gt2 = (gate_types_data[0], gate_types_data[1])

        res1 = Resources(qubit_manager=qm1, gate_types=gt1)
        res1_copy = Resources(qubit_manager=qm1, gate_types=gt1)
        res2 = Resources(qubit_manager=qm2, gate_types=gt2)

        assert res1 == res1
        assert res1 == res1_copy
        assert res1 != res2

    def test_arithmetic_raises_error(self):
        """Test that an assertion error is raised when arithmetic methods are used"""
        res = Resources(qubit_manager=qm1, gate_types=gate_types_data[0])

        with pytest.raises(AssertionError):
            _ = res + 2  # Can only add two Resources instances

        with pytest.raises(AssertionError):
            _ = res & 2  # Can only add two Resources instances

        with pytest.raises(AssertionError):
            _ = res * res  # Can only multiply a Resources instance with an int

        with pytest.raises(AssertionError):
            _ = res @ res  # Can only multiply a Resources instance with an int

    def test_add_in_series(self):
        """Test that we can add two resources assuming the gates occur in series"""
        res1 = Resources(qubit_manager=qm3, gate_types=gate_types_data[2])
        res2 = Resources(qubit_manager=qm2, gate_types=gate_types_data[1])

        expected_qm_add = QubitManager(
            work_wires={
                "clean": 8753,  # max(clean1, clean2)
                "dirty": 2569,  # dirty1 + dirty2
            }
        )
        expected_qm_add.algo_qubits = 108  # max(algo1, algo2)
        expected_gt_add = defaultdict(
            int,
            {h: 467, x: 100, y: 120, z: 1000, cnot: 5314},  # add gate counts
        )

        expected_add = Resources(expected_qm_add, expected_gt_add)
        assert (res1 + res2) == expected_add
        assert add_in_series(res1, res2) == expected_add

    def test_add_in_parallel(self):
        """Test that we can add two resources assuming the gates occur in parallel"""
        res1 = Resources(qubit_manager=qm3, gate_types=gate_types_data[2])
        res2 = Resources(qubit_manager=qm2, gate_types=gate_types_data[1])

        expected_qm_and = QubitManager(
            work_wires={
                "clean": 8753,  # max(clean1, clean2)
                "dirty": 2569,  # dirty1 + dirty2
            }
        )
        expected_qm_and.algo_qubits = 130  # algo1 + algo2
        expected_gt_and = defaultdict(
            int,
            {h: 467, x: 100, y: 120, z: 1000, cnot: 5314},  # add gate counts
        )

        expected_and = Resources(expected_qm_and, expected_gt_and)
        assert (res1 & res2) == expected_and
        assert add_in_parallel(res1, res2) == expected_and

    def test_mul_in_series(self):
        """Test that we can scale resources by an integer assuming the gates occur in series"""
        k = 3
        res = Resources(qubit_manager=qm3, gate_types=gate_types_data[2])

        expected_qm_mul = QubitManager(
            work_wires={
                "clean": 400,  # clean
                "dirty": 222 * k,  # k * dirty1
            }
        )
        expected_qm_mul.algo_qubits = 108  # algo
        expected_gt_mul = defaultdict(
            int,
            {x: 100 * k, y: 120 * k, z: 1000 * k, cnot: 4523 * k},  # multiply gate counts
        )

        expected_mul = Resources(expected_qm_mul, expected_gt_mul)

        assert (k * res) == expected_mul
        assert (res * k) == expected_mul
        assert mul_in_series(res, k) == expected_mul

    def test_mul_in_parallel(self):
        """Test that we can scale resources by an integer assuming the gates occur in parallel"""
        k = 3
        res = Resources(qubit_manager=qm3, gate_types=gate_types_data[2])

        expected_qm_matmul = QubitManager(
            work_wires={
                "clean": 400,  # clean
                "dirty": 222 * k,  # k * dirty1
            }
        )
        expected_qm_matmul.algo_qubits = 108 * k  # k * algo
        expected_gt_matmul = defaultdict(
            int,
            {x: 100 * k, y: 120 * k, z: 1000 * k, cnot: 4523 * k},  # multiply gate counts
        )

        expected_matmul = Resources(expected_qm_matmul, expected_gt_matmul)
        assert (k @ res) == expected_matmul
        assert (res @ k) == expected_matmul
        assert mul_in_parallel(res, k) == expected_matmul


def test_combine_dict():
    """Test the private _combine_dict function works as expected"""
    g0 = defaultdict(int, {})
    g1 = defaultdict(int, {"a": 1, "b": 2, "c": 3})
    g2 = defaultdict(int, {"b": -1, "c": 0, "d": 1})

    g_res = defaultdict(int, {"a": 1, "b": 1, "c": 3, "d": 1})
    assert _combine_dict(g0, g1) == g1
    assert _combine_dict(g1, g2) == g_res


@pytest.mark.parametrize(
    "k, expected_dict",
    (
        (0, defaultdict(int, {"a": 0, "b": 0, "c": 0})),
        (1, defaultdict(int, {"a": 1, "b": 2, "c": 3})),
        (3, defaultdict(int, {"a": 3, "b": 6, "c": 9})),
    ),
)
def test_scale_dict(k, expected_dict):
    """Test the private _scale_dict function works as expected"""
    g0 = defaultdict(int, {})
    g1 = defaultdict(int, {"a": 1, "b": 2, "c": 3})

    assert _scale_dict(g0, k) == g0
    assert _scale_dict(g1, k) == expected_dict
