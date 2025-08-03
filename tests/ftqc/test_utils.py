# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ftqc.utils module"""

import itertools as it
import random

import numpy as np
import pytest

from pennylane.ftqc import QubitMgr
from pennylane.ftqc.utils import parity
from pennylane.measurements import MeasurementValue, MidMeasureMP

# pylint: disable=too-few-public-methods, too-many-public-methods


num_qubits_vals = [1, 3, 7]
acquire_num_vals = [0, 1, 5]
offsets_vals = [0, 8, 9]


@pytest.mark.parametrize("arg_type", [tuple, np.array])
@pytest.mark.parametrize(
    "args, expected_outcome",
    [((0, 1, 1), 0), ((0, 0, 1), 1), ((1, 1, 1), 1), ((0, 0, 0, 0), 0), ((0, 1), 1)],
)
def test_parity_numeric(arg_type, args, expected_outcome):
    """Test the parity function behaves as expected"""
    assert parity(*arg_type(args)) == expected_outcome


@pytest.mark.jax
@pytest.mark.parametrize(
    "args, expected_outcome",
    [((0, 1, 1), 0), ((0, 0, 1), 1), ((1, 1, 1), 1), ((0, 0, 0, 0), 0), ((0, 1), 1)],
)
def test_parity_numeric_jax(args, expected_outcome):
    """Test the parity function behaves as expected with jax and jax-jit"""
    import jax

    values = jax.numpy.array(args)

    assert parity(*values) == expected_outcome
    assert jax.jit(parity)(*values) == expected_outcome


@pytest.mark.parametrize(
    "args, expected_outcome",
    [((0, 1, 1), 0), ((0, 0, 1), 1), ((1, 1, 1), 1), ((0, 0, 0, 0), 0), ((0, 1), 1)],
)
def test_parity_measurement_values(args, expected_outcome):
    """Test that passing a sequence of MeasurementValues to the parity function
    returns a single MeasurementValue referencing all the relevant measurements,
    with a processing function that applied the parity check when concrete values
    are provided."""
    mvs = [
        MeasurementValue([MidMeasureMP(0, id=3)], lambda x: x),
        MeasurementValue([MidMeasureMP(1, id=4)], lambda x: x),
        MeasurementValue([MidMeasureMP(2, id=5)], lambda x: x),
        MeasurementValue([MidMeasureMP(3, id=6)], lambda x: x),
    ]

    # only use as many MeasurementValues as matches the input arguments
    mvs = mvs[: len(args)]
    par = parity(*mvs)

    assert isinstance(par, MeasurementValue)
    # measurements stored are all the input MeasurementValue's measurements
    assert par.measurements == [mv.measurements[0] for mv in mvs]
    # processing function on the new MeasurementValue applies the parity check
    assert par.processing_fn(*args) == expected_outcome


@pytest.mark.parametrize(
    "args, expected_outcome",
    [((0, 1, 1), 1), ((0, 0, 1), 0), ((1, 1, 1), 0)],
)
def test_parity_mvs_and_constant(args, expected_outcome):
    """Test that passing a sequence of MeasurementValues and a constant to the parity
    function returns the expected MeasurementValue with the correct processing function."""
    m1 = MeasurementValue([MidMeasureMP(0, id=3)], lambda x: x)
    m2 = MeasurementValue([MidMeasureMP(1, id=4)], lambda x: x)
    m3 = MeasurementValue([MidMeasureMP(2, id=5)], lambda x: x)

    par = parity(m1, m2, m3, 1)
    assert isinstance(par, MeasurementValue)
    assert par.measurements == [mv.measurements[0] for mv in (m1, m2, m3)]
    # processing function includes adding 1 in the parity check
    assert par.processing_fn(*args) == expected_outcome


def _is_valid(num_q, acquire_q):
    "Utility function to select for valid/invalid parameters in tests"
    if num_q < acquire_q:
        return False
    return True


class TestQubitMgr:
    """Tests for the dynamic qubit wire manager instance"""

    def test_default_init(self):
        "Default initialized manager holds 0 qubit wire indices"
        mgr = QubitMgr()
        assert mgr.num_qubits == 0
        assert len(mgr.active) == 0
        assert len(mgr.inactive) == 0
        assert len(mgr.all_qubits) == 0

    @pytest.mark.parametrize("num_qubits, offset", list(it.product(num_qubits_vals, offsets_vals)))
    def test_explicit_init(self, num_qubits, offset):
        "Test for valid initialization of QubitMgr"

        mgr = QubitMgr(num_qubits, offset)
        assert mgr.num_qubits == num_qubits
        assert len(mgr.active) == 0
        assert len(mgr.inactive) == num_qubits
        assert len(mgr.all_qubits) == num_qubits
        assert mgr.inactive == set(range(offset, offset + num_qubits, 1))
        assert mgr.inactive == mgr.all_qubits
        assert (
            str(mgr)
            == f"QubitMgr(num_qubits={num_qubits}, active={mgr.active}, inactive={mgr.inactive})"
        )

    @pytest.mark.parametrize(
        "num_qubits, offset",
        [
            ("a", 0),
            (0, "a"),
            (1, range(3)),
            (range(5), 0),
        ],
    )
    def test_invalid_init(self, num_qubits, offset):
        "Test for invalid initialization of QubitMgr"

        with pytest.raises(
            TypeError, match="Index counts and starting values must be positive integers"
        ):
            _ = QubitMgr(num_qubits, offset)

    @pytest.mark.parametrize("num_qubits, offset", list(it.product(num_qubits_vals, offsets_vals)))
    def test_acquire_qubit_valid(self, num_qubits, offset):
        "Test that we can acquire a single qubits and make it active"

        mgr = QubitMgr(num_qubits, offset)

        assert len(mgr.inactive) == num_qubits
        assert len(mgr.active) == 0
        assert len(mgr.active.intersection(mgr.inactive)) == 0

        idx_list = mgr.acquire_qubit()

        assert idx_list == offset
        assert {idx_list} == mgr.active
        assert (
            str(mgr)
            == f"QubitMgr(num_qubits={num_qubits}, active={mgr.active}, inactive={mgr.inactive})"
        )

    @pytest.mark.parametrize(
        "num_qubits, acquire_num, offset",
        list(it.product(num_qubits_vals, acquire_num_vals, offsets_vals)),
    )
    def test_acquire_qubits(self, num_qubits, acquire_num, offset):
        "Test that we can acquire multiple qubits and make them active"

        if not _is_valid(num_qubits, acquire_num):
            pytest.skip()

        mgr = QubitMgr(num_qubits, offset)
        idx_list = mgr.acquire_qubits(acquire_num)

        assert set(idx_list) == mgr.active
        assert len(mgr.active.intersection(mgr.inactive)) == 0

    @pytest.mark.parametrize(
        "num_qubits, acquire_num, offset",
        list(it.product(range(5), acquire_num_vals, offsets_vals)),
    )
    def test_acquire_qubits_invalid(self, num_qubits, acquire_num, offset):
        "Test that we have checks for invalid qubit acquisition requests"
        if _is_valid(num_qubits, acquire_num):
            pytest.skip()

        mgr = QubitMgr(num_qubits, offset)
        if num_qubits > 0:  # Pre-burn qubits for valid acquisition
            mgr.acquire_qubits(num_qubits)
        with pytest.raises(RuntimeError, match="Cannot allocate any additional wire indices"):
            mgr.acquire_qubit()

        with pytest.raises(RuntimeError, match="Cannot allocate any additional wire indices"):
            mgr.acquire_qubits(acquire_num)

    @pytest.mark.parametrize(
        "num_qubits, acquire_num, offset",
        list(it.product(num_qubits_vals, acquire_num_vals, offsets_vals)),
    )
    def test_release_qubit(self, num_qubits, acquire_num, offset):
        "Test that we can acquire and release qubits individually"

        if not _is_valid(num_qubits, acquire_num):
            pytest.skip()
        mgr = QubitMgr(num_qubits, offset)
        idx_list = mgr.acquire_qubits(acquire_num)

        for e in idx_list:
            assert e in mgr.active
            assert e not in mgr.inactive
            mgr.release_qubit(e)
            assert e not in mgr.active
            assert e in mgr.inactive

    @pytest.mark.parametrize(
        "num_qubits, acquire_num, offset",
        list(it.product(num_qubits_vals, acquire_num_vals, offsets_vals)),
    )
    def test_release_qubits(self, num_qubits, acquire_num, offset):
        "Test that we can acquire and release a fixed number of qubits"

        if not _is_valid(num_qubits, acquire_num):
            pytest.skip()
        mgr = QubitMgr(num_qubits, offset)
        idx_list = mgr.acquire_qubits(acquire_num)

        assert set(idx_list) == mgr.active
        assert set(idx_list).intersection(mgr.inactive) == set()
        mgr.release_qubits(idx_list)
        if acquire_num > 0:
            assert mgr.inactive.intersection(idx_list) == set(idx_list)
            assert set(idx_list).intersection(mgr.active) == set()

    @pytest.mark.parametrize(
        "num_qubits, acquire_num, offset",
        list(it.product(num_qubits_vals, acquire_num_vals, offsets_vals)),
    )
    def test_release_qubits_invalid(self, num_qubits, acquire_num, offset):
        "Test that we have checks for invalid qubit release requests"
        if _is_valid(num_qubits, acquire_num):
            pytest.skip()

        mgr = QubitMgr(num_qubits, offset)
        with pytest.raises(RuntimeError, match="not found in active set"):
            mgr.release_qubit(101)

        with pytest.raises(RuntimeError, match="not found in active set"):
            mgr.release_qubits([99, 88, 77])

    @pytest.mark.parametrize("num_qubits, offset", list(it.product(num_qubits_vals, offsets_vals)))
    def test_reserve_qubit(self, num_qubits, offset):
        "Test that we can selectively reserve and make active a user-specified qubit wire index"
        mgr = QubitMgr(num_qubits, offset)

        # Ensure randomly ordered indices for reservation
        q_inact = list(mgr.inactive)
        random.shuffle(q_inact)

        for q_idx in q_inact:
            assert q_idx not in mgr.active
            assert q_idx in mgr.inactive
            mgr.reserve_qubit(q_idx)
            assert q_idx in mgr.active
            assert q_idx not in mgr.inactive

    @pytest.mark.parametrize("num_qubits, offset", list(it.product(num_qubits_vals, offsets_vals)))
    def test_reserve_qubit_invalid(self, num_qubits, offset):
        "Test that we can selectively reserve and make active a user-specified qubit wire index"
        mgr = QubitMgr(num_qubits, offset)

        # Ensure randomly ordered indices for reservation
        q_invalid = list(range(99, 100, 1))

        for q_idx in q_invalid:
            with pytest.raises(RuntimeError, match="not found in inactive set"):
                mgr.reserve_qubit(q_idx)
