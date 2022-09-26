# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unittests for QuantumScript"""
import pytest

import pennylane as qml
from pennylane.tape import QuantumScript


class TestInitialization:
    """Test the non-update components of intialization."""

    def test_name(self):
        """Test the name property."""
        name = "hello"
        qs = QuantumScript(name=name)
        assert qs.name == name

    def test_no_update_empty_initialization(self):
        """Test initialization if nothing is provided and update does not occur."""

        qs = QuantumScript(_update=False)
        assert qs.name is None
        assert qs._ops == []
        assert qs._prep == []
        assert qs._measurements == []
        assert qs._par_info == {}
        assert qs._trainable_params == []
        assert qs._graph is None
        assert qs._specs is None
        assert qs._batch_size is None
        assert qs._qfunc_output is None
        assert qs.wires == qml.wires.Wires([])
        assert qs.num_wires == 0
        assert qs.is_sampled is False
        assert qs.all_sampled is False
        assert qs._obs_sharing_wires == []
        assert qs._obs_sharing_wires_id == []

    @pytest.mark.parametrize(
        "ops",
        (
            [qml.S(0)],
            (qml.S(0),),
            (qml.S(i) for i in [0]),
        ),
    )
    def test_provide_ops(self, ops):
        """Test provided ops are coverted to lists."""
        qs = QuantumScript(ops)
        assert len(qs._ops) == 1
        assert isinstance(qs._ops, list)
        assert qml.equal(qs._ops[0], qml.S(0))

    @pytest.mark.parametrize(
        "m",
        (
            [qml.state()],
            (qml.state(),),
            (qml.state() for _ in range(1)),
        ),
    )
    def test_provide_measurements(self, m):
        """Test provided measurements are converted to lists."""
        qs = QuantumScript(measurements=m)
        assert len(qs._measurements) == 1
        assert isinstance(qs._measurements, list)
        assert qs._measurements[0].return_type is qml.measurements.State

    @pytest.mark.parametrize(
        "prep",
        (
            [qml.BasisState([1, 1], wires=(0, 1))],
            (qml.BasisState([1, 1], wires=(0, 1)),),
            (qml.BasisState([1, 1], wires=(0, 1)) for _ in range(1)),
        ),
    )
    def test_provided_state_prep(self, prep):
        """Test state prep are converted to lists"""
        qs = QuantumScript(prep=prep)
        assert len(qs._prep) == 1
        assert isinstance(qs._prep, list)
        assert qml.equal(qs._prep[0], qml.BasisState([1, 1], wires=(0, 1)))


sample_measurements = [
    qml.sample(),
    qml.counts(),
    qml.counts(all_outcomes=True),
    qml.classical_shadow(wires=(0, 1)),
    qml.shadow_expval(qml.PauliX(0)),
]


class TestUpdate:
    def test_update_circuit_info_wires(self):

        prep = [qml.BasisState([1, 1], wires=(-1, -2))]
        ops = [qml.S(0), qml.T("a"), qml.S(0)]
        measurement = [qml.probs(wires=("a"))]

        qs = QuantumScript(ops, measurement, prep)
        assert qs.wires == qml.wires.Wires([-1, -2, 0, "a"])
        assert qs.num_wires == 4

    @pytest.mark.parametrize("sample_ms", sample_measurements)
    def test_update_circuit_info_sampling(self, sample_ms):
        qs = QuantumScript(measurements=[qml.state(), sample_ms])
        assert qs.is_sampled is True
        assert qs.all_sampled is False

        qs = QuantumScript(measurements=[sample_ms, sample_ms, qml.sample()])
        assert qs.is_sampled is True
        assert qs.all_sampled is True

    def test_update_circuit_info_no_sampling(self):
        qs = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        assert qs.is_sampled is False
        assert qs.all_sampled is False
