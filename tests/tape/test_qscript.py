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
"""Unit tests for the QuantumScript"""
from collections import defaultdict
import copy
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import MutualInfo, State, VnEntropy, Shots
from pennylane.tape import QuantumScript

# pylint: disable=protected-access, unused-argument, too-few-public-methods


class TestInitialization:
    """Test the non-update components of intialization."""

    def test_no_update_empty_initialization(self):
        """Test initialization if nothing is provided and update does not occur."""

        qs = QuantumScript(_update=False)
        assert len(qs._ops) == 0
        assert len(qs._measurements) == 0
        assert len(qs._par_info) == 0
        assert len(qs._trainable_params) == 0
        assert qs._graph is None
        assert qs._specs is None
        assert qs._shots.total_shots is None
        assert qs._batch_size is None
        assert qs.wires == qml.wires.Wires([])
        assert qs.num_wires == 0
        assert qs.is_sampled is False
        assert qs.all_sampled is False
        assert qs.samples_computational_basis is False
        assert len(qs._obs_sharing_wires) == 0
        assert len(qs._obs_sharing_wires_id) == 0

    @pytest.mark.parametrize(
        "ops",
        (
            [qml.S(0)],
            (qml.S(0),),
            (qml.S(i) for i in [0]),
        ),
    )
    def test_provide_ops(self, ops):
        """Test provided ops are converted to lists."""
        qs = QuantumScript(ops)
        assert len(qs.operations) == 1
        assert isinstance(qs.operations, list)
        assert qml.equal(qs.operations[0], qml.S(0))

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
        "ops,num_preps",
        [
            ((qml.BasisState([1], 0),), 1),
            ((qml.BasisState([1], 0), qml.PauliX(0)), 1),
            ((qml.BasisState([1], 0), qml.PauliX(0), qml.BasisState([1], 1)), 1),
            ((qml.BasisState([1], 0), qml.BasisState([1], 1), qml.PauliX(0)), 2),
            ((qml.PauliX(0),), 0),
            ((qml.PauliX(0), qml.BasisState([1], 0)), 0),
        ],
    )
    def test_num_preps(self, ops, num_preps):
        """Test the num_preps property."""
        assert QuantumScript(ops).num_preps == num_preps


sample_measurements = [
    qml.sample(),
    qml.counts(),
    qml.counts(all_outcomes=True),
    qml.classical_shadow(wires=(0, 1)),
    qml.shadow_expval(qml.PauliX(0)),
]


class TestUpdate:
    """Test the methods called by _update."""

    def test_cached_graph_specs_reset(self):
        """Test that update resets the graph and specs"""
        qs = QuantumScript()
        qs._graph = "hello"
        qs._specs = "something"

        qs._update()
        assert qs._graph is None
        assert qs._specs is None

    # pylint: disable=superfluous-parens
    def test_update_circuit_info_wires(self):
        """Test that on construction wires and num_wires are set."""
        prep = [qml.BasisState([1, 1], wires=(-1, -2))]
        ops = [qml.S(0), qml.T("a"), qml.S(0)]
        measurement = [qml.probs(wires=("a"))]

        qs = QuantumScript(prep + ops, measurement)
        assert qs.wires == qml.wires.Wires([-1, -2, 0, "a"])
        assert qs.num_wires == 4

    @pytest.mark.parametrize("sample_ms", sample_measurements)
    def test_update_circuit_info_sampling(self, sample_ms):
        qs = QuantumScript(measurements=[qml.expval(qml.PauliZ(0)), sample_ms])
        assert qs.is_sampled is True
        assert qs.all_sampled is False

        shadow_mp = sample_ms.return_type not in (
            qml.measurements.Shadow,
            qml.measurements.ShadowExpval,
        )
        assert qs.samples_computational_basis is shadow_mp

        qs = QuantumScript(measurements=[sample_ms, sample_ms, qml.sample()])
        assert qs.is_sampled is True
        assert qs.all_sampled is True
        assert qs.samples_computational_basis is True

    def test_update_circuit_info_no_sampling(self):
        """Test that all_sampled, is_sampled and samples_computational_basis properties are set to False if no sampling
        measurement process exists."""
        qs = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        assert qs.is_sampled is False
        assert qs.all_sampled is False
        assert qs.samples_computational_basis is False

    def test_samples_computational_basis_correctly(self):
        """Test that the samples_computational_basis property works as expected even if the script is updated."""
        qs = QuantumScript(measurements=[qml.sample()])
        assert qs.samples_computational_basis is True

        qs._measurements = [qml.expval(qml.PauliZ(0))]
        assert qs.samples_computational_basis is False

    def test_update_par_info_update_trainable_params(self):
        """Tests setting the parameter info dictionary.  Makes sure to include operations with
        multiple parameters, operations with matrix parameters, and measurement of observables with
        parameters."""
        ops = [
            qml.RX(1.2, wires=0),
            qml.Rot(2.3, 3.4, 5.6, wires=0),
            qml.QubitUnitary(np.eye(2), wires=0),
            qml.U2(-1, -2, wires=0),
        ]
        m = [qml.expval(qml.Hermitian(2 * np.eye(2), wires=0))]
        qs = QuantumScript(ops, m)

        p_i = qs._par_info

        assert p_i[0] == {"op": ops[0], "op_idx": 0, "p_idx": 0}
        assert p_i[1] == {"op": ops[1], "op_idx": 1, "p_idx": 0}
        assert p_i[2] == {"op": ops[1], "op_idx": 1, "p_idx": 1}
        assert p_i[3] == {"op": ops[1], "op_idx": 1, "p_idx": 2}
        assert p_i[4] == {"op": ops[2], "op_idx": 2, "p_idx": 0}
        assert p_i[5] == {"op": ops[3], "op_idx": 3, "p_idx": 0}
        assert p_i[6] == {"op": ops[3], "op_idx": 3, "p_idx": 1}
        assert p_i[7] == {"op": m[0].obs, "op_idx": 4, "p_idx": 0}

        assert qs._trainable_params == list(range(8))

    # pylint: disable=unbalanced-tuple-unpacking
    def test_get_operation(self):
        """Tests the tape method get_operation."""
        ops = [
            qml.RX(1.2, wires=0),
            qml.Rot(2.3, 3.4, 5.6, wires=0),
            qml.PauliX(wires=0),
            qml.QubitUnitary(np.eye(2), wires=0),
            qml.U2(-1, -2, wires=0),
        ]
        m = [qml.expval(qml.Hermitian(2 * np.eye(2), wires=0))]
        qs = QuantumScript(ops, m)

        op_0, op_id_0, p_id_0 = qs.get_operation(0)
        assert qml.equal(op_0, ops[0]) and op_id_0 == 0 and p_id_0 == 0

        op_1, op_id_1, p_id_1 = qs.get_operation(1)
        assert qml.equal(op_1, ops[1]) and op_id_1 == 1 and p_id_1 == 0

        op_2, op_id_2, p_id_2 = qs.get_operation(2)
        assert qml.equal(op_2, ops[1]) and op_id_2 == 1 and p_id_2 == 1

        op_3, op_id_3, p_id_3 = qs.get_operation(3)
        assert qml.equal(op_3, ops[1]) and op_id_3 == 1 and p_id_3 == 2

        op_4, op_id_4, p_id_4 = qs.get_operation(4)
        assert qml.equal(op_4, ops[3]) and op_id_4 == 3 and p_id_4 == 0

        op_5, op_id_5, p_id_5 = qs.get_operation(5)
        assert qml.equal(op_5, ops[4]) and op_id_5 == 4 and p_id_5 == 0

        op_6, op_id_6, p_id_6 = qs.get_operation(6)
        assert qml.equal(op_6, ops[4]) and op_id_6 == 4 and p_id_6 == 1

        _, obs_id_0, p_id_0 = qs.get_operation(7)
        assert obs_id_0 == 5 and p_id_0 == 0

    def test_update_observables(self):
        """This method needs to be more thoroughly tested, and probably even reconsidered in
        its design. I can't find any unittests in `test_tape.py`."""
        obs = [
            qml.PauliX("a"),
            qml.PauliX(0),
            qml.PauliY(0),
            qml.PauliX("b"),
            qml.PauliX(0) @ qml.PauliY(1),
        ]
        qs = QuantumScript(measurements=[qml.expval(o) for o in obs])
        assert qs._obs_sharing_wires == [obs[1], obs[2], obs[4]]
        assert qs._obs_sharing_wires_id == [1, 2, 4]

    @pytest.mark.parametrize(
        "x, rot, exp_batch_size",
        [
            (0.2, [0.1, -0.9, 2.1], None),
            ([0.2], [0.1, -0.9, 2.1], 1),
            ([0.2], [[0.1], [-0.9], 2.1], 1),
            ([0.2] * 3, [0.1, [-0.9] * 3, 2.1], 3),
        ],
    )
    def test_update_batch_size(self, x, rot, exp_batch_size):
        """Test that the batch size is correctly inferred from all operation's
        batch_size when creating a QuantumScript."""

        obs = [qml.RX(x, wires=0), qml.Rot(*rot, wires=1)]
        m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))]
        qs = QuantumScript(obs, m)
        assert qs.batch_size == exp_batch_size

    @pytest.mark.parametrize(
        "x, rot, y",
        [
            (0.2, [[0.1], -0.9, 2.1], [0.1, 0.9]),
            ([0.2], [0.1, [-0.9] * 2, 2.1], 0.1),
        ],
    )
    def test_error_inconsistent_batch_sizes(self, x, rot, y):
        """Tests that an error is raised if inconsistent batch sizes exist."""
        ops = [qml.RX(x, wires=0), qml.Rot(*rot, 1), qml.RX(y, wires=1)]

        with pytest.raises(
            ValueError, match="batch sizes of the quantum script operations do not match."
        ):
            _ = QuantumScript(ops)

    @pytest.mark.parametrize(
        "m, output_dim",
        [
            ([qml.expval(qml.PauliX(0))], 1),
            ([qml.expval(qml.PauliX(0)), qml.var(qml.PauliY(1))], 2),
            ([qml.probs(wires=(0, 1))], 4),
            ([qml.state()], 0),
            ([qml.probs((0, 1)), qml.expval(qml.PauliX(0))], 5),
        ],
    )
    @pytest.mark.parametrize("ops, factor", [([], 1), ([qml.RX([1.2, 2.3, 3.4], wires=0)], 3)])
    def test_update_output_dim(self, m, output_dim, ops, factor):
        """Test setting the output_dim property."""
        qs = QuantumScript(ops, m)
        assert qs.output_dim == output_dim * factor


class TestIteration:
    """Test the capabilities related to iterating over quantum script."""

    @pytest.fixture
    def make_qs(self):
        ops = [
            qml.RX(0.432, wires=0),
            qml.Rot(0.543, 0, 0.23, wires=0),
            qml.CNOT(wires=[0, "a"]),
            qml.RX(0.133, wires=4),
        ]
        meas = [qml.expval(qml.PauliX(wires="a")), qml.probs(wires=[0, "a"])]

        qs = QuantumScript(ops, meas)

        return qs, ops, meas

    def test_qscript_is_iterable(self, make_qs):
        """Test the iterable protocol: that we can iterate over a tape because
        an iterator object can be obtained using the iter function."""
        qs, ops, meas = make_qs
        expected = ops + meas

        qs_iterator = iter(qs)

        iterating = True

        counter = 0

        while iterating:
            try:
                next_qs_elem = next(qs_iterator)

                assert next_qs_elem is expected[counter]
                counter += 1

            except StopIteration:
                # StopIteration is raised by next when there are no more
                # elements to iterate over
                iterating = False

        assert counter == len(expected)

    def test_qscript_is_sequence(self, make_qs):
        """Test the sequence protocol: that a quantum script is a sequence because its
        __len__ and __getitem__ methods work as expected."""
        tape, ops, meas = make_qs

        expected = ops + meas

        for idx, exp_elem in enumerate(expected):
            assert tape[idx] is exp_elem

        assert len(tape) == len(expected)

    def test_qscript_as_list(self, make_qs):
        """Test that a quantums script can be converted to a list."""
        qs, ops, meas = make_qs
        qs_list = list(qs)

        expected = ops + meas
        for op, exp_op in zip(qs_list, expected):
            assert op is exp_op

        assert len(qs_list) == len(expected)

    def test_iteration_preserves_circuit(self):
        """Test that iterating through a quantum scriptdoesn't change the underlying
        list of operations and measurements in the circuit."""

        ops = [
            qml.RX(0.432, wires=0),
            qml.Rot(0.543, 0, 0.23, wires=0),
            qml.CNOT(wires=[0, "a"]),
            qml.RX(0.133, wires=4),
        ]
        m = [
            qml.expval(qml.PauliX(wires="a")),
            qml.probs(wires=[0, "a"]),
        ]
        qs = QuantumScript(ops, m)

        circuit = ops + m

        # Check that the underlying circuit is as expected
        assert qs.circuit == circuit
        assert list(qs) == circuit
        # Iterate over the tape
        for op, expected in zip(qs, circuit):
            assert op is expected

        # Check that the underlying circuit is still as expected
        assert qs.circuit == circuit


class TestInfomationProperties:
    """Tests the graph and specs properties."""

    @pytest.fixture
    def make_script(self):
        ops = [
            qml.RX(-0.543, wires=0),
            qml.Rot(-4.3, 4.69, 1.2, wires=0),
            qml.CNOT(wires=[0, "a"]),
            qml.RX(0.54, wires=4),
        ]
        m = [qml.expval(qml.PauliX(wires="a")), qml.probs(wires=[0, "a"])]

        return QuantumScript(ops, m)

    def test_graph(self, make_script):
        """Tests the graph is constructed the first time it's requested and then cached."""
        qs = make_script

        assert qs._graph is None

        g = qs.graph
        assert isinstance(g, qml.CircuitGraph)
        assert g.operations == qs.operations
        assert g.observables == qs.observables

        # test that if we request it again, we get the same object
        assert qs.graph is g

    def test_empty_qs_specs(self):
        """Tests the specs of an script."""
        qs = QuantumScript()
        assert qs._specs is None

        assert qs.specs["resources"] == qml.resource.Resources()

        assert qs.specs["num_observables"] == 0
        assert qs.specs["num_diagonalizing_gates"] == 0
        assert qs.specs["num_trainable_params"] == 0

        assert len(qs.specs) == 4

        assert qs._specs is qs.specs

    def test_specs_tape(self, make_script):
        """Tests that regular scripts return correct specifications"""
        qs = make_script

        assert qs._specs is None
        specs = qs.specs
        assert qs._specs is specs

        assert len(specs) == 4

        gate_types = defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 1})
        gate_sizes = defaultdict(int, {1: 3, 2: 1})
        expected_resources = qml.resource.Resources(
            num_wires=3, num_gates=4, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
        )
        assert specs["resources"] == expected_resources
        assert specs["num_observables"] == 2
        assert specs["num_diagonalizing_gates"] == 1
        assert specs["num_trainable_params"] == 5

    @pytest.mark.parametrize(
        "shots, total_shots, shot_vector",
        [
            (None, None, ()),
            (1, 1, ((1, 1),)),
            (10, 10, ((10, 1),)),
            ([1, 1, 2, 3, 1], 8, ((1, 2), (2, 1), (3, 1), (1, 1))),
            (Shots([1, 1, 2]), 4, ((1, 2), (2, 1))),
        ],
    )
    def test_set_shots(self, shots, total_shots, shot_vector):
        qs = QuantumScript([], [], shots=shots)
        assert isinstance(qs.shots, Shots)
        assert qs.shots.total_shots == total_shots
        assert qs.shots.shot_vector == shot_vector


class TestScriptCopying:
    """Test for quantum script copying behaviour"""

    def test_shallow_copy(self):
        """Test that shallow copying of a script results in all
        contained data being shared between the original tape and the copy"""
        prep = [qml.BasisState(np.array([1, 0]), wires=(0, 1))]
        ops = [qml.RY(0.5, wires=1), qml.CNOT((0, 1))]
        m = [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))]
        qs = QuantumScript(prep + ops, m)

        copied_qs = qs.copy()

        assert copied_qs is not qs

        # the operations are simply references
        assert all(o1 is o2 for o1, o2 in zip(copied_qs.operations, qs.operations))
        assert all(o1 is o2 for o1, o2 in zip(copied_qs.observables, qs.observables))
        assert all(m1 is m2 for m1, m2 in zip(copied_qs.measurements, qs.measurements))

        # operation data is also a reference
        assert copied_qs.operations[0].wires is qs.operations[0].wires
        assert copied_qs.operations[0].data[0] is qs.operations[0].data[0]

        # check that all tape metadata is identical
        assert qs.get_parameters() == copied_qs.get_parameters()
        assert qs.wires == copied_qs.wires
        assert qs.data == copied_qs.data
        assert qs.shots is copied_qs.shots

        # check that the output dim is identical
        assert qs.output_dim == copied_qs.output_dim

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "copy_fn", [lambda tape: tape.copy(copy_operations=True), lambda tape: copy.copy(tape)]
    )
    def test_shallow_copy_with_operations(self, copy_fn):
        """Test that shallow copying of a tape and operations allows
        parameters to be set independently"""

        prep = [qml.BasisState(np.array([1, 0]), wires=(0, 1))]
        ops = [qml.RY(0.5, wires=1), qml.CNOT((0, 1))]
        m = [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))]
        qs = QuantumScript(prep + ops, m)

        copied_qs = copy_fn(qs)

        assert copied_qs is not qs

        # the operations are not references; they are unique objects
        assert all(o1 is not o2 for o1, o2 in zip(copied_qs.operations, qs.operations))
        assert all(o1 is not o2 for o1, o2 in zip(copied_qs.observables, qs.observables))
        assert all(m1 is not m2 for m1, m2 in zip(copied_qs.measurements, qs.measurements))

        # however, the underlying operation data *is still shared*
        assert copied_qs.operations[0].wires is qs.operations[0].wires
        # the data list is copied, but the elements of the list remain th same
        assert copied_qs.operations[0].data[0] is qs.operations[0].data[0]

        assert qs.get_parameters() == copied_qs.get_parameters()
        assert qs.wires == copied_qs.wires
        assert qs.data == copied_qs.data
        assert qs.shots is copied_qs.shots

        # check that the output dim is identical
        assert qs.output_dim == copied_qs.output_dim

    def test_deep_copy(self):
        """Test that deep copying a tape works, and copies all constituent data except parameters"""
        prep = [qml.BasisState(np.array([1, 0]), wires=(0, 1))]
        ops = [qml.RY(0.5, wires=1), qml.CNOT((0, 1))]
        m = [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))]
        qs = QuantumScript(prep + ops, m)

        copied_qs = copy.deepcopy(qs)

        assert copied_qs is not qs

        # the operations are not references
        assert all(o1 is not o2 for o1, o2 in zip(copied_qs.operations, qs.operations))
        assert all(o1 is not o2 for o1, o2 in zip(copied_qs.observables, qs.observables))
        assert all(m1 is not m2 for m1, m2 in zip(copied_qs.measurements, qs.measurements))
        assert copied_qs.shots is qs.shots

        # check that the output dim is identical
        assert qs.output_dim == copied_qs.output_dim

        # The underlying operation data has also been copied
        assert copied_qs.operations[0].wires is not qs.operations[0].wires

        # however, the underlying operation *parameters* are still shared
        # to support PyTorch, which does not support deep copying of tensors
        assert copied_qs.operations[0].data[0] is qs.operations[0].data[0]


def test_adjoint():
    """Tests taking the adjoint of a quantum script."""
    ops = [
        qml.BasisState([1, 1], wires=0),
        qml.RX(1.2, wires=0),
        qml.S(0),
        qml.CNOT((0, 1)),
        qml.T(1),
    ]
    m = [qml.expval(qml.PauliZ(0))]
    qs = QuantumScript(ops, m)

    with qml.queuing.AnnotatedQueue() as q:
        adj_qs = qs.adjoint()

    assert len(q.queue) == 0  # not queued

    assert qml.equal(adj_qs.operations[0], qs.operations[0])
    assert adj_qs.measurements == qs.measurements
    assert adj_qs.shots is qs.shots

    # assumes lazy=False
    expected_ops = [qml.adjoint(qml.T(1)), qml.CNOT((0, 1)), qml.adjoint(qml.S(0)), qml.RX(-1.2, 0)]
    for op, expected in zip(adj_qs.operations[1:], expected_ops):
        # update this one qml.equal works with adjoint
        assert isinstance(op, type(expected))
        assert op.wires == expected.wires
        assert op.data == expected.data


class TestHashing:
    """Test for script hashing"""

    @pytest.mark.parametrize(
        "m",
        [
            qml.expval(qml.PauliZ(0)),
            qml.state(),
            qml.probs(wires=0),
            qml.density_matrix(wires=0),
            qml.var(qml.PauliY(0)),
        ],
    )
    def test_identical(self, m):
        """Tests that the circuit hash of identical circuits are identical"""
        ops = [qml.RX(0.3, 0), qml.RY(0.2, 1), qml.CNOT((0, 1))]
        qs1 = QuantumScript(ops, [m])
        qs2 = QuantumScript(ops, [m])

        assert qs1.hash == qs2.hash

    def test_identical_numeric(self):
        """Tests that the circuit hash of identical circuits are identical
        even though the datatype of the arguments may differ"""
        a = 0.3
        b = 0.2

        qs1 = QuantumScript([qml.RX(a, 0), qml.RY(b, 1)])
        qs2 = QuantumScript([qml.RX(np.array(a), 0), qml.RY(np.array(b), 1)])

        assert qs1.hash == qs2.hash

    def test_different_wires(self):
        """Tests that the circuit hash of circuits with the same operations
        on different wires have different hashes"""
        a = 0.3

        qs1 = QuantumScript([qml.RX(a, 0)])
        qs2 = QuantumScript([qml.RX(a, 1)])

        assert qs1.hash != qs2.hash

    def test_different_trainabilities(self):
        """Tests that the circuit hash of identical circuits differ
        if the circuits have different trainable parameters"""
        qs1 = QuantumScript([qml.RX(1.0, 0), qml.RY(1.0, 1)])
        qs2 = copy.copy(qs1)

        qs1.trainable_params = [0]
        qs2.trainable_params = [0, 1]
        assert qs1.hash != qs2.hash

    def test_different_parameters(self):
        """Tests that the circuit hash of circuits with different
        parameters differs"""
        qs1 = QuantumScript([qml.RX(1.0, 0)])
        qs2 = QuantumScript([qml.RX(2.0, 0)])

        assert qs1.hash != qs2.hash

    def test_different_operations(self):
        """Tests that the circuit hash of circuits with different
        operations differs"""
        qs1 = QuantumScript([qml.S(0)])
        qs2 = QuantumScript([qml.T(0)])
        assert qs1.hash != qs2.hash

    def test_different_measurements(self):
        """Tests that the circuit hash of circuits with different
        measurements differs"""
        qs1 = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        qs2 = QuantumScript(measurements=[qml.var(qml.PauliZ(0))])
        assert qs1.hash != qs2.hash

    def test_different_observables(self):
        """Tests that the circuit hash of circuits with different
        observables differs"""
        A = np.diag([1.0, 2.0])
        qs1 = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        qs2 = QuantumScript(measurements=[qml.expval(qml.Hermitian(A, wires=0))])

        assert qs1.hash != qs2.hash

    def test_rotation_modulo_identical(self):
        """Tests that the circuit hash of circuits with single-qubit
        rotations differing by multiples of 2pi have identical hash"""
        a = np.array(np.pi / 2, dtype=np.float64)
        b = np.array(np.pi / 4, dtype=np.float64)

        qs1 = QuantumScript([qml.RX(a, 0), qml.RY(b, 1)])
        qs2 = QuantumScript([qml.RX(a - 2 * np.pi, 0), qml.RY(b + 2 * np.pi, 1)])

        assert qs1.hash == qs2.hash

    def test_controlled_rotation_modulo_identical(self):
        """Tests that the circuit hash of circuits with controlled
        rotations differing by multiples of 4pi have identical hash,
        but those differing by 2pi are different."""
        a = np.array(np.pi / 2, dtype=np.float64)
        b = np.array(np.pi / 2, dtype=np.float64)

        qs = QuantumScript([qml.CRX(a, (0, 1)), qml.CRY(b, (0, 1))])
        qs_add_2pi = QuantumScript([qml.CRX(a + 2 * np.pi, (0, 1)), qml.CRY(b + 2 * np.pi, (0, 1))])
        qs_add_4pi = QuantumScript([qml.CRX(a + 4 * np.pi, (0, 1)), qml.CRY(b + 4 * np.pi, (0, 1))])

        assert qs.hash == qs_add_4pi.hash
        assert qs.hash != qs_add_2pi.hash

    def test_hash_shots(self):
        """Test tha circuits with different shots have different hashes."""
        qs1 = QuantumScript([qml.S(0)], [qml.sample(wires=0)], shots=10)
        qs2 = QuantumScript([qml.T(0)], [qml.sample(wires=0)], shots=20)

        assert qs1.hash != qs2.hash


class TestQScriptDraw:
    """Test the script draw method."""

    def test_default_kwargs(self):
        """Test quantum script's draw with default keyword arguments."""

        qs = QuantumScript(
            [qml.RX(1.23456, wires=0), qml.RY(2.3456, wires="a"), qml.RZ(3.4567, wires=1.234)]
        )

        assert qs.draw() == qml.drawer.tape_text(qs)
        assert qs.draw(decimals=2) == qml.drawer.tape_text(qs, decimals=2)

    def test_show_matrices(self):
        """Test show_matrices keyword argument."""
        qs = QuantumScript([qml.QubitUnitary(qml.numpy.eye(2), wires=0)])

        assert qs.draw() == qml.drawer.tape_text(qs)
        assert qs.draw(show_matrices=True) == qml.drawer.tape_text(qs, show_matrices=True)

    def test_max_length_keyword(self):
        """Test the max_length keyword argument."""
        qs = QuantumScript([qml.PauliX(0)] * 50)

        assert qs.draw() == qml.drawer.tape_text(qs)
        assert qs.draw(max_length=20) == qml.drawer.tape_text(qs, max_length=20)


class TestMakeQscript:
    """Test the make_qscript method."""

    def test_ops_are_recorded_to_qscript(self):
        """Test make_qscript records operations from the quantum function passed to it."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.expval(qml.PauliX(0))

        qscript = qml.tape.make_qscript(qfunc)()
        assert len(qscript.operations) == 2
        assert len(qscript.measurements) == 1

    @pytest.mark.parametrize(
        "shots, total_shots, shot_vector",
        [
            (None, None, ()),
            (1, 1, ((1, 1),)),
            (10, 10, ((10, 1),)),
            ([1, 1, 2, 3, 1], 8, ((1, 2), (2, 1), (3, 1), (1, 1))),
            (Shots([1, 1, 2]), 4, ((1, 2), (2, 1))),
        ],
    )
    def test_make_qscript_with_shots(self, shots, total_shots, shot_vector):
        """Test that ``make_qscript`` creates a ``QuantumScript`` correctly when
        shots are specified."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.expval(qml.PauliX(0))

        qscript = qml.tape.make_qscript(qfunc, shots=shots)()

        assert len(qscript.operations) == 2
        assert len(qscript.measurements) == 1
        assert qscript.shots.total_shots == total_shots
        assert qscript.shots.shot_vector == shot_vector

    def test_qfunc_is_recording_during_make_qscript(self):
        """Test that quantum functions passed to make_qscript run in a recording context."""

        def assert_recording():
            assert qml.QueuingManager.recording()

        qml.tape.make_qscript(assert_recording)()

    def test_ops_are_not_recorded_to_surrounding_context(self):
        """Test that ops are not recorded to any surrounding context during make_qscript."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])

        with qml.queuing.AnnotatedQueue() as q:
            recorded_op = qml.PauliX(0)
            qscript = qml.tape.make_qscript(qfunc)()
        assert q.queue == [recorded_op]
        assert len(qscript.operations) == 2

    def test_make_qscript_returns_callable(self):
        """Test that make_qscript returns a callable."""

        def qfunc():
            qml.Hadamard(0)

        assert callable(qml.tape.make_qscript(qfunc))

    def test_non_queued_ops_are_not_recorded(self):
        """Test that ops are not recorded to the qscript when recording is disabled."""

        def qfunc():
            qml.PauliX(0)
            with qml.QueuingManager.stop_recording():
                qml.Hadamard(0)

        qscript = qml.tape.make_qscript(qfunc)()
        assert len(qscript.operations) == 1
        assert qscript.operations[0].name == "PauliX"


class TestFromQueue:
    """Test that QuantumScript.from_queue behaves properly."""

    def test_from_queue(self):
        """Test that QuantumScript.from_queue works correctly."""
        with qml.queuing.AnnotatedQueue() as q:
            op = qml.PauliX(0)
            with qml.QueuingManager.stop_recording():
                qml.Hadamard(0)
            qml.expval(qml.PauliZ(0))
        qscript = QuantumScript.from_queue(q)
        assert qscript.operations == [op]
        assert len(qscript.measurements) == 1

    def test_from_queue_child_class_preserved(self):
        """Test that a child of QuantumScript gets its own type when calling from_queue."""

        class MyScript(QuantumScript):
            pass

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)

        qscript = MyScript.from_queue(q)
        assert isinstance(qscript, MyScript)

    def test_from_queue_child_with_different_init_fails(self):
        """Test that if a child class overrides init to take different arguments, from_queue will fail."""

        class ScriptWithNewInit(QuantumScript):
            """An arbitrary class that has a different constructor from QuantumScript."""

            def __init__(self, ops, measurements, prep, bonus):
                super().__init__(ops, measurements, prep)
                self.bonus = bonus

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)

        with pytest.raises(TypeError):
            ScriptWithNewInit.from_queue(q)

    @pytest.mark.parametrize("child", QuantumScript.__subclasses__())
    def test_that_fails_if_a_subclass_does_not_match(self, child):
        """
        Makes sure that no subclasses for QuantumScript override the constructor.
        If you have, and you stumbled onto this test, note that QuantumScript.from_queue
        might need some modification for your PR to be complete.
        """
        with qml.queuing.AnnotatedQueue() as q:
            x = qml.PauliZ(0)

        assert child.from_queue(q).operations == [x]

    def test_diagonalizing_gates_not_queued(self):
        """Test that diagonalizing gates don't get added to an active queue when
        requested."""
        qscript = QuantumScript(ops=[qml.PauliZ(0)], measurements=[qml.expval(qml.PauliX(0))])

        with qml.queuing.AnnotatedQueue() as q:
            diag_ops = qscript.diagonalizing_gates

        assert len(diag_ops) == 1
        # Hadamard is the diagonalizing gate for PauliX
        assert qml.equal(diag_ops[0], qml.Hadamard(0))
        assert len(q.queue) == 0


measures = [
    (qml.expval(qml.PauliZ(0)), ()),
    (qml.var(qml.PauliZ(0)), ()),
    (qml.probs(wires=[0]), (2,)),
    (qml.probs(wires=[0, 1]), (4,)),
    (qml.state(), (8,)),  # Assumes 3-qubit device
    (qml.density_matrix(wires=[0, 1]), (4, 4)),
    (
        qml.sample(qml.PauliZ(0)),
        None,
    ),  # Shape is None because the expected shape is in the test case
    (qml.sample(), None),  # Shape is None because the expected shape is in the test case
    (qml.mutual_info(wires0=[0], wires1=[1]), ()),
    (qml.vn_entropy(wires=[0, 1]), ()),
]

multi_measurements = [
    ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))], ((), ())),
    ([qml.probs(wires=[0]), qml.probs(wires=[1])], ((2,), (2,))),
    ([qml.probs(wires=[0]), qml.probs(wires=[1, 2])], ((2,), (4,))),
    ([qml.probs(wires=[0, 2]), qml.probs(wires=[1])], ((4,), (2,))),
    (
        [qml.probs(wires=[0]), qml.probs(wires=[1, 2]), qml.probs(wires=[0, 1, 2])],
        ((2,), (4,), (8,)),
    ),
]


class TestMeasurementProcess:
    """Tests for the shape and numeric type of a measurement process"""

    measurements_no_shots = [
        (qml.expval(qml.PauliZ(0)), ()),
        (qml.var(qml.PauliZ(0)), ()),
        (qml.probs(wires=[0, 1]), (4,)),
        (qml.state(), (8,)),
        (qml.density_matrix(wires=[0, 1]), (4, 4)),
        (qml.mutual_info(wires0=[0], wires1=[1]), ()),
        (qml.vn_entropy(wires=[0, 1]), ()),
    ]

    measurements_finite_shots = [
        (qml.expval(qml.PauliZ(0)), ()),
        (qml.var(qml.PauliZ(0)), ()),
        (qml.probs(wires=[0, 1]), (4,)),
        (qml.state(), (8,)),
        (qml.density_matrix(wires=[0, 1]), (4, 4)),
        (qml.sample(qml.PauliZ(0)), (10,)),
        (qml.sample(), (10, 3)),
        (qml.mutual_info(wires0=0, wires1=1), ()),
        (qml.vn_entropy(wires=[0, 1]), ()),
    ]

    measurements_shot_vector = [
        (qml.expval(qml.PauliZ(0)), ((), (), ())),
        (qml.var(qml.PauliZ(0)), ((), (), ())),
        (qml.probs(wires=[0, 1]), ((4,), (4,), (4,))),
        (qml.state(), ((8,), (8,), (8,))),
        (qml.density_matrix(wires=[0, 1]), ((4, 4), (4, 4), (4, 4))),
        (qml.sample(qml.PauliZ(0)), ((10,), (20,), (30,))),
        (qml.sample(), ((10, 3), (20, 3), (30, 3))),
        (qml.mutual_info(wires0=0, wires1=1), ((), (), ())),
        (qml.vn_entropy(wires=[0, 1]), ((), (), ())),
    ]

    @pytest.mark.parametrize("measurement, expected_shape", measurements_no_shots)
    def test_output_shapes_no_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots=None"""
        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires, shots=None)

        assert measurement.shape(dev, Shots(None)) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measurements_finite_shots)
    def test_output_shapes_finite_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots is finite"""
        num_wires = 3
        num_shots = 10
        dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)

        assert measurement.shape(dev, Shots(num_shots)) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measurements_shot_vector)
    def test_output_shapes_shot_vector(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots is a vector"""
        num_wires = 3
        shot_vector = [10, 20, 30]
        dev = qml.device("default.qubit", wires=num_wires, shots=shot_vector)

        assert measurement.shape(dev, Shots(shot_vector)) == expected_shape

    def test_undefined_shape_error(self):
        """Test that an error is raised for a measurement with an undefined shape"""
        measurement = qml.counts(wires=[0, 1])
        dev = qml.device("default.qubit", wires=2)
        shots = Shots(None)
        msg = "The shape of the measurement CountsMP is not defined"

        with pytest.raises(qml.QuantumFunctionError, match=msg):
            measurement.shape(dev, shots)


class TestOutputShape:
    """Tests for determining the tape output shape of tapes."""

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_output_shapes_single(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, [measurement], shots=shots)

        shot_dim = len(shots) if isinstance(shots, tuple) else shots
        if expected_shape is None:
            expected_shape = shot_dim if shot_dim == 1 else (shot_dim,)

        if measurement.return_type is qml.measurements.Sample:
            if measurement.obs is None:
                expected_shape = (num_wires,) if shots == 1 else (shots, num_wires)

            else:
                expected_shape = () if shots == 1 else (shots,)
        assert qs.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10, (1, 2, 5, 3)])
    def test_output_shapes_single_qnode_check(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the output
        shape of a QNode for a single measurement."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")
        if shots and isinstance(measurement, qml.measurements.StateMeasurement):
            pytest.skip("State measurements with finite shots not supported.")

        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, [measurement], shots=shots)
        program, _ = dev.preprocess()
        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([qs], dev, gradient_fn=None, transform_program=program)[0]

        if isinstance(shots, tuple):
            res_shape = tuple(r.shape for r in res)
        else:
            res_shape = res.shape

        assert qs.shape(dev) == res_shape

    def test_output_shapes_single_qnode_check_cutoff(self):
        """Test that the tape output shape is correct when computing
        probabilities with a dummy device that defines a cutoff value."""

        class CustomDevice(qml.QubitDevice):
            """A dummy device that has a cutoff value specified and returns
            analytic probabilities in a fashion similar to the
            strawberryfields.fock device.

            Note: this device definition is used as PennyLane-SF is not a
            dependency of PennyLane core and there are no CV device in
            PennyLane core using a cutoff value.
            """

            name = "Device with cutoff"
            short_name = "dummy.device"
            pennylane_requires = "0.1.0"
            version = "0.0.1"
            author = "CV quantum"

            operations = {}
            observables = {"Identity"}

            def __init__(self, shots=None, wires=None, cutoff=None):
                super().__init__(wires=wires, shots=shots)
                self.cutoff = cutoff

            def apply(self, operations, **kwargs):
                pass

            def analytic_probability(self, wires=None):
                if wires is None:
                    wires = self.wires
                return np.zeros(self.cutoff ** len(wires))

        dev = CustomDevice(wires=2, cutoff=13)

        # If PennyLane-SF is installed, the following can be checked e.g., locally:
        # dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=13)

        qs = QuantumScript(measurements=[qml.probs(wires=[0])])

        res_shape = qml.execute([qs], dev, gradient_fn=None)[0]
        assert qs.shape(dev) == res_shape.shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_multi_measure(self, measurements, expected, shots):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        qs = QuantumScript(measurements=measurements, shots=shots)

        if measurements[0].return_type is qml.measurements.Sample:
            expected[1] = shots
            expected = tuple(expected)

        res = qs.shape(dev)
        assert res == expected

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([qs], dev, gradient_fn=None)[0]
        res_shape = tuple(r.shape for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    def test_multi_measure_shot_vector(self, measurements, expected):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements with a shot
        vector."""
        if measurements[0].return_type is qml.measurements.Probability:
            num_wires = set(len(m.wires) for m in measurements)
            if len(num_wires) > 1:
                pytest.skip(
                    "Multi-probs with varying number of varies when using a shot vector is to be updated in PennyLane."
                )

        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, measurements, shots=shots)

        # Update expected as we're using a shotvector
        expected = tuple(expected for _ in shots)
        res = qs.shape(dev)
        assert res == expected

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([qs], dev, gradient_fn=None)[0]
        res_shape = tuple(tuple(r_.shape for r_ in r) for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [1, 10])
    def test_multi_measure_sample(self, shots):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(
            ops, [qml.sample(qml.PauliZ(i)) for i in range(num_samples)], shots=shots
        )

        expected = tuple(() if shots == 1 else (shots,) for _ in range(num_samples))

        res = qs.shape(dev)
        assert res == expected

        res = qml.execute([qs], dev, gradient_fn=None)[0]
        res_shape = tuple(r.shape for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_broadcasting_single(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape for a single measurement and parameter broadcasting"""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        if (
            measurement.return_type in {State, MutualInfo, VnEntropy}
            and measurement.wires is not None
        ):
            pytest.skip("Density matrix does not support parameter broadcasting")

        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires, shots=shots)

        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        program, _ = dev.preprocess()
        expected_shape = qml.execute([tape], dev, gradient_fn=None, transform_program=program)[
            0
        ].shape

        assert tape.shape(dev) == expected_shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurement, expected", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_broadcasting_multi(self, measurement, expected, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape for multiple measurements and parameter broadcasting"""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        if measurement.return_type in {State, MutualInfo, VnEntropy}:
            pytest.skip("Density matrix does not support parameter broadcasting.")

        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for _ in range(2):
                qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        program, _ = dev.preprocess()
        expected = qml.execute([tape], dev, gradient_fn=None, transform_program=program)[0]
        actual = tape.shape(dev)

        for exp, act in zip(expected, actual):
            assert exp.shape == act

    @pytest.mark.autograd
    def test_multi_measure_sample_obs_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with an observable with a shot vector."""
        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(
            ops, [qml.sample(qml.PauliZ(i)) for i in range(num_samples)], shots=shots
        )

        expected = tuple(tuple(() if s == 1 else (s,) for _ in range(num_samples)) for s in shots)

        res = qs.shape(dev)
        assert res == expected

        expected = qml.execute([qs], dev, gradient_fn=None)[0]
        expected_shape = tuple(tuple(e_.shape for e_ in e) for e in expected)

        assert res == expected_shape

    @pytest.mark.autograd
    def test_multi_measure_sample_wires_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with wires with a shot vector."""
        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        num_samples = 3
        ops = [qml.RY(0.3, 0), qml.RX(0.2, 0)]
        qs = QuantumScript(ops, [qml.sample()] * num_samples, shots=shots)

        expected = tuple(
            tuple((3,) if s == 1 else (s, 3) for _ in range(num_samples)) for s in shots
        )

        res = qs.shape(dev)
        assert res == expected

        program, _ = dev.preprocess()
        expected = qml.execute([qs], dev, gradient_fn=None, transform_program=program)[0]
        expected_shape = tuple(tuple(e_.shape for e_ in e) for e in expected)

        assert res == expected_shape

    def test_raises_broadcasting_shot_vector(self):
        """Test that getting the output shape of a tape that uses parameter
        broadcasting along with a device with a shot vector raises an error."""
        dev = qml.device("default.qubit", wires=3, shots=(1, 2, 3))

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(np.array([0.1, 0.2]), wires=0)
            qml.RX(np.array([0.3, 0.4]), wires=0)
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q, shots=(1, 2, 3))
        msg = "Parameter broadcasting when using a shot vector is not supported yet"

        with pytest.raises(NotImplementedError, match=msg):
            tape.shape(dev)


class TestNumericType:
    """Tests for determining the numeric type of the tape output."""

    @pytest.mark.parametrize(
        "ret",
        [
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.probs(wires=[0]),
            qml.mutual_info(wires0=0, wires1=1),
            qml.vn_entropy(wires=[0, 1]),
        ],
    )
    @pytest.mark.parametrize("shots", [None, 1, (1, 2, 3)])
    def test_float_measures(self, ret, shots):
        """Test that most measurements output floating point values and that
        the tape output domain correctly identifies this."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
        if shots and isinstance(ret, (qml.measurements.MutualInfoMP, qml.measurements.VnEntropyMP)):
            pytest.skip("Shots and entropies not supported.")

        a, b = 0.3, 0.2
        ops = [qml.RY(a, 0), qml.RZ(b, 0)]
        qs = QuantumScript(ops, [ret], shots=shots)
        result = qml.execute([qs], dev, gradient_fn=None)[0]

        if not isinstance(result, tuple):
            result = (result,)

        # Double-check the domain of the QNode output
        assert all(np.issubdtype(res.dtype, float) for res in result)
        assert qs.numeric_type is float

    @pytest.mark.parametrize(
        "ret", [qml.state(), qml.density_matrix(wires=[0, 1]), qml.density_matrix(wires=[2, 0])]
    )
    def test_complex_state(self, ret):
        """Test that a tape with qml.state correctly determines that the output
        domain will be complex."""
        dev = qml.device("default.qubit", wires=3)

        a, b = 0.3, 0.2
        ops = [qml.RY(a, 0), qml.RZ(b, 0)]
        qs = QuantumScript(ops, [ret])

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, complex)
        assert qs.numeric_type is complex

    @pytest.mark.parametrize("ret", [qml.sample(), qml.sample(qml.PauliZ(wires=0))])
    def test_sample_int_eigvals(self, ret):
        """Test that the tape can correctly determine the output domain for a
        sampling measurement with a Hermitian observable with integer
        eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)
        qs = QuantumScript([qml.RY(0.4, 0)], [ret], shots=5)

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, np.int64)
        assert qs.numeric_type is int

    # TODO: add cases for each interface once qml.Hermitian supports other
    # interfaces
    def test_sample_real_eigvals(self):
        """Test that the tape can correctly determine the output domain when
        sampling a Hermitian observable with real eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        qs = QuantumScript([qml.RY(0.4, 0)], [qml.sample(qml.Hermitian(herm, wires=0))], shots=5)

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, float)
        assert qs.numeric_type is float

    @pytest.mark.autograd
    def test_sample_real_and_int_eigvals(self):
        """Test that the tape can correctly determine the output domain for
        multiple sampling measurements with a Hermitian observable with real
        eigenvalues and another one with integer eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        a, b = 0, 3
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        m = [qml.sample(qml.Hermitian(herm, wires=0)), qml.sample(qml.PauliZ(1))]
        qs = QuantumScript(ops, m, shots=5)

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result[0].dtype, float)
        assert np.issubdtype(result[1].dtype, np.int64)
        assert qs.numeric_type == (float, int)


@pytest.mark.parametrize("qscript_type", (QuantumScript, qml.tape.QuantumTape))
def test_flatten_unflatten(qscript_type):
    """Test the flatten and unflatten methods."""
    ops = [qml.RX(0.1, wires=0), qml.U3(0.2, 0.3, 0.4, wires=0)]
    mps = [qml.expval(qml.PauliZ(0)), qml.state()]

    tape = qscript_type(ops, mps, shots=100)
    tape.trainable_params = {0}

    data, metadata = tape._flatten()
    assert all(o1 is o2 for o1, o2 in zip(ops, data[0]))
    assert all(o1 is o2 for o1, o2 in zip(mps, data[1]))
    assert metadata[0] == qml.measurements.Shots(100)
    assert metadata[1] == (0,)
    assert hash(metadata)

    new_tape = qscript_type._unflatten(data, metadata)
    assert all(o1 is o2 for o1, o2 in zip(new_tape.operations, tape.operations))
    assert all(o1 is o2 for o1, o2 in zip(new_tape.measurements, tape.measurements))
    assert new_tape.shots == qml.measurements.Shots(100)
    assert new_tape.trainable_params == [0]


@pytest.mark.jax
@pytest.mark.parametrize("qscript_type", (QuantumScript, qml.tape.QuantumTape))
def test_jax_pytree_integration(qscript_type):
    """Test that QuantumScripts are integrated with jax pytress."""

    eye_mat = np.eye(4)
    ops = [qml.adjoint(qml.RY(0.5, wires=0)), qml.Rot(1.2, 2.3, 3.4, wires=0)]
    mps = [
        qml.var(qml.s_prod(2.0, qml.PauliX(0))),
        qml.expval(qml.Hermitian(eye_mat, wires=(0, 1))),
    ]

    tape = qscript_type(ops, mps, shots=100)
    tape.trainable_params = [2]

    import jax

    data, _ = jax.tree_util.tree_flatten(tape)
    assert data[0] == 0.5
    assert data[1] == 1.2
    assert data[2] == 2.3
    assert data[3] == 3.4
    assert data[4] == 2.0
    assert qml.math.allclose(data[5], eye_mat)
