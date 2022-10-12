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
"""Unittests for QuantumScript.

Things left to unittest:
* Output shape
* Numeric Type
* Expand
* parameter stuff
* qasm
"""
from collections import defaultdict
import copy
import pytest

import pennylane as qml
from pennylane import numpy as np
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
        """Test provided ops are converted to lists."""
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
    """Test the methods called by _update."""

    def test_cached_graph_specs_reset(self):
        """Test that update resets the graph and specs"""
        qs = QuantumScript()
        qs._graph = "hello"
        qs._specs = "something"

        qs._update()
        assert qs._graph is None
        assert qs._specs is None

    def test_update_circuit_info_wires(self):
        """Test that on construction wires and num_wires are set."""
        prep = [qml.BasisState([1, 1], wires=(-1, -2))]
        ops = [qml.S(0), qml.T("a"), qml.S(0)]
        measurement = [qml.probs(wires=("a"))]

        qs = QuantumScript(ops, measurement, prep)
        assert qs.wires == qml.wires.Wires([-1, -2, 0, "a"])
        assert qs.num_wires == 4

    @pytest.mark.parametrize("sample_ms", sample_measurements)
    def test_update_circuit_info_sampling(self, sample_ms):
        qs = QuantumScript(measurements=[qml.expval(qml.PauliZ(0)), sample_ms])
        assert qs.is_sampled is True
        assert qs.all_sampled is False

        qs = QuantumScript(measurements=[sample_ms, sample_ms, qml.sample()])
        assert qs.is_sampled is True
        assert qs.all_sampled is True

    def test_update_circuit_info_no_sampling(self):
        """Test that all_sampled and is_sampled properties are set to False if no sampling
        measurement process exists."""
        qs = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        assert qs.is_sampled is False
        assert qs.all_sampled is False

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

        assert p_i[0] == {"op": ops[0], "p_idx": 0}
        assert p_i[1] == {"op": ops[1], "p_idx": 0}
        assert p_i[2] == {"op": ops[1], "p_idx": 1}
        assert p_i[3] == {"op": ops[1], "p_idx": 2}
        assert p_i[4] == {"op": ops[2], "p_idx": 0}
        assert p_i[5] == {"op": ops[3], "p_idx": 0}
        assert p_i[6] == {"op": ops[3], "p_idx": 1}
        assert p_i[7] == {"op": m[0].obs, "p_idx": 0}

        assert qs._trainable_params == list(range(8))

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
        batch_size, when creating and when using `set_parameters`."""

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
            qs = QuantumScript(ops)

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
            tape[idx] is exp_elem

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

        assert qs.specs["gate_sizes"] == defaultdict(int)
        assert qs.specs["gate_types"] == defaultdict(int)

        assert qs.specs["num_operations"] == 0
        assert qs.specs["num_observables"] == 0
        assert qs.specs["num_diagonalizing_gates"] == 0
        assert qs.specs["num_used_wires"] == 0
        assert qs.specs["num_trainable_params"] == 0
        assert qs.specs["depth"] == 0

        assert len(qs.specs) == 8

        assert qs._specs is qs.specs

    def test_specs_tape(self, make_script):
        """Tests that regular scripts return correct specifications"""
        qs = make_script

        assert qs._specs is None
        specs = qs.specs
        assert qs._specs is specs

        assert len(specs) == 8

        assert specs["gate_sizes"] == defaultdict(int, {1: 3, 2: 1})
        assert specs["gate_types"] == defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 1})
        assert specs["num_operations"] == 4
        assert specs["num_observables"] == 2
        assert specs["num_diagonalizing_gates"] == 1
        assert specs["num_used_wires"] == 3
        assert specs["num_trainable_params"] == 5
        assert specs["depth"] == 3


class TestScriptCopying:
    """Test for quantum script copying behaviour"""

    def test_shallow_copy(self):
        """Test that shallow copying of a script results in all
        contained data being shared between the original tape and the copy"""
        prep = [qml.BasisState(np.array([1, 0]), wires=(0, 1))]
        ops = [qml.RY(0.5, wires=1), qml.CNOT((0, 1))]
        m = [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))]
        qs = QuantumScript(ops, m, prep=prep)

        copied_qs = qs.copy()

        assert copied_qs is not qs

        # the operations are simply references
        assert copied_qs.operations == qs.operations
        assert copied_qs.observables == qs.observables
        assert copied_qs.measurements == qs.measurements
        assert copied_qs.operations[0] is qs.operations[0]

        # operation data is also a reference
        assert copied_qs.operations[0].wires is qs.operations[0].wires
        assert copied_qs.operations[0].data[0] is qs.operations[0].data[0]

        # check that all tape metadata is identical
        assert qs.get_parameters() == copied_qs.get_parameters()
        assert qs.wires == copied_qs.wires
        assert qs.data == copied_qs.data

        # check that the output dim is identical
        assert qs.output_dim == copied_qs.output_dim

        # since the copy is shallow, mutating the parameters
        # on one tape will affect the parameters on another tape
        new_params = [np.array([0, 0]), 0.2]
        qs.set_parameters(new_params)

        # check that they are the same objects in memory
        for i, j in zip(qs.get_parameters(), new_params):
            assert i is j

        for i, j in zip(copied_qs.get_parameters(), new_params):
            assert i is j

    @pytest.mark.parametrize(
        "copy_fn", [lambda tape: tape.copy(copy_operations=True), lambda tape: copy.copy(tape)]
    )
    def test_shallow_copy_with_operations(self, copy_fn):
        """Test that shallow copying of a tape and operations allows
        parameters to be set independently"""

        prep = [qml.BasisState(np.array([1, 0]), wires=(0, 1))]
        ops = [qml.RY(0.5, wires=1), qml.CNOT((0, 1))]
        m = [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))]
        qs = QuantumScript(ops, m, prep=prep)

        copied_qs = copy_fn(qs)

        assert copied_qs is not qs

        # the operations are not references; they are unique objects
        assert copied_qs.operations != qs.operations
        assert copied_qs.observables != qs.observables
        assert copied_qs.measurements != qs.measurements
        assert copied_qs.operations[0] is not qs.operations[0]

        # however, the underlying operation data *is still shared*
        assert copied_qs.operations[0].wires is qs.operations[0].wires
        # the data list is copied, but the elements of the list remain th same
        assert copied_qs.operations[0].data is not qs.operations[0].data
        assert copied_qs.operations[0].data[0] is qs.operations[0].data[0]

        assert qs.get_parameters() == copied_qs.get_parameters()
        assert qs.wires == copied_qs.wires
        assert qs.data == copied_qs.data

        # check that the output dim is identical
        assert qs.output_dim == copied_qs.output_dim

        # Since they have unique operations, mutating the parameters
        # on one script will *not* affect the parameters on another script
        new_params = [np.array([0, 0]), 0.2]
        qs.set_parameters(new_params)

        for i, j in zip(qs.get_parameters(), new_params):
            assert i is j

        for i, j in zip(copied_qs.get_parameters(), new_params):
            assert not np.all(i == j)
            assert i is not j

    def test_deep_copy(self):
        """Test that deep copying a tape works, and copies all constituent data except parameters"""
        prep = [qml.BasisState(np.array([1, 0]), wires=(0, 1))]
        ops = [qml.RY(0.5, wires=1), qml.CNOT((0, 1))]
        m = [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))]
        qs = QuantumScript(ops, m, prep=prep)

        copied_qs = copy.deepcopy(qs)

        assert copied_qs is not qs

        # the operations are not references
        assert copied_qs.operations != qs.operations
        assert copied_qs.observables != qs.observables
        assert copied_qs.measurements != qs.measurements
        assert copied_qs.operations[0] is not qs.operations[0]

        # check that the output dim is identical
        assert qs.output_dim == copied_qs.output_dim

        # The underlying operation data has also been copied
        assert copied_qs.operations[0].wires is not qs.operations[0].wires

        # however, the underlying operation *parameters* are still shared
        # to support PyTorch, which does not support deep copying of tensors
        assert copied_qs.operations[0].data[0] is qs.operations[0].data[0]


def test_adjoint():
    """Tests taking the adjoint of a quantum script."""
    ops = [qml.RX(1.2, wires=0), qml.S(0), qml.CNOT((0, 1)), qml.T(1)]
    prep = [qml.BasisState([1, 1], wires=0)]
    m = [qml.expval(qml.PauliZ(0))]
    qs = QuantumScript(ops, m, prep)

    with qml.queuing.AnnotatedQueue() as q:
        adj_qs = qs.adjoint()

    assert len(q.queue) == 0  # not queued

    assert adj_qs._prep == qs._prep
    assert adj_qs._measurements == qs._measurements

    # assumes lazy=False
    expected_ops = [qml.adjoint(qml.T(1)), qml.CNOT((0, 1)), qml.adjoint(qml.S(0)), qml.RX(-1.2, 0)]
    for op, expected in zip(adj_qs._ops, expected_ops):
        # update this one qml.equal works with adjoint
        assert isinstance(op, type(expected))
        assert op.wires == expected.wires
        assert op.data == expected.data


@pytest.mark.torch
def test_unwrap():
    """Tests the unwrap method."""

    import torch

    x = torch.tensor(1.2)
    qs = QuantumScript([qml.RX(x, 0)])

    unwrapper = qs.unwrap()
    assert isinstance(unwrapper, qml.tape.UnwrapTape)

    with unwrapper:
        assert qml.math.get_interface(qs.data[0]) == "numpy"
    assert qml.math.get_interface(qs.data[0]) == "torch"


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
        a = 0.3
        b = 0.2
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
