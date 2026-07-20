# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.circuit_graph` module.
"""

# pylint: disable=no-self-use,too-many-arguments,protected-access

import numpy as np
import pytest

import pennylane as qp
from pennylane.circuit_graph import CircuitGraph, _WrappedObj
from pennylane.ops.mid_measure.measurement_value import MeasurementValue
from pennylane.ops.mid_measure.mid_measure import MidMeasure
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.ops.op_math.condition import Conditional
from pennylane.wires import Wires


class Test_WrappedObj:
    """Tests for the ``_WrappedObj`` class"""

    @pytest.mark.parametrize("obj", [qp.PauliX(0), qp.expval(qp.PauliZ(0)), [0, 1, 2], ("a", "b")])
    def test_wrapped_obj_init(self, obj):
        """Test that ``_WrappedObj`` is initialized correctly"""
        wo = _WrappedObj(obj)
        assert wo.obj is obj

    @pytest.mark.parametrize(
        "obj1, obj2",
        [(qp.PauliX(0), qp.PauliZ(0)), (qp.PauliX(0), qp.PauliX(0)), ((1,), (1, 2))],
    )
    def test_wrapped_obj_eq_false(self, obj1, obj2):
        """Test that ``_WrappedObj.__eq__`` returns False when expected."""
        wo1 = _WrappedObj(obj1)
        wo2 = _WrappedObj(obj2)
        assert wo1 != wo2

    def test_wrapped_obj_eq_false_other_obj(self):
        """Test that _WrappedObj.__eq__ returns False when the object being compared is not
        a _WrappedObj."""
        op = qp.PauliX(0)
        wo = _WrappedObj(op)
        assert wo != op

    def test_wrapped_obj_eq_true(self):
        """Test that ``_WrappedObj.__eq__`` returns True when expected."""
        op = qp.PauliX(0)
        assert _WrappedObj(op) == _WrappedObj(op)

    @pytest.mark.parametrize("obj", [qp.PauliX(0), qp.expval(qp.PauliZ(0)), [0, 1, 2], ("a", "b")])
    def test_wrapped_obj_hash(self, obj):
        """Test that ``_WrappedObj.__hash__`` is the object id."""
        wo = _WrappedObj(obj)
        assert wo.__hash__() == id(obj)  # pylint: disable=unnecessary-dunder-call

    def test_wrapped_obj_repr(self):
        """Test that the ``_WrappedObj` representation is equivalent to the repr of the
        object it wraps."""

        class Dummy:  # pylint: disable=too-few-public-methods
            """Dummy class with custom repr"""

            def __repr__(self):
                return "test_repr"

        obj = Dummy()
        wo = _WrappedObj(obj)
        assert wo.__repr__() == "_Wrapped(test_repr)"  # pylint: disable=unnecessary-dunder-call


@pytest.fixture(name="ops")
def ops_fixture():
    """A fixture of a complex example of operations that depend on previous operations."""
    return [
        qp.RX(0.43, wires=0),
        qp.RY(0.35, wires=1),
        qp.RZ(0.35, wires=2),
        qp.CNOT(wires=[0, 1]),
        qp.Hadamard(wires=2),
        qp.CNOT(wires=[2, 0]),
        qp.PauliX(wires=1),
    ]


@pytest.fixture(name="obs")
def obs_fixture():
    """A fixture of observables to go after the queue fixture."""
    return [
        qp.expval(qp.PauliX(wires=0)),
        qp.expval(qp.Hermitian(np.identity(4), wires=[1, 2])),
    ]


@pytest.fixture(name="circuit")
def circuit_fixture(ops, obs):
    """A fixture of a circuit generated based on the ops and obs fixtures above."""
    return CircuitGraph(ops, obs, Wires([0, 1, 2]))


def circuit_measure_max_once():
    """A fixture of a circuit that measures wire 0 once."""
    return qp.expval(qp.PauliX(wires=0))


def circuit_measure_max_twice():
    """A fixture of a circuit that measures wire 0 twice."""
    return qp.expval(qp.PauliZ(wires=0)), qp.probs(wires=0)


def circuit_measure_multiple_with_max_twice():
    """A fixture of a circuit that measures wire 0 twice."""
    return (
        qp.expval(qp.PauliZ(wires=0)),
        qp.probs(wires=[0, 1, 2]),
        qp.var(qp.PauliZ(wires=[1]) @ qp.PauliZ([2])),
    )


# pylint: disable=too-many-public-methods
class TestCircuitGraph:
    """Test conversion of queues to DAGs"""

    def test_no_dependence(self):
        """Test case where operations do not depend on each other.
        This should result in a graph with no edges."""

        ops = [qp.RX(0.43, wires=0), qp.RY(0.35, wires=1)]

        res = CircuitGraph(ops, [], Wires([0, 1])).graph
        assert len(res) == 2
        assert not res.edges()

    def test_dependence(self, ops, obs):
        """Test a more complex example containing operations
        that do depend on the result of previous operations"""

        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        graph = circuit.graph
        assert len(graph.node_indexes()) == 9
        assert len(graph.edges()) == 9

        a = {(graph.get_node_data(e[0]), graph.get_node_data(e[1])) for e in graph.edge_list()}

        b = {
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (5, 7),
            (5, 8),
            (6, 8),
        }
        assert a == b

    def test_ancestors_and_descendants_example(self, ops, obs):
        """
        Test that the ``ancestors`` and ``descendants`` methods return the expected result.
        """
        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))

        queue = ops + obs

        ancestors = circuit.ancestors([queue[6]])
        ancestors_index = circuit.ancestors_of_indexes([6])
        assert len(ancestors) == len(ancestors_index) == 3
        for o_idx in (0, 1, 3):
            assert queue[o_idx] in ancestors
            assert queue[o_idx] in ancestors_index

        descendants = circuit.descendants([queue[6]])
        descendants_index = circuit.descendants_of_indexes([6])
        assert descendants == [queue[8]]
        assert descendants_index == [queue[8]]

    @pytest.mark.parametrize("sort", [True, False])
    def test_ancestors_and_descendents_repeated_op(self, sort):
        """Test ancestors and descendents raises a ValueError is the requested operation occurs more than once."""

        op = qp.X(0)
        ops = [op, qp.Y(0), op, qp.Z(0), op]
        graph = CircuitGraph(ops, [], qp.wires.Wires([0, 1, 2]))

        with pytest.raises(ValueError, match=r"operator that occurs multiple times."):
            graph.ancestors([op], sort=sort)
        with pytest.raises(ValueError, match=r"operator that occurs multiple times."):
            graph.descendants([op], sort=sort)

    @pytest.mark.parametrize("sort", [True, False])
    def test_ancestors_and_descendents_single_op_error(self, sort):
        """Test ancestors and descendents raises a ValueError is the requested operation occurs more than once."""

        op = qp.Z(0)
        graph = CircuitGraph([op], [], [0, 1, 2])

        with pytest.raises(
            ValueError, match=r"CircuitGraph.ancestors accepts an iterable of operators"
        ):
            graph.ancestors(op, sort=sort)
        with pytest.raises(
            ValueError, match=r"CircuitGraph.descendants accepts an iterable of operators"
        ):
            graph.descendants(op, sort=sort)

    def test_update_node(self, ops, obs):
        """Changing nodes in the graph."""

        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        new = qp.RX(0.1, wires=0)
        circuit.update_node(ops[0], new)
        assert circuit.operations[0] is new
        new_mp = qp.var(qp.Y(0))
        circuit.update_node(obs[0], new_mp)
        assert circuit.observables[0] is new_mp

    def test_update_node_error(self, ops, obs):
        """Test that changing nodes in the graph may raise an error."""
        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        new = qp.RX(0.1, wires=0)
        new = qp.CNOT(wires=[0, 1])
        with pytest.raises(ValueError):
            circuit.update_node(ops[0], new)

    def test_observables(self, circuit, obs):
        """Test that the `observables` property returns the list of observables in the circuit."""
        assert str(circuit.observables) == str(obs)

    def test_operations(self, circuit, ops):
        """Test that the `operations` property returns the list of operations in the circuit."""
        assert str(circuit.operations) == str(ops)

    def test_op_indices(self, circuit):
        """Test that for the given circuit, this method will fetch the correct operation indices for
        a given wire"""
        op_indices_for_wire_0 = [0, 3, 5, 7]
        op_indices_for_wire_1 = [1, 3, 6, 8]
        op_indices_for_wire_2 = [2, 4, 5, 8]

        assert circuit.wire_indices(0) == op_indices_for_wire_0
        assert circuit.wire_indices(1) == op_indices_for_wire_1
        assert circuit.wire_indices(2) == op_indices_for_wire_2

    def test_layers(self):
        """A test of a simple circuit with 3 parametrized layers."""
        ops = [
            qp.RX(0.1, wires=0),
            qp.RX(0.2, wires=1),
            qp.RX(0.3, wires=2),
            qp.CRX(0.4, wires=[0, 1]),
            qp.RX(0.5, wires=1),
            qp.RX(0.6, wires=2),
        ]
        par_info = [{"op": op, "op_idx": idx, "p_idx": 0} for idx, op in enumerate(ops)]
        circuit = CircuitGraph(
            ops,
            [],
            wires=Wires([0, 1, 2]),
            par_info=par_info,
            trainable_params=set(range(len(ops))),
        )
        layers = circuit.parametrized_layers

        assert len(layers) == 3
        assert layers[0].ops == [ops[x] for x in [0, 1, 2]]
        assert layers[0].param_inds == [0, 1, 2]
        assert layers[1].ops == [ops[3]]
        assert layers[1].param_inds == [3]
        assert layers[2].ops == [ops[x] for x in [4, 5]]
        assert layers[2].param_inds == [4, 5]

    def test_iterate_layers_repeat_op(self):
        """Test iterate_parametrized_layers can work when the operation is repeated."""
        op = qp.RX(0.5, 0)
        par_info = [{"op": op, "op_idx": 0, "p_idx": 0}, {"op": op, "op_idx": 2, "p_idx": 0}]
        graph = qp.CircuitGraph(
            [op, qp.X(0), op], [], wires=op.wires, trainable_params={0, 1}, par_info=par_info
        )
        layers = list(graph.iterate_parametrized_layers())

        assert len(layers) == 2

        assert layers[0].pre_ops == []
        assert layers[0].ops == [op]
        assert layers[0].param_inds == (0,)
        assert layers[0].post_ops == [qp.X(0), op]

        assert layers[1].ops == [op]
        assert layers[1].param_inds == (1,)
        assert layers[1].pre_ops == [op, qp.X(0)]
        assert layers[1].post_ops == []

    def test_iterate_layers(self):
        """A test of the different layers, their successors and ancestors using a simple circuit."""
        ops = [
            qp.RX(0.1, wires=0),
            qp.RX(0.2, wires=1),
            qp.RX(0.3, wires=2),
            qp.CRX(0.4, wires=[0, 1]),
            qp.RX(0.5, wires=1),
            qp.RX(0.6, wires=2),
        ]
        par_info = [{"op": op, "op_idx": idx, "p_idx": 0} for idx, op in enumerate(ops)]
        circuit = CircuitGraph(
            ops,
            [],
            wires=Wires([0, 1, 2]),
            par_info=par_info,
            trainable_params=set(range(len(ops))),
        )
        result = list(circuit.iterate_parametrized_layers())

        assert len(result) == 3

        assert set(result[0][0]) == set()
        assert set(result[0][1]) == set(ops[:3])
        assert result[0][2] == (0, 1, 2)
        assert set(result[0][3]) == set(ops[3:])

        assert set(result[1][0]) == set(ops[:2])
        assert set(result[1][1]) == {ops[3]}
        assert result[1][2] == (3,)
        assert set(result[1][3]) == {ops[4]}

        assert set(result[2][0]) == set(ops[:4])
        assert set(result[2][1]) == set(ops[4:])
        assert result[2][2] == (4, 5)
        assert set(result[2][3]) == set()

    @pytest.mark.parametrize(
        "circ, expected",
        [
            (circuit_measure_max_once, 1),
            (circuit_measure_max_twice, 2),
            (circuit_measure_multiple_with_max_twice, 2),
        ],
    )
    def test_max_simultaneous_measurements(self, circ, expected):
        """A test for getting the maximum number of measurements on any wire in
        the circuit graph."""

        dev = qp.device("default.qubit", wires=3)
        qnode = qp.QNode(circ, dev)
        tape = qp.workflow.construct_tape(qnode)()
        circuit = tape.graph
        assert circuit.max_simultaneous_measurements == expected

    def test_str_print(self):
        """Tests if the circuit prints correct."""
        ops = [qp.Hadamard(wires=0), qp.CNOT(wires=[0, 1])]
        obs_w_wires = [qp.measurements.sample(op=None, wires=[0, 1, 2])]

        circuit_w_wires = CircuitGraph(ops, obs_w_wires, wires=Wires([0, 1, 2]))
        expected = """Operations\n==========\nH(0)\nCNOT(wires=[0, 1])\n\nObservables\n===========\nsample(wires=[0, 1, 2])\n"""
        assert str(circuit_w_wires) == expected


def test_has_path():
    """Test has_path and has_path_idx."""

    ops = [qp.X(0), qp.X(3), qp.CNOT((0, 1)), qp.X(1), qp.X(3)]
    graph = CircuitGraph(ops, [], wires=[0, 1, 2, 3, 4, 5])

    assert graph.has_path(ops[0], ops[2])
    assert graph.has_path_idx(0, 2)
    assert not graph.has_path(ops[0], ops[4])
    assert not graph.has_path_idx(0, 4)


def test_path_from_mcm_to_conditional():
    mcm = MidMeasure(wires=Wires([0]))
    ppm = PauliMeasure("XY", wires=Wires([0, 1]))
    m0 = MeasurementValue([mcm, ppm])
    ops = [mcm, ppm, Conditional(m0, qp.Z(0))]
    graph = CircuitGraph(ops, [], wires=Wires([0, 1, 2]))
    assert graph.has_path(mcm, ops[2])
    assert graph.has_path(ppm, ops[2])


def test_has_path_repeated_ops():
    """Test has_path and has_path_idx when an operation is repeated."""

    op = qp.X(0)
    ops = [op, qp.CNOT((0, 1)), op, qp.Y(1)]

    graph = CircuitGraph(ops, [], [0, 1, 2, 3])

    assert graph.has_path_idx(0, 3)
    assert graph.has_path_idx(1, 2)
    with pytest.raises(ValueError, match="does not work with operations that have been repeated. "):
        graph.has_path(op, ops[3])
    with pytest.raises(ValueError, match="does not work with operations that have been repeated. "):
        graph.has_path(ops[1], op)

    # still works if they are the same operation.
    assert graph.has_path(op, op)
