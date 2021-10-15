# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`pennylane.operation`.
"""
import pytest
import pennylane as qml


class TestCommutingFunction:
    """Commutation function tests."""

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1, 0]], False),
            ([[1, 0], [1, 0]], True),
            ([[0, 1], [2, 3]], True),
            ([[0, 1], [3, 1]], True),
        ],
    )
    def test_cnot(self, wires, res, tol):
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1, 2], [1, 0, 2]], True),
            ([[1, 2], [0, 1, 2]], True),
            ([[3, 2], [0, 1, 2]], True),
            ([[0, 1], [0, 1, 2]], False),
        ],
    )
    def test_cnot_toffoli(self, wires, res, tol):
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.Toffoli(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1, 2], [1, 0]], True),
            ([[0, 1], [0, 1]], False),
            ([[0, 1], [2, 0]], True),
            ([[0, 1], [0, 2]], True),
        ],
    )
    def test_cnot_cz(self, wires, res, tol):
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.CZ(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1, 2]], True),
            ([[0, 2], [0, 1, 2]], True),
            ([[0, 2], [0, 2, 1]], True),
        ],
    )
    def test_cz_mcz(self, wires, res, tol):
        def z():
            qml.PauliZ(wires=wires[1][1])

        commutation = qml.is_commuting(
            qml.CZ(wires=wires[0]), qml.transforms.ctrl(z, control=wires[1][0])()
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1, 2]], False),
            ([[0, 2], [0, 1, 2]], False),
            ([[0, 2], [0, 2, 1]], False),
            ([[0, 3], [0, 2, 1]], True),
            ([[0, 3], [1, 2, 0]], True),
        ],
    )
    def test_cnot_mcz(self, wires, res, tol):
        def z():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(
            qml.CNOT(wires=wires[0]), qml.transforms.ctrl(z, control=wires[1][:-1])()
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1], [0, 1]], True),
            ([[0], [0, 1]], False),
            ([[2], [0, 1]], True),
        ],
    )
    def test_x_cnot(self, wires, res, tol):

        commutation = qml.is_commuting(qml.PauliX(wires=wires[0]), qml.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1], [0, 1]], True),
            ([[0], [0, 1]], False),
            ([[2], [0, 1]], True),
        ],
    )
    def test_cnot_x(self, wires, res, tol):

        commutation = qml.is_commuting(qml.CNOT(wires=wires[1]), qml.PauliX(wires=wires[0]))
        assert commutation == res


class TestCommutationDAG:
    """Commutation DAG tests."""

    @pytest.mark.parametrize(
        "wires",
        [
            ([0, 1]),
            ([1, 0]),
        ],
    )
    def test_empty_dag(self, wires):
        qml.commutation_dag.CommutationDAG(qml.wires.Wires(wires))


    def test_dag_transform_simple_dag_function(self):
        "Test a simple DAG on 1 wire with a quantum function."

        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=0)

        dag = qml.transforms.get_dag_commutation(circuit)()

        a = qml.PauliZ(wires=0)
        b = qml.PauliX(wires=0)

        assert dag.get_node(0).op.compare(a)
        assert dag.get_node(1).op.compare(b)
        assert dag.get_edge(0, 1) == {0: {"commute": False}}
        assert dag.get_edge(0, 2) is None

    def test_dag_transform_simple_dag_qnode(self):
        "Test a simple DAG on 1 wire with a qnode."

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliX(wires=0))

        dag = qml.transforms.get_dag_commutation(circuit)()

        a = qml.PauliZ(wires=0)
        b = qml.PauliX(wires=0)

        assert dag.get_node(0).op.compare(a)
        assert dag.get_node(1).op.compare(b)
        assert dag.get_edge(0, 1) == {0: {"commute": False}}
        assert dag.get_edge(0, 2) is None
        assert dag.observables[0].return_type.__repr__() == "expval"
        assert dag.observables[0].name == "PauliX"
        assert dag.observables[0].wires.tolist() == [0]