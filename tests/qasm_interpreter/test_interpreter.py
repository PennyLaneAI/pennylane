import pytest
from openqasm3.parser import parse

from pennylane import CNOT, RX, RY, PauliX
from pennylane.io.qasm_interpreter import QasmInterpreter


class TestInterpreter:

    qasm_programs = [
        (open("tests/qasm_interpreter/adder.qasm", mode="r").read(), 22, "adder"),
        (open("tests/qasm_interpreter/qec.qasm", mode="r").read(), 15, "qec"),
        (open("tests/qasm_interpreter/teleport.qasm", mode="r").read(), 19, "teleport"),
    ]

    @pytest.mark.parametrize("qasm_program, count_nodes, program_name", qasm_programs)
    def test_visits_each_node(self, qasm_program, count_nodes, program_name, mocker):
        """Tests that visitor is called on each element of the AST."""
        ast = parse(qasm_program, permissive=True)
        spy = mocker.spy(QasmInterpreter, "visit")
        QasmInterpreter().generic_visit(ast, context={"name": program_name})
        assert spy.call_count == count_nodes

    def test_interprets_simple_qasm(self, mocker):

        # parse the QASM program
        ast = parse(open('gates.qasm', mode='r').read(), permissive=True)
        context = QasmInterpreter().generic_visit(ast, context={"name": "gates"})

        # setup mocks
        x = mocker.spy(PauliX, "__init__")
        cx = mocker.spy(CNOT, "__init__")
        rx = mocker.spy(RX, "__init__")
        ry = mocker.spy(RY, "__init__")

        # execute the QNode
        context["qnode"].func()

        # asserts
        assert len(context["device"].wires) == 2

        assert x.call_count == 5  # RX calls PauliX under the hood
        x.assert_called_with(PauliX(0), wires=0)

        assert cx.call_count == 2  # verifies that ctrl @ x q1, q0 calls cx too
        cx.assert_called_with(CNOT([0, 1]), wires=[0, 1])
        cx.assert_called_with(CNOT([1, 0]), wires=[1, 0])

        assert rx.call_count == 2  # one adjoint call and one direct call
        rx.assert_called_with(RX(0.5, 0), 0.5, wires=0)

        assert ry.call_count == 1
        ry.assert_called_with(RY(0.2, [0]), 0.2, wires=[0])

    def test_control_flow(self):

        # parse the QASM program
        ast = parse(open('control_flow.qasm', mode='r').read(), permissive=True)
        context = QasmInterpreter().generic_visit(ast, context={"name": "control-flow"})

        # TODO: compare to a QNode constructed in the typical way with a decorator
