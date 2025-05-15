import pytest
from openqasm3.parser import parse

from pennylane import CNOT, RX, RY, PauliX
from pennylane.io.qasm_interpreter import QasmInterpreter


class TestInterpreter:

    def test_parses_simple_qasm(self, mocker):

        # parse the QASM program
        ast = parse(open("gates.qasm", mode="r").read(), permissive=True)
        context = QasmInterpreter().generic_visit(ast, context={"program_name": "gates"})

        # setup mocks
        x = mocker.spy(PauliX, "__init__")
        cx = mocker.spy(CNOT, "__init__")
        rx = mocker.spy(RX, "__init__")
        ry = mocker.spy(RY, "__init__")

        # execute the callable
        context["callable"]()

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
