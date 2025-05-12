import pytest
from openqasm3.parser import parse
from pennylane.io.qasm_interpreter import QasmInterpreter

class TestInterpreter:

    qasm_programs = [
        (
            open('adder.qasm', mode='r').read(),
            22
        ),
        (
            open('qec.qasm', mode='r').read(),
            15
        ),
        (
            open('teleport.qasm', mode='r').read(),
            19
        )
    ]

    @pytest.mark.parametrize("qasm_program, count_nodes", qasm_programs)
    def test_visits_each_node(self, qasm_program, count_nodes, mocker):
        """Tests that visitor is called on each element of the AST."""
        ast = parse(qasm_program, permissive=True)
        spy = mocker.spy(QasmInterpreter, "visit")
        QasmInterpreter().generic_visit(ast)
        assert spy.call_count == count_nodes

