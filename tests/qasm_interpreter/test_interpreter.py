import pytest
from openqasm3.parser import parse
from pennylane.io.qasm_interpreter import QasmInterpreter

class TestInterpreter:

    qasm_programs = [
        (
            open('adder.qasm', mode='r').read(),
            22,
            'adder'
        ),
        (
            open('qec.qasm', mode='r').read(),
            15,
            'qec'
        ),
        (
            open('teleport.qasm', mode='r').read(),
            19,
            'teleport'
        )
    ]

    @pytest.mark.parametrize("qasm_program, count_nodes, program_name", qasm_programs)
    def test_visits_each_node(self, qasm_program, count_nodes, program_name, mocker):
        """Tests that visitor is called on each element of the AST."""
        ast = parse(qasm_program, permissive=True)
        spy = mocker.spy(QasmInterpreter, "visit")
        QasmInterpreter().generic_visit(ast, context={"program_name": program_name})
        assert spy.call_count == count_nodes

    def test_parses_simple_qasm(self):
        ast = parse(open('gates.qasm', mode='r').read(), permissive=True)
        context = QasmInterpreter().generic_visit(ast, context={"program_name": "gates"})
        context["qnode"].func()
