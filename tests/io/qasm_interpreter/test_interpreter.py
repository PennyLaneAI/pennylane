"""
Unit tests for the :mod:`pennylane.io.qasm_interpreter` module.
"""

import numpy as np
import pytest

from pennylane import (
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    SX,
    U1,
    U2,
    U3,
    CPhase,
    Hadamard,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    S,
    T,
    Toffoli,
)
from pennylane.wires import Wires

pytest.importorskip("openqasm3")

from openqasm3.parser import parse

from pennylane.io.qasm_interpreter import QasmInterpreter


@pytest.mark.external
class TestInterpreter:

    qasm_programs = [
        (open("adder.qasm", mode="r").read(), 22, "adder"),
        (open("qec.qasm", mode="r").read(), 32, "qec"),
        (open("teleport.qasm", mode="r").read(), 25, "teleport"),
    ]

    @pytest.mark.parametrize("qasm_program, count_nodes, program_name", qasm_programs)
    def test_visits_each_node(self, qasm_program, count_nodes, program_name, mocker):
        """Tests that visitor is called on each element of the AST."""
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        ast = parse(qasm_program, permissive=True)
        spy = mocker.spy(QasmInterpreter, "visit")
        QasmInterpreter(permissive=True).generic_visit(ast, context={"name": program_name})
        assert spy.call_count == count_nodes

    def test_variables(self, mocker):
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(open("variables.qasm", mode="r").read(), permissive=True)

        # run the program
        context, execution_context = QasmInterpreter(permissive=True).generic_visit(
            ast, context={"name": "advanced-vars"}
        )
        context["callable"]()

        # static vars are available in the compilation context
        assert context["vars"]["f"]["val"] == 3.2
        assert context["vars"]["g"]["val"] == 3
        assert context["vars"]["h"]["val"] == 2
        assert context["vars"]["k"]["val"] == 3

        # classical logic is represented in the execution context after execution
        assert execution_context["vars"]["l"]["val"] == True
        assert execution_context["vars"]["m"]["val"] == (3.14159 / 2) * 3.3
        assert execution_context["vars"]["a"]["val"] == 3.3333333

    def test_updating_constant(self, mocker):
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(
            """
            const int i = 2;
            i = 3;
            """,
            permissive=True,
        )

        with pytest.raises(
            ValueError,
            match=f"Attempt to mutate a constant i on line 3 that was defined on line 2",
        ):
            QasmInterpreter().generic_visit(ast, context={"name": "mutate-error"})

    def test_classical_variables(self, mocker):
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(open("classical.qasm", mode="r").read(), permissive=True)

        # run the program
        context, _ = QasmInterpreter(permissive=True).generic_visit(
            ast, context={"name": "basic-vars"}
        )
        context["callable"]()

        assert context["vars"]["i"]["val"] == 4
        assert context["vars"]["j"]["val"] == 4
        assert context["vars"]["c"]["val"] == 0

    def test_updating_variables(self, mocker):
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(
            open("updating_variables.qasm", mode="r").read(),
            permissive=True,
        )

        # setup mocks
        x = mocker.spy(PauliX, "__init__")
        y = mocker.spy(PauliY, "__init__")
        z = mocker.spy(PauliZ, "__init__")
        rx = mocker.spy(RX, "__init__")

        # run the program
        context, _ = QasmInterpreter(permissive=True).generic_visit(
            ast, context={"name": "updating-vars"}
        )
        context["callable"]()

        # assertions
        assert x.call_count == 1
        assert y.call_count == 2
        assert z.call_count == 0
        assert rx.call_count == 1

    def test_loops(self, mocker):
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(open("loops.qasm", mode="r").read(), permissive=True)

        # setup mocks
        x = mocker.spy(PauliX, "__init__")
        y = mocker.spy(PauliY, "__init__")
        z = mocker.spy(PauliZ, "__init__")
        ry = mocker.spy(RY, "__init__")
        rx = mocker.spy(RX, "__init__")
        rz = mocker.spy(RZ, "__init__")

        # run the program
        context, _ = QasmInterpreter(permissive=True).generic_visit(ast, context={"name": "loops"})
        context["callable"]()

        # assertions
        assert x.call_count == 10
        assert rx.call_count == 4294967306 - 4294967296
        assert ry.call_count == 4
        assert y.call_count == 6
        assert z.call_count == 6
        assert rz.call_count == 6

        for i in range(4294967296, 4294967306):
            rx.assert_called_with(RX(i, 0), i, 0)

        for f in [1.2, -3.4, 0.5, 9.8]:
            ry.assert_called_with(RY(f, 0), f, 0)

        rz.assert_called_with(RZ(0.1, 0), 0.1, 0)

    def test_switch(self, mocker):
        from openqasm3.parser import parse

        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(open("switch.qasm", mode="r").read(), permissive=True)

        # setup mocks
        x = mocker.spy(PauliX, "__init__")
        y = mocker.spy(PauliY, "__init__")
        z = mocker.spy(PauliZ, "__init__")
        rx = mocker.spy(RX, "__init__")

        # run the program
        context, _ = QasmInterpreter(permissive=True).generic_visit(ast, context={"name": "switch"})
        context["callable"]()

        # assertions
        assert x.call_count == 1
        assert y.call_count == 1
        assert rx.call_count == 1
        assert z.call_count == 0

    def test_if_else(self, mocker):
        from openqasm3.parser import parse

        from pennylane import ops
        from pennylane.io.qasm_interpreter import QasmInterpreter

        # parse the QASM
        ast = parse(open("if_else.qasm", mode="r").read(), permissive=True)

        # setup mocks
        cond = mocker.spy(ops, "cond")
        x = mocker.spy(PauliX, "__init__")
        y = mocker.spy(PauliY, "__init__")
        z = mocker.spy(PauliZ, "__init__")

        # run the program
        context, _ = QasmInterpreter(permissive=True).generic_visit(
            ast, context={"name": "if_else"}
        )
        context["callable"]()

        # assertions
        assert cond.call_count == 3
        assert x.call_count == 1
        x.assert_called_with(PauliX(Wires(["q0"])), Wires(["q0"]))
        assert y.call_count == 1
        y.assert_called_with(PauliY(Wires(["q0"])), Wires(["q0"]))
        assert z.call_count == 0

    def test_mod_with_declared_param(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            int power = 2;
            float phi = 0.2;
            qubit q0;
            pow(power) @ rx(phi) q0;
            """,
            permissive=True,
        )

        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "parameterized-gate"})

        # setup mocks
        rx = mocker.spy(RX, "__init__")

        # execute the callable
        context["callable"]()

        assert rx.call_count == 1  # RX calls PauliX under the hood
        rx.assert_called_with(RX(2, 0), 2, wires=0)

    def test_uninitialized_param(self):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            rx(phi) q0;
            """,
            permissive=True,
        )

        with pytest.raises(
            NameError,
            match="Uninitialized variable encountered in QASM.",
        ):
            QasmInterpreter().generic_visit(ast, context={"name": "name-error"})

    def test_unsupported_node_type_raises(self):

        # parse the QASM program
        ast = parse(
            """
            include "stdgates.inc";
            """,
            permissive=True,
        )

        with pytest.raises(
            NotImplementedError,
            match="An unsupported QASM instruction Include was encountered on line 2, in unsupported-error.",
        ):
            QasmInterpreter().generic_visit(ast, context={"name": "unsupported-error"})

    def test_no_qubits(self):

        # parse the QASM program
        ast = parse(
            """
            float theta;
            rx(theta) q0;
            """,
            permissive=True,
        )

        with pytest.raises(
            NameError,
            match="Attempt to reference wires that have not been declared in uninit-qubit",
        ):
            QasmInterpreter(permissive=False).generic_visit(ast, context={"name": "uninit-qubit"})

    def test_unsupported_gate(self):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            float theta = 0.5;
            Rxx(theta) q0, q1;
            """,
            permissive=True,
        )

        with pytest.raises(NotImplementedError, match="Unsupported gate encountered in QASM: Rxx"):
            QasmInterpreter(permissive=False).generic_visit(
                ast, context={"name": "unsupported-gate"}
            )

    def test_missing_param(self):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            float theta;
            rx() q0;
            """,
            permissive=True,
        )

        with pytest.raises(
            TypeError, match=r"Missing required argument\(s\) for parameterized gate rx"
        ):
            QasmInterpreter(permissive=False).generic_visit(ast, context={"name": "missing-param"})

    def test_uninitialized_var(self):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            float theta;
            rx(theta) q0;
            """,
            permissive=True,
        )

        with pytest.raises(NameError, match="Attempt to reference uninitialized parameter theta!"):
            QasmInterpreter(permissive=False).generic_visit(ast, context={"name": "uninit-param"})

    def test_parses_simple_qasm(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            float theta = 0.5;
            x q0;
            cx q0, q1;
            rx(theta) q0;
            ry(0.2) q0;
            inv @ rx(theta) q0;
            pow(2) @ x q0;
            ctrl @ x q1, q0;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "gates"})

        # setup mocks
        x = mocker.spy(PauliX, "__init__")
        cx = mocker.spy(CNOT, "__init__")
        rx = mocker.spy(RX, "__init__")
        ry = mocker.spy(RY, "__init__")

        # execute the callable
        context["callable"]()

        # asserts
        assert len(context["wires"]) == 2

        assert x.call_count == 5  # RX calls PauliX under the hood
        x.assert_called_with(PauliX(0), wires=0)

        assert cx.call_count == 2  # verifies that ctrl @ x q1, q0 calls cx too
        cx.assert_called_with(CNOT([0, 1]), wires=[0, 1])
        cx.assert_called_with(CNOT([1, 0]), wires=[1, 0])

        assert rx.call_count == 2  # one adjoint call and one direct call
        rx.assert_called_with(RX(0.5, 0), 0.5, wires=0)

        assert ry.call_count == 1
        ry.assert_called_with(RY(0.2, [0]), 0.2, wires=[0])

    def test_interprets_two_qubit_gates(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            ch q0, q1;
            cx q1, q0;
            cy q0, q1;
            cz q1, q0;
            swap q0, q1;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "two-qubit-gates"})

        # setup mocks

        # two qubit gate ops
        ch = mocker.spy(CH, "__init__")
        cx = mocker.spy(CNOT, "__init__")
        cy = mocker.spy(CY, "__init__")
        cz = mocker.spy(CZ, "__init__")
        swap = mocker.spy(SWAP, "__init__")

        # execute the callable
        context["callable"]()

        assert ch.call_count == 1
        ch.assert_called_with(CH([0, 1]), wires=[0, 1])

        assert cx.call_count == 1
        cx.assert_called_with(CNOT([1, 0]), wires=[1, 0])

        assert cy.call_count == 1
        cy.assert_called_with(CY([0, 1]), wires=[0, 1])

        assert cz.call_count == 1
        cz.assert_called_with(CZ([1, 0]), wires=[1, 0])

        assert swap.call_count == 1
        swap.assert_called_with(SWAP([0, 1]), [0, 1])

    def test_interprets_parameterized_two_qubit_gates(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            cp(0.4) q0, q1;
            cphase(0.4) q0, q1;
            crx(0.2) q0, q1;
            cry(0.1) q0, q1;
            crz(0.3) q1, q0;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "param-two-qubit-gates"})

        # setup mocks

        # parameterized two qubit gate ops
        cp = mocker.spy(CPhase, "__init__")
        crx = mocker.spy(CRX, "__init__")
        cry = mocker.spy(CRY, "__init__")
        crz = mocker.spy(CRZ, "__init__")

        # execute the callable
        context["callable"]()

        assert cp.call_count == 2
        cp.assert_called_with(CPhase(0.4, [0, 1]), 0.4, wires=[0, 1])

        assert crx.call_count == 1
        crx.assert_called_with(CRX(0.2, [0, 1]), 0.2, wires=[0, 1])

        assert cry.call_count == 1
        cry.assert_called_with(CRY(0.1, [0, 1]), 0.1, wires=[0, 1])

        assert crz.call_count == 1
        crz.assert_called_with(CRZ(0.1, [0, 1]), 0.1, wires=[0, 1])

    def test_interprets_multi_qubit_gates(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            qubit[1] q2;
            ccx q0, q2, q1;
            cswap q1, q2, q0;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "multi-qubit-gates"})

        # setup mocks

        # multi qubit gate ops
        ccx = mocker.spy(Toffoli, "__init__")
        cswap = mocker.spy(CSWAP, "__init__")

        # execute the callable
        context["callable"]()

        assert ccx.call_count == 1
        ccx.assert_called_with(Toffoli([0, 1, 2]), wires=[0, 1, 2])

        assert cswap.call_count == 1
        cswap.assert_called_with(CSWAP([1, 2, 0]), wires=[1, 2, 0])

    def test_interprets_parameterized_single_qubit_gates(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            qubit q2;
            rx(0.9) q0;
            ry(0.8) q1;
            rz(1.1) q2;
            p(8) q0;
            phase(2.0) q1;
            u1(3.3) q0;
            u2(1.0, 2.0) q1;
            u3(1.0, 2.0, 3.0) q2;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(
            ast, context={"name": "param-single-qubit-gates"}
        )

        # setup mocks

        # parameterized single qubit gate ops
        rx = mocker.spy(RX, "__init__")
        ry = mocker.spy(RY, "__init__")
        rz = mocker.spy(RZ, "__init__")
        p = mocker.spy(PhaseShift, "__init__")
        u1 = mocker.spy(U1, "__init__")
        u2 = mocker.spy(U2, "__init__")
        u3 = mocker.spy(U3, "__init__")

        # execute the callable
        context["callable"]()

        assert rx.call_count == 1
        rx.assert_called_with(RX(0.9, [0]), 0.9, wires=[0])

        assert ry.call_count == 1
        ry.assert_called_with(RY(0.8, [0]), 0.8, wires=[0])

        assert rz.call_count == 1
        rz.assert_called_with(RZ(1.1, [2]), 1.1, wires=[2])

        assert p.call_count == 2
        p.assert_called_with(PhaseShift(8, 0), 8, wires=0)
        p.assert_called_with(PhaseShift(2.0, 1), 2.0, wires=1)

        assert u1.call_count == 1
        u1.assert_called_with(U1(3.3, 0), 3.3, wires=0)

        assert u2.call_count == 1
        u2.assert_called_with(U2(1.0, 2.0, 1), 1.0, 2.0, wires=1)

        assert u3.call_count == 1
        u3.assert_called_with(U3(1.0, 2.0, 3.0, 2), 1.0, 2.0, 3.0, wires=2)

    def test_single_qubit_gates(self, mocker):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            qubit q2;
            id q0;
            h q2;
            x q1;
            y q2;
            z q0;
            s q2;
            t q1;
            sx q0;
            ctrl @ id q0, q1;
            inv @ h q2;
            pow(2) @ t q1;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "single-qubit-gates"})

        # setup mocks

        # single qubit gate ops
        id = mocker.spy(Identity, "__init__")
        h = mocker.spy(Hadamard, "__init__")
        x = mocker.spy(PauliX, "__init__")
        y = mocker.spy(PauliY, "__init__")
        z = mocker.spy(PauliZ, "__init__")
        s = mocker.spy(S, "__init__")
        t = mocker.spy(T, "__init__")
        sx = mocker.spy(SX, "__init__")

        # execute the callable
        context["callable"]()

        assert id.call_count == 2
        id.assert_called_with(Identity(0), wires=0)
        id.assert_called_with(Identity(1), wires=1)

        assert h.call_count == 2
        h.assert_called_with(Hadamard(2), wires=2)

        assert x.call_count == 1
        x.assert_called_with(PauliX(1), wires=1)

        assert y.call_count == 1
        y.assert_called_with(PauliY(2), wires=2)

        assert z.call_count == 1
        z.assert_called_with(PauliZ(0), wires=0)

        assert s.call_count == 1
        s.assert_called_with(S(2), 2)

        assert t.call_count == 2
        t.assert_called_with(T(1), 1)

        assert sx.call_count == 1
        sx.assert_called_with(SX(1), 1)
