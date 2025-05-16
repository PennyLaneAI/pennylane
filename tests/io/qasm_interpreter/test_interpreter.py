import pytest
from openqasm3.parser import parse

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
from pennylane.io.qasm_interpreter import QasmInterpreter


class TestInterpreter:

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
        context = QasmInterpreter().generic_visit(ast, context={"program_name": "two-qubit-gates"})

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
        context = QasmInterpreter().generic_visit(
            ast, context={"program_name": "param-two-qubit-gates"}
        )

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
        context = QasmInterpreter().generic_visit(
            ast, context={"program_name": "multi-qubit-gates"}
        )

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
        context = QasmInterpreter().generic_visit(
            ast, context={"program_name": "param-single-qubit-gates"}
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
        context = QasmInterpreter().generic_visit(
            ast, context={"program_name": "single-qubit-gates"}
        )

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
