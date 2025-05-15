from openqasm3.parser import parse

from pennylane import CNOT, RX, RY, PauliX, CH, CY, CZ, SWAP, CPhase, CRX, CRY, CRZ, Toffoli, CSWAP, RZ, PhaseShift, U1, \
    U2, U3, Identity, Hadamard, PauliZ, PauliY, S, SX, T
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

    def test_parses_the_gate_library(self, mocker):
        # TODO: break this up into multiple tests

        # parse the QASM program
        ast = parse(open("gate_library.qasm", mode="r").read(), permissive=True)
        context = QasmInterpreter().generic_visit(ast, context={"program_name": "gate-lib"})

        # setup mocks

        # two qubit gate ops
        ch = mocker.spy(CH, "__init__")
        cx = mocker.spy(CNOT, "__init__")
        cy = mocker.spy(CY, "__init__")
        cz = mocker.spy(CZ, "__init__")
        swap = mocker.spy(SWAP, "__init__")

        # parameterized two qubit gate ops
        cp = mocker.spy(CPhase, "__init__")
        crx = mocker.spy(CRX, "__init__")
        cry = mocker.spy(CRY, "__init__")
        crz = mocker.spy(CRZ, "__init__")

        # multi qubit gate ops
        ccx = mocker.spy(Toffoli, "__init__")
        cswap = mocker.spy(CSWAP, "__init__")

        # parameterized single qubit gate ops
        rx = mocker.spy(RX, "__init__")
        ry = mocker.spy(RY, "__init__")
        rz = mocker.spy(RZ, "__init__")
        p = mocker.spy(PhaseShift, "__init__")
        u1 = mocker.spy(U1, "__init__")
        u2 = mocker.spy(U2, "__init__")
        u3 = mocker.spy(U3, "__init__")

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

        assert ch.call_count == 1
        ch.assert_called_with(CH([0, 1]), wires=[0, 1])

        assert cx.call_count == 1
        cx.assert_called_with(CNOT([1, 0]), wires=[1, 0])

        assert cy.call_count == 1
        cy.assert_called_with(CY([0, 1]), wires=[0, 1])

        assert cz.call_count == 1
        cz.assert_called_with(CZ([1, 0]), wires=[1, 0])

        assert swap.call_count == 2

        assert cp.call_count == 2
        cp.assert_called_with(CPhase(0.4, [0, 1]), 0.4, wires=[0, 1])

        assert crx.call_count == 1
        crx.assert_called_with(CRX(0.2, [0, 1]), 0.2, wires=[0, 1])

        assert cry.call_count == 1
        cry.assert_called_with(CRY(0.1, [0, 1]), 0.1, wires=[0, 1])

        assert crz.call_count == 1
        crz.assert_called_with(CRX(0.1, [0, 1]), 0.1, wires=[0, 1])

        assert ccx.call_count == 1
        ccx.assert_called_with(Toffoli([0, 1, 2]), wires=[0, 1, 2])

        assert cswap.call_count == 1
        cswap.assert_called_with(CSWAP([1, 2, 0]), wires=[1, 2, 0])
