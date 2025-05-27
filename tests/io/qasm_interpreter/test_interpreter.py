"""
Unit tests for the :mod:`pennylane.io.qasm_interpreter` module.
"""

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
    Hadamard,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    S,
    T,
    Toffoli,
    queuing,
)
from pennylane.ops import Adjoint, Controlled, ControlledPhaseShift
from pennylane.ops.op_math.pow import PowOperation, PowOpObs
from pennylane.wires import Wires

try:
    pytest.importorskip("openqasm3")

    from openqasm3.parser import parse

    from pennylane.io.qasm_interpreter import QasmInterpreter  # pylint: disable=ungrouped-imports
except (ModuleNotFoundError, ImportError) as import_error:
    pass


@pytest.mark.external
class TestInterpreter:

    def test_subroutines(self, mocker):
        # parse the QASM
        ast = parse(open("subroutines.qasm", mode="r").read(), permissive=True)

        # run the program
        context, execution_context = QasmInterpreter(permissive=True).generic_visit(
            ast, context={"name": "subroutines"}
        )

        spy = mocker.spy(Hadamard, "__init__")

        # execute the callable
        context["callable"]()

        spy.assert_called_with(Hadamard('q0'), 'q0')

    def test_variables(self):
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

    def test_updating_constant(self):
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

    def test_classical_variables(self):
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

    def test_updating_variables(self):
        # parse the QASM
        ast = parse(
            open("updating_variables.qasm", mode="r").read(),
            permissive=True,
        )

        # run the program
        context, _ = QasmInterpreter(permissive=True).generic_visit(
            ast, context={"name": "updating-vars"}
        )

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue == [
            RX(1, "q0"),
            RX(2, "q0"),
            RX(0, "q0"),
            RX(2, "q0"),
        ]

    def test_mod_with_declared_param(self):

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue[0] == PowOperation(RX(0.2, wires=["q0"]), 2)

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
            TypeError,
            match="Attempt to use unevaluated variable phi in name-error, last updated on line unknown.",
        ):
            QasmInterpreter().generic_visit(ast, context={"name": "name-error"})

    def test_unsupported_node_type_raises(self):

        # parse the QASM program
        ast = parse(
            """
            bit b;
            qubit q0;
            float theta = 0.2;
            rx(theta) q0;
            measure q0 -> b;
            """,
            permissive=True,
        )

        with pytest.raises(
            NotImplementedError,
            match="An unsupported QASM instruction QuantumMeasurementStatement was "
            "encountered on line 6, in unsupported-error.",
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
            QasmInterpreter().generic_visit(ast, context={"name": "uninit-qubit"})

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
            QasmInterpreter().generic_visit(ast, context={"name": "unsupported-gate"})

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
            QasmInterpreter().generic_visit(ast, context={"name": "missing-param"})

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
            QasmInterpreter().generic_visit(ast, context={"name": "uninit-param"})

    def test_parses_simple_qasm(self):

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
            ctrl @ id q0, q1;
            """,
            permissive=True,
        )
        context, _ = QasmInterpreter().generic_visit(ast, context={"name": "gates"})

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

            # asserts
            assert len(context["wires"]) == 2

        assert q.queue == [
            PauliX("q0"),
            CNOT(wires=["q0", "q1"]),
            RX(0.5, wires=["q0"]),
            RY(0.2, wires=["q0"]),
            Adjoint(RX(0.5, wires=["q0"])),
            PowOpObs(PauliX(wires=["q0"]), 2),
            Controlled(Identity("q1"), control_wires=["q0"]),
        ]

    def test_interprets_two_qubit_gates(self):

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue == [
            CH(wires=["q0", "q1"]),
            CNOT(wires=["q1", "q0"]),
            CY(wires=["q0", "q1"]),
            CZ(wires=["q1", "q0"]),
            SWAP(wires=["q0", "q1"]),
        ]

    def test_interprets_parameterized_two_qubit_gates(self):

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue == [
            ControlledPhaseShift(0.4, wires=Wires(["q0", "q1"])),
            ControlledPhaseShift(0.4, wires=Wires(["q0", "q1"])),
            CRX(0.2, wires=["q0", "q1"]),
            CRY(0.1, wires=["q0", "q1"]),
            CRZ(0.3, wires=Wires(["q1", "q0"])),
        ]

    def test_interprets_multi_qubit_gates(self):

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue == [Toffoli(wires=["q0", "q2", "q1"]), CSWAP(wires=["q1", "q2", "q0"])]

    def test_interprets_parameterized_single_qubit_gates(self):

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue == [
            RX(0.9, wires=["q0"]),
            RY(0.8, wires=["q1"]),
            RZ(1.1, wires=["q2"]),
            PhaseShift(8, wires=["q0"]),
            PhaseShift(2.0, wires=["q1"]),
            U1(3.3, wires=["q0"]),
            U2(1.0, 2.0, wires=["q1"]),
            U3(1.0, 2.0, 3.0, wires=["q2"]),
        ]

    def test_single_qubit_gates(self):

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            context["callable"]()

        assert q.queue == [
            Identity("q0"),
            Hadamard("q2"),
            PauliX("q1"),
            PauliY("q2"),
            PauliZ("q0"),
            S("q2"),
            T("q1"),
            SX("q0"),
            Controlled(Identity("q1"), control_wires=["q0"]),
            Adjoint(Hadamard("q2")),
            T("q1") ** 2,
        ]
