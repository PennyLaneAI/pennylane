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
    device,
    queuing, measure,
)
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.ops import Adjoint, Controlled, ControlledPhaseShift, MultiControlledX
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

    def test_subroutines(self):
        # parse the QASM
        ast = parse(open("subroutines.qasm", mode="r").read(), permissive=True)

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter(permissive=True).interpret(
                ast, context={"name": "subroutines", "wire_map": None}
            )

        assert q.queue[0] == Hadamard('q0')
        assert isinstance(q.queue[1], MidMeasureMP)

    def test_param_as_expression(self):
        # parse the QASM
        ast = parse(
            """
            qubit q0;
            int p = 1;
            pow(p * 2) @ x q0;
            """,
            permissive=True,
        )

        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "expression-implemented"}
            )

        assert q.queue == [PauliX('q0')**2]

    def test_nested_modifiers(self):
        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            qubit q2;
            inv @ negctrl @ x q0, q1;
            ctrl @ ctrl @ x q2, q1, q0;
            inv @ ctrl @ x q0, q1;
            pow(2) @ ctrl @ x q1, q0;
            pow(2) @ inv @ y q0;
            inv @ pow(2) @ ctrl @ y q0, q1;
            """,
            permissive=True,
        )
        # execute
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "nested-modifiers"})

        assert q.queue == [
            Adjoint(MultiControlledX(wires=["q0", "q1"], control_values=[False])),
            Toffoli(wires=["q2", "q1", "q0"]),
            Adjoint(CNOT(wires=["q0", "q1"])),
            (CNOT(wires=["q1", "q0"])) ** 2,
            (Adjoint(PauliY("q0"))) ** 2,
            Adjoint((CY(wires=["q0", "q1"])) ** 2),
        ]

    def test_variables(self, mocker):
        # parse the QASM
        ast = parse(open("variables.qasm", mode="r").read(), permissive=True)

        # run the program
        context = QasmInterpreter(permissive=True).interpret(
            ast, context={"wire_map": None, "name": "advanced-vars"}
        )

        # static vars
        assert context["vars"]["f"]["val"] == 3.2
        assert context["vars"]["g"]["val"] == 3
        assert context["vars"]["h"]["val"] == 2
        assert context["vars"]["k"]["val"] == 3

        # dynamic vars
        assert context["vars"]["l"]["val"] == True
        assert context["vars"]["m"]["val"] == (3.14159 / 2) * 3.3
        assert context["vars"]["a"]["val"] == 3.3333333

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
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "mutate-error"})

    def test_integer_wire_maps(self):
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
            """,
            permissive=True,
        )

        # we would initialize the device like so
        device("default.qubit", wires=[0, 1, 2])

        # execute
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast,
                context={
                    "name": "single-qubit-gates",
                    "wire_map": {"q0": 0, "q1": 1, "q2": 2},
                },
            )

        assert q.queue == [Identity(0), Hadamard(2), PauliX(1), PauliY(2)]

    def test_wire_maps(self):
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
            """,
            permissive=True,
        )

        # we would initialize the device like so
        device("default.qubit", wires=["0q", "1q", "2q"])

        # execute
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast,
                context={
                    "name": "single-qubit-gates",
                    "wire_map": {"q0": "0q", "q1": "1q", "q2": "2q"},
                },
            )

        assert q.queue == [Identity("0q"), Hadamard("2q"), PauliX("1q"), PauliY("2q")]

    def test_classical_variables(self):
        # parse the QASM
        ast = parse(open("classical.qasm", mode="r").read(), permissive=True)

        # run the program
        context = QasmInterpreter(permissive=True).interpret(
            ast, context={"wire_map": None, "name": "basic-vars"}
        )

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
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter(permissive=True).interpret(
                ast, context={"name": "updating-vars", "wire_map": None}
            )

        assert q.queue == [
            PauliX('q0'),
            PauliY('q0'),
            RX(0.1, 'q0'),
            PauliY('q0'),
        ]

    def test_loops(self, mocker):

        # parse the QASM
        ast = parse(open("loops.qasm", mode="r").read(), permissive=True)

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter(permissive=True).interpret(ast, context={"name": "loops", "wire_map": None})

        assert q.queue == [PauliZ('q0')] + \
            [PauliX('q0') for _ in range(10)] + \
            [RX(phi, wires=['q0']) for phi in range(4294967296, 4294967306)] + \
            [RY(phi, wires=['q0']) for phi in [1.2, -3.4, 0.5, 9.8]] + \
            [RZ(0.1, wires=['q0']) for _ in range(6)] + \
            [PauliY('q0') for _ in range(6)] + \
            [PauliZ('q0') for _ in range(5)]

    def test_switch(self, mocker):

        # parse the QASM
        ast = parse(open("switch.qasm", mode="r").read(), permissive=True)

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter(permissive=True).interpret(ast, context={"name": "switch", "wire_map": None})

        assert q.queue == [
            PauliX('q0'),
            PauliY('q0'),
            RX(0.1, wires=['q0'])
        ]

    def test_if_else(self, mocker):
        from pennylane import ops

        # parse the QASM
        ast = parse(open("if_else.qasm", mode="r").read(), permissive=True)

        # setup mocks
        cond = mocker.spy(ops, "cond")

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter(permissive=True).interpret(
                ast, context={"name": "if_else", "wire_map": None}
            )

        # assertions
        assert cond.call_count == 3

        assert q.queue == [
            PauliX('q0'),
            PauliY('q0')
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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "parameterized-gate"}
            )

        assert q.queue[0] == PowOperation(RX(0.2, wires=["q0"]), 2)

    def test_uninitialized_param(self):

        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            float phi;
            rx(phi) q0;
            """,
            permissive=True,
        )

        with pytest.raises(
            NameError,
            match="Attempt to reference uninitialized parameter phi!",
        ):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "name-error"})

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
            match="An unsupported QASM instruction Include was encountered "
            "on line 2, in unsupported-error.",
        ):
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "unsupported-error"}
            )

    def test_no_qubits(self):

        # parse the QASM program
        ast = parse(
            """
            float theta = 0.1;
            rx(theta) q0;
            """,
            permissive=True,
        )

        with pytest.raises(
            NameError,
            match=r"Attempt to reference wire\(s\): \['q0'\] that have not been declared in uninit-qubit",
        ):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "uninit-qubit"})

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
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "unsupported-gate"})

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
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "missing-param"})

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
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "uninit-param"})

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "gates"})

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "two-qubit-gates"})

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "param-two-qubit-gates"}
            )

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "multi-qubit-gates"}
            )

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

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "param-single-qubit-gates"}
            )

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

        # setup mocks

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "single-qubit-gates"}
            )

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
