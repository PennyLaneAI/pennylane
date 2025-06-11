"""
Unit tests for the :mod:`pennylane.io.qasm_interpreter` module.
"""

from re import escape
from unittest.mock import MagicMock, call

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
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.ops import Adjoint, Controlled, ControlledPhaseShift, MultiControlledX
from pennylane.ops.op_math.pow import PowOperation, PowOpObs
from pennylane.wires import Wires

try:
    pytest.importorskip("openqasm3")

    from openqasm3.parser import parse

    from pennylane.io.qasm_interpreter import (  # pylint: disable=ungrouped-imports
        Context,
        QasmInterpreter,
        Variable,
        preprocess_operands,
    )
except (ModuleNotFoundError, ImportError) as import_error:
    pass


@pytest.mark.external
class TestIO:

    def test_missing_input(self):
        ast = parse(
            """
            input float theta;
            qubit q;
            rx(theta) q;
            """
        )

        with pytest.raises(
            ValueError,
            match="Missing input theta. Please pass theta as " "a keyword argument to from_qasm3.",
        ):
            QasmInterpreter().interpret(ast, context={"name": "missing-input", "wire_map": None})

    def test_input(self):
        ast = parse(
            """
            input float theta;
            qubit q;
            rx(theta) q;
            """
        )

        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"name": "inputs", "wire_map": None}, theta=0.1
            )
            QasmInterpreter().interpret(
                ast, context={"name": "inputs", "wire_map": None}, theta=0.2
            )

        assert q.queue == [
            RX(0.1, "q"),
            RX(0.2, "q"),
        ]


@pytest.mark.external
class TestMeasurementReset:

    def test_resets(self):
        # parse the QASM
        ast = parse(open("tests/io/qasm_interpreter/resets.qasm", mode="r").read(), permissive=True)

        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"name": "post_processing", "wire_map": None})

        assert isinstance(q.queue[0], MidMeasureMP)
        assert q.queue[0].wires == Wires(["qubits"])
        assert q.queue[0].reset == True

    def test_post_processing_measurement(self, mocker):
        import pennylane

        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/post_processing.qasm", mode="r").read(), permissive=True
        )

        # setup mocks
        eval_binary = mocker.spy(pennylane.io.qasm_interpreter, "_eval_binary_op")

        mock_one = MagicMock(return_value=1)
        mock_zero = MagicMock(return_value=0)

        vars = {
            "c": Variable(
                "MeasurementValue", MeasurementValue([PauliX(0)], mock_one), -1, 0, False
            ),
            "d": Variable(
                "MeasurementValue", MeasurementValue([PauliX(0)], mock_zero), -1, 0, False
            ),
        }

        # run the program
        context = QasmInterpreter().interpret(
            ast, context={"name": "post_processing", "wire_map": None, "vars": vars}
        )

        # ops on MeasurementValues should yield MeasurementValues
        assert isinstance(context["vars"]["c"].val, MeasurementValue)
        assert isinstance(context["vars"]["d"].val, MeasurementValue)

        # the right ops should have been called

        # Note: we can't compare MeasurementValues using __eq__ as they don't have a truthiness,
        # __eq__ returns a MeasurementValue
        def compare_measurement_values(mv_1: MeasurementValue[T], mv_2: MeasurementValue[T]):
            res_1 = mv_1.processing_fn()
            res_2 = mv_2.processing_fn()
            meas_1 = mv_1.measurements
            meas_2 = mv_2.measurements
            return res_1 == res_2 and meas_1 == meas_2

        # first call
        call = 0
        # c = c + 1;
        assert compare_measurement_values(
            MeasurementValue([PauliX(0)], mock_one), eval_binary.call_args_list[call].args[0]
        )
        assert eval_binary.call_args_list[call].args[1:] == ("+", 1, 4)

        # second call
        call += 1
        # c = d / c;
        assert compare_measurement_values(
            MeasurementValue([PauliX(0)], mock_zero), eval_binary.call_args_list[call].args[0]
        )
        assert eval_binary.call_args_list[call].args[1] == "/"
        assert compare_measurement_values(
            MeasurementValue([PauliX(0)], (lambda: mock_one() + 1)),
            eval_binary.call_args_list[call].args[2],
        )
        assert eval_binary.call_args_list[call].args[3] == 5

    def test_measurement(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/measurements.qasm", mode="r").read(), permissive=True
        )

        # run the program
        context = QasmInterpreter().interpret(
            ast, context={"name": "measurements", "wire_map": None}
        )

        assert isinstance(context["vars"]["c"].val, MeasurementValue)
        assert isinstance(context["vars"]["d"].val, MeasurementValue)


@pytest.mark.external
class TestControlFlow:

    def test_nested_end(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/nested_end.qasm", mode="r").read(), permissive=True
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"name": "nested-end", "wire_map": None})

        assert q.queue == [PauliX("q0")]

    def test_loops(self, mocker):

        # parse the QASM
        ast = parse(open("tests/io/qasm_interpreter/loops.qasm", mode="r").read(), permissive=True)

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"name": "loops", "wire_map": None})

        assert q.queue == [PauliZ("q0")] + [PauliX("q0") for _ in range(10)] + [
            RX(phi, wires=["q0"]) for phi in range(4294967296, 4294967306)
        ] + [RY(phi, wires=["q0"]) for phi in [1.2, -3.4, 0.5, 9.8]] + [
            RZ(0.1, wires=["q0"]) for _ in range(6)
        ] + [
            PauliY("q0") for _ in range(6)
        ] + [
            PauliZ("q0") for _ in range(5)
        ]

    def test_switch(self, mocker):

        # parse the QASM
        ast = parse(open("tests/io/qasm_interpreter/switch.qasm", mode="r").read(), permissive=True)

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"name": "switch", "wire_map": None})

        assert q.queue == [PauliX("q0"), PauliY("q0"), RX(0.1, wires=["q0"])]

    def test_if_else(self, mocker):
        from pennylane import ops

        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/if_else.qasm", mode="r").read(), permissive=True
        )

        # setup mocks
        cond = mocker.spy(ops, "cond")

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"name": "if_else", "wire_map": None})

        # assertions
        assert cond.call_count == 3

        assert q.queue == [PauliX("q0"), PauliY("q0")]

    def test_end_statement(self):
        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            qubit q1;
            ch q0, q1;
            cx q1, q0;
            end;
            cy q0, q1;
            cz q1, q0;
            swap q0, q1;
            """,
            permissive=True,
        )

        # execute the callable
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "end-early"})

        assert q.queue == [
            CH(wires=["q0", "q1"]),
            CNOT(wires=["q1", "q0"]),
        ]


@pytest.mark.external
class TestSubroutine:

    def test_repeated_calls(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/repeated_calls.qasm", mode="r").read(),
            permissive=True,
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"name": "repeated-subroutines", "wire_map": None}
            )

        assert q.queue == [
            RX(2, "q0"),
            RY(0.5, "q0"),
            RX(2, "q0"),
            RY(0.5, "q0"),
            RX(11, "q0"),
            RY(0.5, "q0"),
        ]

    def test_subroutine_not_defined(self):
        ast = parse(
            """
            undefined_subroutine();
            """,
            permissive=True,
        )

        with pytest.raises(
            NameError,
            match="Reference to subroutine undefined_subroutine not "
            "available in calling namespace on line 2.",
        ):
            QasmInterpreter().interpret(
                ast, context={"name": "undefined-subroutine", "wire_map": None}
            )

    def test_stand_alone_call_of_subroutine(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/standalone_subroutines.qasm", mode="r").read(),
            permissive=True,
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"name": "standalone-subroutines", "wire_map": None}
            )

        assert q.queue == [Hadamard("q0"), PauliY("q0")]

    def test_complex_subroutines(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/complex_subroutines.qasm", mode="r").read(),
            permissive=True,
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            context = QasmInterpreter().interpret(
                ast, context={"name": "complex-subroutines", "wire_map": None}
            )

        assert q.queue == [Hadamard("q0"), PauliY("q0"), Hadamard("q0"), RX(0.1, "q0")]
        assert context.vars["c"].val == 0

    def test_subroutines(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/subroutines.qasm", mode="r").read(), permissive=True
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"name": "subroutines", "wire_map": None})

        assert q.queue[0] == Hadamard("q0")
        assert isinstance(q.queue[1], MidMeasureMP)


@pytest.mark.external
class TestExpressions:

    def test_different_unary_exprs(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/unary_expressions.qasm", mode="r").read(),
            permissive=True,
        )

        context = QasmInterpreter().interpret(
            ast, context={"wire_map": None, "name": "unary-exprs"}
        )

        # unary expressions
        assert context.vars["a"].val == -1
        assert context.vars["b"].val == ~2
        assert context.vars["c"].val == (not False)

    def test_different_binary_exprs(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/binary_expressions.qasm", mode="r").read(),
            permissive=True,
        )

        context = QasmInterpreter().interpret(
            ast, context={"wire_map": None, "name": "binary-exprs"}
        )

        # comparison expressions
        assert context.vars["a"].val == (0 == 1)
        assert context.vars["b"].val == (0 != 1)
        assert context.vars["c"].val == (0 > 1)
        assert context.vars["d"].val == (0 < 1)
        assert context.vars["e"].val == (0 >= 1)
        assert context.vars["f"].val == (0 <= 1)

        # bitwise operations
        assert context.vars["g"].val == 2 >> 1
        assert context.vars["h"].val == 2 << 1
        assert context.vars["i"].val == 2 | 1
        assert context.vars["j"].val == 2 ^ 1
        assert context.vars["k"].val == 2 & 1

        # arithmetic operators
        assert context.vars["m"].val == 3 + 2
        assert context.vars["o"].val == 3 - 2
        assert context.vars["p"].val == 3 * 2
        assert context.vars["q"].val == 3**2
        assert context.vars["n"].val == 3 / 2
        assert context.vars["s"].val == 3 % 2

        # boolean operators
        assert context.vars["t"].val == (True or False)
        assert context.vars["u"].val == (True and False)

    def test_different_assignments(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/assignment.qasm", mode="r").read(),
            permissive=True,
        )

        context = QasmInterpreter().interpret(
            ast, context={"wire_map": None, "name": "assignment-exprs"}
        )
        assert context.vars["a"].val == 1
        assert context.vars["b"].val == 1 + 2
        assert context.vars["c"].val == 2 - 3
        assert context.vars["d"].val == 3 * 4
        assert context.vars["e"].val == 4 / 5
        assert context.vars["f"].val == (True and True)
        assert context.vars["g"].val == (True or False)
        assert context.vars["i"].val == 2 ^ 9
        assert context.vars["j"].val == 2 << 10
        assert context.vars["k"].val == 3 >> 1
        assert context.vars["l"].val == 5 % 2
        assert context.vars["m"].val == 6**13

    def test_nested_expr(self):
        # parse the QASM
        ast = parse(
            """
            qubit q0;
            int p = 3 + 3;
            pow(p * 2) @ x q0;
            pow(p * (2 + 1)) @ x q0;
            pow((8.0 - 0.0) * (2.0 + 1)) @ x q0;
            """,
            permissive=True,
        )

        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "expression-implemented"}
            )

        assert q.queue == [
            PauliX("q0") ** 12,
            PauliX("q0") ** 18,
            PauliX("q0") ** 24,
        ]

    def test_bare_expression(self):
        # parse the QASM
        ast = parse(
            """
            bit[8] a = "1001";
            a << 1;
            let b = a;
            """,
            permissive=True,
        )

        context = QasmInterpreter().interpret(
            ast, context={"wire_map": None, "name": "expression-visit"}
        )

        # exprs are not in-place modifiers without an assignment
        assert context.aliases["b"](context).val == 9

    operands = [
        0.0,
        0.1,
        -0.1,
        5.0,
        99999.99,
        -9999.99,
        0,
        1,
        -1,
        10,
        -10,
        -99999,
        99999,
        "0.0",
        "0.1",
        "-0.1",
        "5.0",
        "99999.99",
        "-9999.99",
        "0",
        "1",
        "-1",
        "10",
        "-10",
        "-99999",
        "99999",
    ]

    @pytest.mark.parametrize("operand", operands)
    def test_preprocess_operands(self, operand):
        if isinstance(operand, float):
            assert isinstance(preprocess_operands(operand), float)
        elif isinstance(operand, int):
            assert isinstance(preprocess_operands(operand), int)
        elif operand.isdigit():
            assert isinstance(preprocess_operands(operand), int)
        elif operand.replace(".", "").isnumeric():
            assert isinstance(preprocess_operands(operand), float)


@pytest.mark.external
class TestVariables:

    def test_retrieve_non_existent_attr(self):
        context = Context({"wire_map": None, "name": "retrieve-non-existent-attr"})

        with pytest.raises(
            KeyError,
            match=r"No attribute potato on Context and no potato key found "
            r"on context retrieve-non-existent-attr",
        ):
            print(context.potato)

    def test_bad_alias(self):
        # parse the QASM
        ast = parse(
            """
            let k = j;
            """,
            permissive=True,
        )

        with pytest.raises(
            TypeError, match="Attempt to alias an undeclared variable j in bad-alias"
        ):
            context = QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "bad-alias"}
            )
            context.aliases["k"](context)

    def test_ref_undeclared_var_in_expr(self):
        # parse the QASM program
        ast = parse(
            """
            const float phi = theta + 1.0;
            """,
            permissive=True,
        )

        with pytest.raises(
            TypeError, match="Attempt to use undeclared variable theta in undeclared-var"
        ):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "undeclared-var"})

    def test_ref_uninitialized_var_in_expr(self):
        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            float theta;
            const float phi = theta + 1.0;
            """,
            permissive=True,
        )

        with pytest.raises(ValueError, match="Attempt to reference uninitialized parameter theta!"):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "uninit-var"})

    def test_ref_uninitialized_alias_in_expr(self):
        # parse the QASM program
        ast = parse(
            """
            qubit q0;
            float theta;
            let alpha = theta;
            const float phi = alpha;
            """,
            permissive=True,
        )

        with pytest.raises(ValueError, match="Attempt to reference uninitialized parameter theta!"):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "uninit-var"})

    def test_alias(self):
        ast = parse(
            """
            bit[6] register = "011011";
            let alias = register[0:5];
            bool z = alias[0] + "1";
            bit single = "0";
            let single_alias = single;
            bool x = single_alias + "0";
            """,
            permissive=True,
        )

        # run the program
        context = QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "aliases"})
        assert context.aliases["alias"](context) == "01101"
        assert context.vars["z"].val == 1
        assert context.vars["x"].val == 0

    def test_variables(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/variables.qasm", mode="r").read(), permissive=True
        )

        # run the program
        context = QasmInterpreter().interpret(
            ast, context={"wire_map": None, "name": "advanced-vars"}
        )

        # static vars
        assert context.vars["f"].val == 3.2
        assert context.vars["g"].val == 3
        assert context.vars["h"].val == 2
        assert context.vars["k"].val == 3

        # dynamic vars
        assert context.vars["l"].val
        assert context.vars["m"].val == (3.14159 / 2) * 3.3
        assert context.vars["a"].val == 3.3333333
        assert context.aliases["alias"](context) == "01101"

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
            match="Attempt to mutate a constant i on line 3 that was defined on line 2",
        ):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "mutate-error"})

    def test_retrieve_wire(self):
        # parse the QASM
        ast = parse(
            """
            qubit[0] q;
            let s = q;
            """,
            permissive=True,
        )

        # run the program
        context = QasmInterpreter().interpret(
            ast, context={"wire_map": None, "name": "qubit-retrieve"}
        )

        assert context.aliases["s"](context) == "q"

    def test_retrieve_uninitialized_var(self):
        # parse the QASM program
        ast = parse(
            """
            float theta;
            float phi;
            phi = theta;
            """,
            permissive=True,
        )

        with pytest.raises(ValueError, match="Attempt to reference uninitialized parameter theta"):
            QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "ref-uninitialized-var"}
            )

    def test_classical_variables(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/classical.qasm", mode="r").read(), permissive=True
        )

        # run the program
        context = QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "basic-vars"})

        assert context.vars["i"].val == 4
        assert context.vars["j"].val == 4
        assert context.vars["c"].val == 0
        assert context.vars["k"].val == 4
        assert context.vars["neg"].val == -4
        assert context.vars["comp"].val == 3.5j + 2.5
        assert context.aliases["arr_alias"](context) == [0.0, 1.0]
        assert context.aliases["literal_alias"](context) == 0.0

    def test_updating_variables(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/updating_variables.qasm", mode="r").read(),
            permissive=True,
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "updating-vars"})

        assert q.queue == [PauliX("q0"), PauliY("q0"), RX(0.1, wires=["q0"]), PauliY("q0")]

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

        with pytest.raises(ValueError, match="Attempt to reference uninitialized parameter theta!"):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "uninit-param"})

    def test_cannot_cast(self):
        # parse the QASM program
        ast = parse(
            """
            complex k = 3.0 + 2.0im;
            const uint l = uint(k);
            """,
            permissive=True,
        )

        with pytest.raises(
            TypeError,
            match=escape(
                "Unable to cast complex to UintType: int() argument must be a string, "
                "a bytes-like object or a real number, not 'complex'"
            ),
        ):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "cannot-cast"})

    def test_cast(self):
        # parse the QASM program
        ast = parse(
            """
            float i = 3.0;
            const uint j = int(i);
            float k = 3.0;
            const complex l = complex(k);
            int m = 1;
            float n = float(m);
            bool o = bool(m);
            """,
            permissive=True,
        )

        context = QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "cast"})

        assert isinstance(context.vars["j"].val, int)
        assert isinstance(context.vars["l"].val, complex)
        assert isinstance(context.vars["n"].val, float)
        assert isinstance(context.vars["o"].val, bool)

    def test_update_non_existent_var(self):
        # parse the QASM program
        ast = parse(
            """
            p = 1;
            """,
            permissive=True,
        )

        with pytest.raises(
            TypeError, match="Attempt to use undeclared variable p in non-existent-var"
        ):
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "non-existent-var"})

    def test_invalid_index(self):
        # parse the QASM
        ast = parse(
            """
            qubit q0;
            const float[2] arr = {0.0, 1.0};
            const float[2] arr_two = {0.0, 1.0};
            const [float[2]] index = {arr, arr_two};
            let slice = arr[index];
            """,
            permissive=True,
        )

        # run the program
        with pytest.raises(
            NotImplementedError,
            match="Array index does not evaluate to a single RangeDefinition or Literal at line 6.",
        ):
            context = QasmInterpreter().interpret(
                ast, context={"wire_map": None, "name": "bad-index"}
            )
            context.aliases["slice"](context)


@pytest.mark.external
class TestGates:

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
        # device("default.qubit", wires=[0, 1, 2])

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
        # device("default.qubit", wires=["0q", "1q", "2q"])

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

    def test_updating_variables(self):
        # parse the QASM
        ast = parse(
            open("tests/io/qasm_interpreter/updating_variables.qasm", mode="r").read(),
            permissive=True,
        )

        # run the program
        with queuing.AnnotatedQueue() as q:
            QasmInterpreter().interpret(ast, context={"wire_map": None, "name": "updating-vars"})

        assert q.queue == [
            PauliX("q0"),
            PauliY("q0"),
            RX(0.1, "q0"),
            PauliY("q0"),
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
            ValueError,
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
            match=escape(
                "Attempt to reference wire(s): {'q0'} that have not been declared in uninit-qubit"
            ),
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
            ctrl @ x q1, q0;
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
            CNOT(wires=["q1", "q0"]),
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
