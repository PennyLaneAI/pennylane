"""
This submodule contains the interpreter for QASM 3.0.
"""

import functools
import copy
import inspect
import re
from typing import Callable
from functools import partial

from openqasm3.visitor import QASMNode
from openqasm3.ast import BitstringLiteral, ArrayLiteral, Cast, IndexExpression, RangeDefinition, Identifier, \
    BinaryExpression, UnaryExpression, ClassicalDeclaration, QuantumGate, QubitDeclaration, ConstantDeclaration, \
    ClassicalAssignment, Cast, Identifier, IndexExpression, RangeDefinition, UnaryExpression, BinaryExpression

from pennylane import ops
from pennylane.operation import Operator
from pennylane.control_flow.while_loop import WhileLoopCallable

NON_PARAMETERIZED_GATES = {
    "ID": ops.Identity,
    "H": ops.Hadamard,
    "X": ops.PauliX,
    "Y": ops.PauliY,
    "Z": ops.PauliZ,
    "S": ops.S,
    "T": ops.T,
    "SX": ops.SX,
    "CX": ops.CNOT,
    "CY": ops.CY,
    "CZ": ops.CZ,
    "CH": ops.CH,
    "SWAP": ops.SWAP,
    "CCX": ops.Toffoli,
    "CSWAP": ops.CSWAP,
}

PARAMETERIZED_GATES = {
    "RX": ops.RX,
    "RY": ops.RY,
    "RZ": ops.RZ,
    "P": ops.PhaseShift,
    "PHASE": ops.PhaseShift,
    "U1": ops.U1,
    "U2": ops.U2,
    "U3": ops.U3,
    "CP": ops.CPhase,
    "CPHASE": ops.CPhase,
    "CRX": ops.CRX,
    "CRY": ops.CRY,
    "CRZ": ops.CRZ,
}


class QasmInterpreter:
    """
    Takes the top level node of the AST as a parameter and recursively descends the AST, calling the
    visitor function on each node.
    """

    def __init__(self, permissive=False):
        """
        Initializes the QASM interpreter.
        """
        self.permissive = permissive
        self.executing = False

    @functools.singledispatchmethod
    def visit(self, node: QASMNode, context: dict):
        """
        Visitor function is called on each node in the AST, which is traversed using recursive descent.
        The purpose of this function is to pass each node to the appropriate handler.

        Args:
            node (QASMNode): the QASMNode to visit next.
            context (dict): the current context populated with any locally available variables, etc.

        Returns:
            dict: The context updated after the compilation of the current node into Callables
                to queue into a QNode.

        Raises:
            NameError: When a (so far) unsupported node type is encountered.
        """
        if self.permissive:
            print(
                f"An unrecognized QASM instruction {node.__class__.__name__} "
                f"was encountered on line {node.span.start_line}, in {context['name']}."
            )
        else:
            raise NotImplementedError(
                f"An unsupported QASM instruction {node.__class__.__name__} "
                f"was encountered on line {node.span.start_line}, in {context['name']}."
            )

        return context

    def interpret(self, node: QASMNode, context: dict):
        """
        Entry point for visiting the QASMNodes of a parsed QASM 3.0 program.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor.
        """

        context.update({"wires": [], "vars": {}, "gates": [], "callable": None})
        init_context = copy.deepcopy(context)  # preserved for use in second (execution) pass

        # begin recursive descent traversal
        for value in node.__dict__.values():
            if not isinstance(value, list):
                value = [value]
            for item in value:
                if isinstance(item, QASMNode):
                    self.visit(item, context)
        return context

    # needs to have same signature as visit()
    @visit.register(QubitDeclaration)
    def visit_qubit_declaration(
        self, node: QubitDeclaration, context: dict
    ):  # pylint: disable=no-self-use
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"]. Note: Qubit declarations must be global.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        context["wires"].append(node.qubit.name)

    @staticmethod
    def _all_context_bound(quantum_function):
        """
        Checks whether a partial received all required context during compilation, or if
        it requires execution context because it has parameters that are based on, for example,
        the outcomes of mid-circuit measurements or subroutines.
        Args:
            partial_function (Callable): the partial with compilation context bound.
        """
        if hasattr(quantum_function, "func"):
            if isinstance(quantum_function.func, WhileLoopCallable):
                return False
            else:
                try:
                    inspect.signature(quantum_function.func).bind(
                        *quantum_function.args, **quantum_function.keywords
                    )
                    return True
                except TypeError:
                    return False
        elif len(inspect.signature(quantum_function).parameters) > 0:
            return False
        return True

    def _update_var(self, value: any, name: str, node: QASMNode, context: dict):
        """
        Updates a variable, or raises if it is constant.
        Args:
            value (any): the value to set.
            name (str): the name of the variable.
            node (QASMNode): the QASMNode that corresponds to the update.
            context (dict): the current context.
        """
        context["vars"][name]["val"] = value
        if context["vars"][name]["constant"]:
            raise ValueError(
                f"Attempt to mutate a constant {name} on line {node.span.start_line} that was "
                f"defined on line {context['vars'][name]['line']}"
            )
        context["vars"][name]["line"] = node.span.start_line

    @visit.register(ClassicalAssignment)
    def visit_classical_assignment(self, node: QASMNode, context: dict):
        """
        Registers a classical assignment.
        Args:
            node (QASMNode): the assignment QASMNode.
            context (dict): the current context.
        """
        # references to an unresolved value see a func for now
        name = (
            node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
        )  # str or Identifier
        res = self.eval_expr(node.rvalue, context)
        self._update_var(res, name, node, context)

    def visit_alias_statement(self, node: QASMNode, context: dict):
        """
        Registers an alias statement.
        Args:
            node (QASMNode): the alias QASMNode.
            context (dict): the current context.
        """
        self._init_aliases(context)
        context["aliases"][node.target.name] = self.eval_expr(node.value, context, aliasing=True)

        # we append anything that needs to be computed at execution time to the gates list...
        # aliases can change throughout a program
        context["gates"].append(partial(self.visit_alias_statement, node))

    def retrieve_variable(self, name: str, context: dict):
        """
        Attempts to retrieve a variable from the current context by name.
        Args:
            name (str): the name of the variable to retrieve.
            context (dict): the current context.
        """

        def _warning(context, name):
            raise TypeError(
                f"Attempt to use unevaluated variable {name} in {context['name']}, "
                f"last updated on line {context['vars'][name]['line'] if name in context['vars'] else 'unknown'}."
            )

        if "vars" in context and context["vars"] is not None and name in context["vars"]:
            if isinstance(context["vars"][name], Callable):
                _warning(context, name)
            else:
                res = context["vars"][name]
                if res["val"] is not None:
                    return res
                else:
                    raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        elif (
            "wires" in context
            and context["wires"] is not None
            and name in context["wires"]
            or "outer_wires" in context
            and name in context["outer_wires"]
        ):
            return name
        elif "aliases" in context and context["wires"] is not None and name in context["aliases"]:
            res = context["aliases"][name](context)  # evaluate the alias and de-reference
            if isinstance(res, str):
                return res
            if res["val"] is not None:
                return res
            else:
                raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        else:
            _warning(context, name)

    def _init_vars(self, context: dict):
        """
        context["callable"] = partial(self._execute_all, context)
        Inits the vars dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "vars" not in context:
            context["vars"] = dict()

    def _init_aliases(self, context: dict):
        """
        Inits the aliases dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "aliases" not in context:
            context["aliases"] = dict()

    def _init_gates_list(self, context: dict):
        """
        Inits the gates list on the current context.

        Args:
            context (dict): the current context.
        """
        if "gates" not in context:
            context["gates"] = []

    @staticmethod
    def _get_bit_type_val(var):
        return bin(var["val"])[2:].zfill(var["size"])

    @visit.register(ConstantDeclaration)
    def visit_constant_declaration(self, node: QASMNode, context: dict):
        """
        Registers a constant declaration. Traces data flow through the context, transforming QASMNodes into
        Python type variables that can be readily used in expression eval, etc.
        Args:
            node (QASMNode): The constant QASMNode.
            context (dict): The current context.
        """
        self.visit_classical_declaration(node, context, constant=True)

    @visit.register(ClassicalDeclaration)
    def visit_classical_declaration(self, node: QASMNode, context: dict, constant:bool=False):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.
        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (dict): The current context.
            constant (bool): Whether the classical variable is a constant.
        """

        # compile time tracking of static variables
        self._init_vars(context)
        if node.init_expression is not None:
            # TODO: store AST objects in context instead of these dicts?

            if isinstance(node.init_expression, BitstringLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "size": node.init_expression.width,
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
            elif not isinstance(node.init_expression, ArrayLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
            else:
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": [
                        self.eval_expr(literal, context) for literal in node.init_expression.values
                    ],
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
        else:
            # the var is declared but uninitialized
            context["vars"][node.identifier.name] = {
                "ty": node.type.__class__.__name__,
                "val": None,
                "line": node.span.start_line,
                "constant": constant,
            }

        # runtime Callable
        self._init_gates_list(context)
        context["gates"].append(partial(self.visit_classical_declaration, node))

    def construct_callable(self, context: dict):
        """
        Constructs a callable that can be queued into a QNode.

        Args:
            context (dict): The final context populated with the Callables (called gates) to queue in the QNode.
        """
        context["callable"] = partial(self._execute_all, context)

    def _execute_all(self, context: dict, execution_context: dict):
        """
        Executes all the gates in the context.
        Args:
            context (dict): the context populated with the gates.
            execution_context (dict): the execution context that handles dynamic variables, etc.
        """
        for gate in context["gates"]:
            gate(execution_context) if not self._all_context_bound(gate) else gate()

    def construct_qfunc(self, context: dict, init_context: dict):
        """
        Constructs a Callable quantum function that may be queued into a QNode.
        Args:
            context (dict): the final context resulting from the compilation pass.
            init_context (dict): the initial context.
        Returns:
            dict: the final executed context.
        """
        if "device" not in context:
            wires = [w for w in context["wires"]] if "wires" in context else []
            curr = context
            if "scopes" in curr:
                wires = self._get_wires_helper(curr, wires)
            context["wires"] = wires
        # passes and modifies init_context through gate calls during second pass
        # static variables are already bound in context["gates"]
        context["callable"] = partial(self._execute_all, context, init_context)
        return init_context

    @visit.register(QuantumGate)
    def visit_quantum_gate(self, node: QuantumGate, context: dict):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.
        """
        name = node.name.name.upper()
        if name in PARAMETERIZED_GATES:
            if not node.arguments:
                raise TypeError(
                    f"Missing required argument(s) for parameterized gate {node.name.name}"
                )
            gates_dict = PARAMETERIZED_GATES
        elif name in NON_PARAMETERIZED_GATES:
            gates_dict = NON_PARAMETERIZED_GATES
        else:
            raise NotImplementedError(f"Unsupported gate encountered in QASM: {node.name.name}")

        gate, args, wires = self._gate_setup_helper(node, gates_dict, context)

        if len(node.modifiers) > 0:
            num_control = sum("ctrl" in mod.modifier.name for mod in node.modifiers)
            op_wires = wires[num_control:]
            control_wires = wires[:num_control]
            if "ctrl" in node.modifiers[-1].modifier.name:
                prev, wires = self.apply_modifier(
                    node.modifiers[-1], gate(*args, wires=op_wires), context, control_wires
                )
            else:
                prev, wires = self.apply_modifier(
                    node.modifiers[-1], gate(*args, wires=wires), context, wires
                )

            for mod in node.modifiers[::-1][1:]:
                prev, wires = self.apply_modifier(mod, prev, context, wires)
        else:
            gate(*args, wires=wires)

    def _gate_setup_helper(self, node: QuantumGate, gates_dict: dict, context: dict):
        """
        Helper to setup the quantum gate call, also resolving arguments and wires.

        Args:
            node (QuantumGate): The QuantumGate QASMNode.
            gates_dict (dict): the gates dictionary.
            context (dict): the current context.

        Returns:
            QuantumGate: The gate to execute.
            list: The list of arguments to the QuantumGate.
            list: The wires the gate applies to.
        """
        # setup arguments
        args = []
        for arg in node.arguments:
            args.append(self.evaluate_argument(arg, context))

        # retrieve gate method
        gate = gates_dict[node.name.name.upper()]

        # setup wires
        wires = [
            # parser will sometimes represent as a str and sometimes as an Identifier
            (
                node.qubits[q].name
                if isinstance(node.qubits[q].name, str)
                else node.qubits[q].name.name
            )
            for q in range(len(node.qubits))
        ]

        self._require_wires(wires, context)

        if context["wire_map"] is not None:
            wires = list(map(lambda wire: context["wire_map"][wire], wires))

        return gate, args, wires

    @staticmethod
    def apply_modifier(mod: QuantumGate, previous: Operator, context: dict, wires: list):
        """
        Applies a modifier to the previous gate or modified gate.

        Args:
            mod (QASMNode): The modifier QASMNode.
            previous (Operator): The previous (called) operator.
            context (dict): The current context.
            wires (list): The wires that the operator is applied to.

        Raises:
            NotImplementedError: If the modifier has a param of an as-yet unsupported type.
        """
        # the parser will raise when a modifier name is anything but the three modifiers (inv, pow, ctrl)
        # in the QASM 3.0 spec. i.e. if we change `pow(power) @` to `wop(power) @` it will raise:
        # `no viable alternative at input 'wop(power)@'`, long before we get here.
        assert mod.modifier.name in ("inv", "pow", "ctrl", "negctrl")
        next = None

        if mod.modifier.name == "inv":
            next = ops.adjoint(previous)
        elif mod.modifier.name == "pow":
            if re.search("Literal", mod.argument.__class__.__name__) is not None:
                power = mod.argument.value
            elif "vars" in context and mod.argument.name in context["vars"]:
                power = context["vars"][mod.argument.name]["val"]
            else:
                raise NotImplementedError(
                    f"Unable to handle expression {mod.argument} at this time"
                )
            next = ops.pow(previous, z=power)
        elif mod.modifier.name == "ctrl":
            next = ops.ctrl(previous, control=wires[-1])
            wires = wires[:-1]
        elif mod.modifier.name == "negctrl":
            next = ops.ctrl(previous, control=wires[-1], control_values=[0])
            wires = wires[:-1]

        return next, wires

    @staticmethod
    def evaluate_argument(arg: str, context: dict):
        """
        Constructs a callable that can be queued into a QNode.
        Evaluates an expression.
        Args:
            context (dict): The final context populated with the Callables (called gates) to queue in the QNode.
            node (QASMNode): the expression QASMNode.
            context (dict): the current context.
            aliasing (bool): whether to use aliases or not.
        """
        if re.search("Literal", arg.__class__.__name__) is not None:
            return arg.value
        if arg.name in context["vars"]:
            # the context at this point should reflect the states of the
            # variables as evaluated in the correct (current) scope.
            if context["vars"][arg.name]["val"] is not None:
                return context["vars"][arg.name]["val"]
            raise NameError(f"Attempt to reference uninitialized parameter {arg.name}!")
        raise NameError(f"Undeclared variable {arg.name} encountered in QASM.")

    def eval_expr(self, node: QASMNode, context: dict, aliasing: bool = False):
        """
        Constructs a callable that can be queued into a QNode.
        Evaluates an expression.
        Args:
            context (dict): The final context populated with the Callables (called gates) to queue in the QNode.
            node (QASMNode): the expression QASMNode.
            context (dict): the current context.
            aliasing (bool): whether to use aliases or not.
        """
        res = None
        if isinstance(node, Cast):
            return self.retrieve_variable(node.argument.name, context)["val"]
        elif isinstance(node, BinaryExpression):
            lhs = self.eval_expr(node.lhs, context)
            rhs = self.eval_expr(node.rhs, context)
            res = eval(f"{lhs}{node.op.name}{rhs}")  # TODO: don't use eval
        elif isinstance(node, UnaryExpression):
            res = eval(f"{node.op.name}{self.eval_expr(node.expression, context)}")
        elif isinstance(node, IndexExpression):

            def _index_into_var(var):
                if var["ty"] == "BitType":
                    var = bin(var["val"])[2:].zfill(var["size"])
                else:
                    var = var["val"]
                if isinstance(node.index[0], RangeDefinition):
                    return var[node.index[0].start.value : node.index[0].end.value]
                elif re.search("Literal", node.index[0].__class__.__name__):
                    return var[node.index[0].value]
                else:
                    raise TypeError(
                        f"Array index is not a RangeDefinition or Literal at line {node.span.start_line}."
                    )

            if aliasing:

                def alias(context):
                    try:
                        var = self.retrieve_variable(node.collection.name, context)
                        return _index_into_var(var)
                    except NameError:
                        raise NameError(
                            f"Attempt to alias an undeclared variable "
                            f"{node.collection.name} in {context['name']}."
                        ) from e

                res = alias
            else:
                var = self.retrieve_variable(node.collection.name, context)
                return _index_into_var(var)
        elif isinstance(node, Identifier):
            if aliasing:

                def alias(context):
                    try:
                        return self.retrieve_variable(node.collection.name, context)
                    except NameError as e:
                        raise NameError(
                            f"Attempt to alias an undeclared variable "
                            f"{node.name} in {context['name']}."
                        ) from e

                res = alias
            else:
                try:
                    var = self.retrieve_variable(node.name, context)
                    value = var["val"] if isinstance(var, dict) and "val" in var else var
                    if isinstance(value, Callable):
                        var["val"] = (
                            value(context) if not self._all_context_bound(value) else value()
                        )
                        var["line"] = node.span.start_line
                        value = var["val"]
                    return value
                except NameError as e:
                    raise NameError(
                        str(e) or f"Reference to an undeclared variable {node.name} in {context['name']}."
                    ) from e
        elif isinstance(node, Callable):
            res = node()
        elif re.search("Literal", node.__class__.__name__):
            res = node.value
        # TODO: include all other cases here
        return res

    @staticmethod
    def _require_wires(wires, context):
        """
        Simple helper that checks if we have wires in the current context.

        Args:
            context (dict): The current context.
            wires (list): The wires that are required.

        Raises:
            NameError: If the context is missing a wire.
        """
        missing_wires = []
        for wire in wires:
            if wire not in context["wires"]:
                missing_wires.append(wire)
        if len(missing_wires) > 0:
            raise NameError(
                f"Attempt to reference wire(s): {missing_wires} that have not been declared in {context['name']}"
            )
