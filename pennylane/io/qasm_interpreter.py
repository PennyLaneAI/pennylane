"""
This submodule contains the interpreter for QASM 3.0.
"""
import copy
import inspect
import re
from functools import partial
from typing import Callable

from openqasm3.ast import BitstringLiteral, ArrayLiteral, Cast, IndexExpression, RangeDefinition, Identifier, \
    BinaryExpression, UnaryExpression

from pennylane import ops
from pennylane.control_flow.while_loop import WhileLoopCallable

has_openqasm = True
try:
    from openqasm3.visitor import QASMNode, QASMVisitor
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_openqasm = False  # pragma: no cover

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


class DirtyError(Exception):  # pragma: no cover
    """Exception raised when attempt is made to use a dirty variable in a compilation context."""


class QasmInterpreter(QASMVisitor):
    """
    Overrides generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    overriden visitor function on each node.
    """

    def __init__(self, permissive=False):
        """
        Checks that the openqasm3 package is available, otherwise raises an error.

        Raises:
            ImportError: if the openqasm3 package is not available.
        """
        if not has_openqasm:  # pragma: no cover
            raise ImportError(
                "QASM interpreter requires openqasm3 to be installed"
            )  # pragma: no cover
        super().__init__()

        self.permissive = permissive
        self.raise_if_dirty = False

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
        handler_name = 'visit_' + node.__class__.__name__
        if node.__class__ == list:
            for sub_node in node:
                self.visit(sub_node, context)
        elif hasattr(self, handler_name):
            try:
                getattr(self, handler_name)(node, context)
            except NotImplementedError as e:
                if self.permissive:
                    pass
                else:
                    raise NotImplementedError(str(e)) from e
        elif self.permissive:
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

    def generic_visit(self, node: QASMNode, context: dict):
        """
        Wraps the provided generic_visit method to make the context a required parameter
        and return the context for testability. Constructs the QNode after all of the nodes
        have been visited.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor.
        """

        context.update({"wires": [], "vars": {}, "gates": [], "callable": None})
        init_context = copy.deepcopy(context)  # preserved for use in second (execution) pass

        # begin recursive descent traversal
        try:
            super().generic_visit(node, context)
        except InterruptedError as e:
            print(str(e))
        execution_context = self.construct_qfunc(context, init_context)
        return context, execution_context

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
        if not context["vars"][name]["constant"]:
            context["vars"][name]["dirty"] = True
        else:
            raise ValueError(
                f"Attempt to mutate a constant {name} on line {node.span.start_line} that was "
                f"defined on line {context['vars'][name]['line']}"
            )
        context["vars"][name]["line"] = node.span.start_line

    def visit_ClassicalAssignment(self, node: QASMNode, context: dict):
        """
        Registers a classical assignment.
        Args:
            node (QASMNode): the assignment QASMNode.
            context (dict): the current context.
        """

        def set_local_var(execution_context: dict):
            rhs = partial(self.eval_expr, node.rvalue, execution_context)
            name = (
                node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
            )  # str or Identifier
            res = rhs()
            self._update_var(res, name, node, execution_context)
            return res

        # references to an unresolved value see a func for now
        name = (
            node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
        )  # str or Identifier
        self._update_var(set_local_var, name, node, context)
        context["gates"].append(set_local_var)

    def _choose_context(self, func: Callable, context: dict, execution_context: dict):
        """
        Executes the callable with the right context depending on whether it was bound at compile time or
        needs execution context.
        Args:
            func (Callable): the callable to be executed.
            context (dict): the context to be passed to the function.
            execution_context (dict): the execution context to be passed to the function.
        Returns:
            dict: the context to use.
        """
        if not self._depends_on_dirty_vars(func):
            return context
        else:
            return execution_context

    def _depends_on_dirty_vars(self, func: Callable):
        """
        Checks if any state (variables) relevant to the node being processed is dirty.
        Args:
            func (Callable): the function which may use dirty state.
            context (dict): the current compilation context.
        Returns:
            bool: whether the func depends on dirty state. If it does, the state should not be bound during compilation.
        """
        self.raise_if_dirty = True
        try:
            func()
            self.raise_if_dirty = False
            return False
        except DirtyError:
            self.raise_if_dirty = False
            return True

    def visit_AliasStatement(self, node: QASMNode, context: dict):
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
        context["gates"].append(partial(self.visit_AliasStatement, node))

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

        def _dirty(context, name):
            raise DirtyError(
                f"Attempt to use dirty variable {name} in compilation context {context['name']}."
            )

        if "vars" in context and context["vars"] is not None and name in context["vars"]:
            if isinstance(context["vars"][name], Callable):
                _warning(context, name)
            else:
                res = context["vars"][name]
                if res["dirty"] == True and self.raise_if_dirty:
                    _dirty(context, name)
                return res
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
            else:
                if res["dirty"] == True and self.raise_if_dirty:
                    _dirty(context, name)
            return res
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

    def visit_ConstantDeclaration(self, node: QASMNode, context: dict):
        """
        Registers a constant declaration. Traces data flow through the context, transforming QASMNodes into
        Python type variables that can be readily used in expression eval, etc.
        Args:
            node (QASMNode): The constant QASMNode.
            context (dict): The current context.
        """
        self.visit_ClassicalDeclaration(node, context, constant=True)

    def visit_ClassicalDeclaration(self, node: QASMNode, context: dict, constant:bool=False):
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

            # Note: vars which are clean may be bound at compile time, dirty ones
            # must be calculated during execution
            if isinstance(node.init_expression, BitstringLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "size": node.init_expression.width,
                    "line": node.init_expression.span.start_line,
                    "dirty": False,
                    "constant": constant,
                }
            elif not isinstance(node.init_expression, ArrayLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "line": node.init_expression.span.start_line,
                    "dirty": False,
                    "constant": constant,
                }
            else:
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": [
                        self.eval_expr(literal, context) for literal in node.init_expression.values
                    ],
                    "line": node.init_expression.span.start_line,
                    "dirty": False,
                    "constant": constant,
                }
        else:
            # the var is declared but uninitialized
            context["vars"][node.identifier.name] = {
                "ty": node.type.__class__.__name__,
                "val": None,
                "line": node.span.start_line,
                "dirty": False,
                "constant": constant,
            }

        # runtime Callable
        self._init_gates_list(context)
        context["gates"].append(partial(self.visit_ClassicalDeclaration, node))

    @staticmethod
    def _execute_all(context: dict):
        """
        Executes all the gates in the context.

        Args:
            context (dict): the current context populated with the gates.
        """
        for func in context["gates"]:
            func()

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

    @staticmethod
    def visit_QubitDeclaration(node: QASMNode, context: dict):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"]. TODO: this should be changed to have greater specificity. Coming in a follow-up PR.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        context["wires"].append(node.qubit.name)

    def visit_QuantumGate(self, node: QASMNode, context: dict):
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
            gate = self.gate(PARAMETERIZED_GATES, node, context)
        elif name in NON_PARAMETERIZED_GATES:
            gate = self.gate(NON_PARAMETERIZED_GATES, node, context)
        else:
            raise NotImplementedError(f"Unsupported gate encountered in QASM: {node.name.name}")

        if len(node.modifiers) > 0:
            gate = self.modifiers(gate, node, context)

        context["gates"].append(gate)

    @staticmethod
    def retrieve_variable(name: str, context: dict):
        """
        Attempts to retrieve a variable from the current context by name.

        Args:
            name (str): the name of the variable to retrieve.
            context (dict): the current context.
        """
        if name in context["vars"]:
            # the context at this point should reflect the states of the
            # variables as evaluated in the correct (current) scope.
            if context["vars"][name]["val"] is not None:
                return context["vars"][name]["val"]
            raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        raise NameError(f"Undeclared variable {name} encountered in QASM.")

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
                    return var[node.index[0].start.value: node.index[0].end.value]
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
                        var["dirty"] = True
                        value = var["val"]
                    return value
                except NameError as e:
                    raise NameError(
                        f"Reference to an undeclared variable {node.name} in {context['name']}."
                    ) from e
        elif isinstance(node, Callable):
            res = node()
        elif re.search("Literal", node.__class__.__name__):
            res = node.value
        # TODO: include all other cases here
        return res

    @staticmethod
    def modifiers(gate: Callable, node: QASMNode, context: dict):
        """
        Registers a modifier on a gate. Modifiers are applied to gates differently in Pennylane
        depending on the type of modifier. We build a Callable that applies the modifier appropriately
        at execution time, evaluating the gate Callable appropriately as well.

        Args:
            gate (Callable): The Callable partial built for the gate we wish to modify.
            node (QASMNode): The original QquantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The callable which will appropriately apply the modifier and execute the gate.
        """
        call_stack = [gate]
        for mod in node.modifiers:
            # the parser will raise when a modifier name is anything but the three modifiers (inv, pow, ctrl)
            # in the QASM 3.0 spec. i.e. if we change `pow(power) @` to `wop(power) @` it will raise:
            # `no viable alternative at input 'wop(power)@'`, long before we get here.
            assert mod.modifier.name in ("inv", "pow", "ctrl")
            wrapper = None
            if mod.modifier.name == "inv":
                wrapper = ops.adjoint
            elif mod.modifier.name == "pow":
                if re.search("Literal", mod.argument.__class__.__name__) is not None:
                    wrapper = partial(ops.pow, z=mod.argument.value)
                elif "vars" in context and mod.argument.name in context["vars"]:
                    wrapper = partial(ops.pow, z=context["vars"][mod.argument.name]["val"])
            elif mod.modifier.name == "ctrl":
                wrapper = partial(ops.ctrl, control=gate.keywords["wires"][0:-1])

            call_stack.append(wrapper)

        def call():
            res = None
            for func in call_stack:
                # if there is a control in the stack
                if (
                    call_stack[-1].__class__.__name__ == "partial"
                    and "control" in call_stack[-1].keywords
                ):
                    # if we are processing the control now
                    if "control" in func.keywords:
                        res.keywords["wires"] = [res.keywords["wires"][-1]]
                    # i.e. qml.ctrl(qml.RX, (1))(2, wires=0)
                    res = func(res.func)(**res.keywords) if res is not None else func
                else:
                    # i.e. qml.pow(qml.RX(1.5, wires=0), z=4)
                    res = func(res) if res is not None else func()

        return call

    @staticmethod
    def _require_wires(context):
        """
        Simple helper that checks if we have wires in the current context.

        Args:
            context (dict): The current context.

        Raises:
            NameError: If the context is missing a wire.
        """
        if len(context["wires"]) == 0:
            raise NameError(
                f"Attempt to reference wires that have not been declared in {context['name']}"
            )

    def gate(self, gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a gate application. Builds a Callable partial
        that can be executed when the QNode is called. The gate will be executed at that time
        with the appropriate arguments.
        TODO: a robust method for retrieving vars from context will be provided in follow-up PR according to [sc-90383]

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The Callable partial that will execute the gate with the appropriate arguments at
                "runtime".

        Raises:
            NameError: If an argument is not found in the current context.
        """
        self._require_wires(context)
        args = []
        for arg in node.arguments:
            if re.search("Literal", arg.__class__.__name__) is not None:
                args.append(arg.value)
            else:
                args.append(self.retrieve_variable(arg.name, context))
        return partial(
            gates_dict[node.name.name.upper()],
            *args,
            wires=[
                # parser will sometimes represent as a str and sometimes as a Identifier
                (
                    node.qubits[q].name
                    if isinstance(node.qubits[q].name, str)
                    else node.qubits[q].name.name
                )
                for q in range(len(node.qubits))
            ],
        )
