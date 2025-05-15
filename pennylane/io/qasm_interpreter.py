import re

from functools import partial
from typing import Callable, Iterable

from openqasm3.ast import QuantumArgument, Identifier, IntegerLiteral, BinaryExpression, Cast, RangeDefinition, \
    ArrayLiteral, UnaryExpression, WhileLoop, IndexExpression, BitstringLiteral, ForInLoop, EndStatement
from openqasm3.visitor import QASMVisitor, QASMNode

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
    QNode,
    S,
    T,
    Toffoli,
    device,
    while_loop,
    for_loop,
    wires,
    ctrl,
    adjoint,
    pow,
    measure,
    cond
)

SINGLE_QUBIT_GATES = {
    "ID": Identity,
    "H": Hadamard,
    "X": PauliX,
    "Y": PauliY,
    "Z": PauliZ,
    "S": S,
    "T": T,
    "SX": SX,
}

PARAMETERIZED_SIGNLE_QUBIT_GATES = {
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "P": PhaseShift,
    "PHASE": PhaseShift,
    "U1": U1,
    "U2": U2,
    "U3": U3,
}

TWO_QUBIT_GATES = {
    "CX": CNOT,
    "CY": CY,
    "CZ": CZ,
    "CH": CH,
    "SWAP": SWAP,
    "CSWAP": CSWAP,
    "CP": CPhase,
    "CPHASE": CPhase,
    "CRX": CRX,
    "CRY": CRY,
    "CRZ": CRZ,
}

MULTI_QUBIT_GATES = {
    "CCX": Toffoli,
}


class QasmInterpreter(QASMVisitor):
    """
    Overrides generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    overriden visitor function on each node.

    There are two passes. The first queues Callables such as gate partials into a QNode,
    so that the QNode can be called and the program will be executed at that time. During
    this first pass, any available values provided by the program like literals are used to
    optimize the compilation with as much detail as possible. A simulation does not occur
    during the first pass which just queues Callables into a QNode. The second pass occurs
    during the execution of the QNode and involves simulating everything. All remaining data
    flow and control flow is evaluated completely during this second pass. The control flow
    is handled using Pennylane provided qml.while_loop and qml.for_loop etc. to be compatible
    with qjit. The data flow is traced through the context, which is mutated during each pass.

    The first pass does optimization using static values only. We therefore need to track whether
    values are dirty.
    """

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
            InterruptedError: When a QASM program is terminated by an end instruction.
        """
        # CamelCase -> snake_case
        handler_name = re.sub(r'(?<!^)(?=[A-Z])', '_', node.__class__.__name__).lower()
        if node.__class__ == list:
            for sub_node in node:
                self.visit(sub_node, context)
        elif node.__class__ == EndStatement:
            raise InterruptedError(
                f"The QASM program was terminated om line {node.span.start_line}."
                f"There may be unprocessed QASM code."
            )
        elif hasattr(self, handler_name):
            getattr(self, handler_name)(node, context)
        else:
            print(f"An unrecognized QASM instruction {node.__class__.__name__} "  # TODO: change to warning
                  f"was encountered on line {node.span.start_line}, in {context['name']}.")
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
            dict: The context updated after the compilation of all nodes by the visitor. Contains a QNode
                with a list of Callables that are queued into it.
        """
        try:
            super().generic_visit(node, context)
        except InterruptedError as e:
            print(str(e))
        self.construct_qnode(context)
        return context

    def branching_statement(self, node: QASMNode, context: dict):
        """
        Registers a branching statement. Like switches, uses qml.cond.
        """
        self._init_branches_scope(node, context)
        self._init_gates_list(context)

        # create the true body context
        context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"] = \
            self._init_clause_in_same_namespace(
                context,
                f'{context["name"]}_branch_{node.span.start_line}_true_body'
            )


        # process the true body
        self.visit(
            node.if_block,
            context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"]
        )

        if hasattr(node, "else_block"):

            # create the false body context
            context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"] = \
                self._init_clause_in_same_namespace(
                    context,
                    f'{context["name"]}_branch_{node.span.start_line}_false_body'
                )

            # process the false body
            self.visit(
                node.else_block,
                context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"]
            )

        def branch():
            cond(
                self.eval_expr(node.condition, context),
                lambda: [
                    gate() for gate in
                    context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"]["gates"]
                ],
                lambda: [
                    gate() for gate in
                    context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"]["gates"]
                ] if "false_body" in context["scopes"]["branches"][f"branch_{node.span.start_line}"] else None,
            )()

        context["gates"].append(branch)

    def switch_statement(self, node: QASMNode, context: dict):
        """
        Registers a switch statement.
        """
        self._init_switches_scope(node, context)
        self._init_gates_list(context)

        # switches need to have access to the outer context but not get called unless the condition is met

        # we need to keep track of each clause individually
        for i, case in enumerate(node.cases):
            context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i}"] = \
                self._init_clause_in_same_namespace(
                    context,
                    f'{context["name"]}_switch_{node.span.start_line}_cond_{i}'
                )

            # process the individual clauses
            self.visit(
                case[1].statements,
                context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i}"]
            )

        context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"] = \
            self._init_clause_in_same_namespace(
                context,
                f'{context["name"]}_switch_{node.span.start_line}_cond_{i + 1}'
            )

        # process the default case
        self.visit(
            node.default.statements,
            context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"]
        )

        # TODO: need to propagate any qubit declarations to outer scope if and when the inner scope(s) are called

        def switch():
            target = self.retrieve_variable(node.target.name, context)
            cond(
                # TODO: support eval of lists, etc. to match
                target["val"] == self.eval_expr(node.cases[0][0][0], context),
                lambda: [
                    gate() for gate in
                    context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_0"]["gates"]
                ],
                lambda: [
                    gate() for gate in
                    context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"]["gates"]
                ],
                [
                    (target == self.eval_expr(node.cases[j][0][0], context), lambda: [
                        gate() for gate in
                        context["scopes"]["switches"][f"switch_{node.span.start_line}"][case]["gates"]
                    ]) for j, case in
                    enumerate(list(context["scopes"]["switches"][f"switch_{node.span.start_line}"].keys())[1:-1])
                ]
            )()

        context["gates"].append(switch)

    def alias_statement(self, node: QASMNode, context: dict):
        """
        Registers an alias statement.
        """
        self._init_aliases(context)
        context["aliases"][node.target.name] = self.eval_expr(node.value, context, aliasing=True)

    @staticmethod
    def retrieve_variable(name: str, context: dict):
        """
        Attempts to retrieve a variable from the current context by name.
        """
        def _warning(context, name):
            print(
                f"Attempt to use unevaluated variable {name} in {context['name']}, "
                f"last updated on line {context['vars'][name]['line'] if name in context['vars'] else 'unknown'}."
            )  # TODO: make a warning

        if name in context["vars"]:
            if isinstance(context["vars"][name], Callable):
                _warning(context, name)
            else:
                return context["vars"][name]
        elif "wires" in context and name in context["wires"] \
                or "outer_wires" in context and name in context["outer_wires"]:
            return name
        elif "aliases" in context and name in context["aliases"]:
            return context["aliases"][name](context)  # evaluate the alias and de-reference
        else:
            _warning(context, name)

    def eval_expr(self, node: QASMNode, context: dict, aliasing: bool = False):
        """
        Evaluates an expression.
        """
        res = None
        if isinstance(node, Cast):
            return self.retrieve_variable(node.argument.name, context)["val"]
        elif isinstance(node, BinaryExpression):
            lhs = self.eval_expr(node.lhs, context)
            rhs = self.eval_expr(node.rhs, context)
            res = eval(f'{lhs}{node.op.name}{rhs}')
        elif isinstance(node, UnaryExpression):
            res = eval(f'{node.op.name}{self.eval_expr(node.expression, context)}')
        # TODO: aliasing should be possible when an index is not provided as well
        elif isinstance(node, IndexExpression):
            if aliasing:
                def alias(context):
                    try:
                        return self.retrieve_variable(node.collection.name, context)
                    except NameError:
                        # TODO: make a warning
                        print(f"Attempt to alias an undeclared variable {node.collection.name} in {context['name']}.")

                    # if isinstance(node.index[0], RangeDefinition): TODO: support indexing here
                    #     ret = ret[node.index[0].start: node.index[0].end: node.index[0].step]
                    # return ret
                res = alias
            else:
                return self.retrieve_variable(node.collection.name, context)["val"]
        elif isinstance(node, Identifier):
            try:
                var = self.retrieve_variable(node.name, context)
                return var["val"] if isinstance(var, dict) and "val" in var else var  # could be var or qubit
            except NameError:
                # TODO: make a warning
                print(f"Reference to an undeclared variable {node.name} in {context['name']}.")
        elif re.search('Literal', node.__class__.__name__):
            res = node.value
        # TODO: include all other cases here
        return res

    def _init_clause_in_same_namespace(self, outer_context: dict, name: str):
        outer_wires = outer_context["wires"] if "wires" in outer_context else None
        if "outer_wires" in outer_context:
            outer_wires = outer_context["outer_wires"]
        context = {
                "vars": outer_context["vars"]  if "vars" in outer_context else None,
                "outer_wires": outer_wires,
                "wires": [],
                "name": name
        }
        return context

    def _init_aliases(self, context: dict):
        """
        Inits the aliases dict on the current context.
        """
        if "aliases" not in context:
            context["aliases"] = dict()

    def _init_switches_scope(self, node: QASMNode, context: dict):
        """
        Inits the switches scope on the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {
                "switches": dict()
            }
        elif "switches" not in context["scopes"]:
            context["scopes"]["switches"] = dict()

        context["scopes"]["switches"][f"switch_{node.span.start_line}"] = dict()

    def _init_branches_scope(self, node: QASMNode, context: dict):
        """
        Inits the branches scope on the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {
                "branches": dict()
            }
        elif "branches" not in context["scopes"]:
            context["scopes"]["branches"] = dict()

        context["scopes"]["branches"][f"branch_{node.span.start_line}"] = dict()

    def _init_gates_list(self, context: dict):
        """
        Inits the gates list on the current context.
        """
        if "gates" not in context:
            context["gates"] = []

    def _init_outer_wires_list(self, context: dict):
        """
        Inits the outer wires list on a sub context.
        """
        if "outer_wires" not in context:
            context["outer_wires"] = []

    def _init_loops_scope(self, node: QASMNode, context: dict):
        """
        Inits the loops scope on the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {
                "loops": dict()
            }
        elif "loops" not in context["scopes"]:
            context["scopes"]["loops"] = dict()

        # the namespace is shared with the outer scope, but we need to keep track of the gates separately
        if isinstance(node, WhileLoop):
            context["scopes"]["loops"][f"while_{node.span.start_line}"] = \
                self._init_clause_in_same_namespace(context, f'{context["name"]}_while_{node.span.start_line}')

        elif isinstance(node, ForInLoop):
            context["scopes"]["loops"][f"for_{node.span.start_line}"] = \
                self._init_clause_in_same_namespace(context, f'{context["name"]}_for_{node.span.start_line}')

    def _init_subroutine_scope(self, node: QASMNode, context: dict):
        """
        Inits the subroutine scope on the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {
                "subroutines": dict()
            }
        elif "subroutines" not in context["scopes"]:
            context["scopes"]["subroutines"] = dict()

        # outer scope variables are available to inner scopes... but not vice versa!
        # names prefixed with outer scope names for specificity
        context["scopes"]["subroutines"][node.name.name] = \
            self._init_clause_in_same_namespace(context, f'{context["name"]}_{node.name.name}')

    def while_loop(self, node: QASMNode, context: dict):
        """
        Registers a while loop. TODO: break and continue
        """
        self._init_gates_list(context)
        self._init_loops_scope(node, context)

        @while_loop(partial(self.eval_expr, node.while_condition, context))  # traces data dep through context
        def loop(context):
            # we don't want to populate the gates again with every call to visit
            context["scopes"]["loops"][f"while_{node.span.start_line}"]["gates"] = []
            # process loop body...
            inner_context = self.visit(
                node.block,
                context["scopes"]["loops"][f"while_{node.span.start_line}"]
            )
            for gate in inner_context["gates"]:
                gate()  # updates vars in context... need to propagate these to outer scope
            context["vals"] = inner_context["vals"] if "vals" in inner_context else None
            context["wires"] += inner_context["wires"] if "wires" in inner_context else None
            return context

        context["gates"].append(loop)

    def for_in_loop(self, node: QASMNode, context: dict):
        """
        Registers a for loop.  TODO: break and continue
        """
        self._init_gates_list(context)
        self._init_loops_scope(node, context)

        loop_params = node.set_declaration

        # de-referencing
        if isinstance(loop_params, Identifier):
            # TODO: could be a ref to a range? support AST types, etc. in context instead of python types?
            loop_params = self.retrieve_variable(loop_params.name, context)
            if loop_params["ty"] == "BitType":
                loop_params =  bin(loop_params["val"])[2:].zfill(loop_params["size"])

        if isinstance(loop_params, RangeDefinition):
            start = self.eval_expr(loop_params.start, context)
            stop = self.eval_expr(loop_params.end, context)
            step = self.eval_expr(loop_params.step, context)
            if step is None:
                step = 1

            @for_loop(start, stop, step)
            def loop(i, context):
                # we don't want to populate the gates again with every call to visit
                context["scopes"]["loops"][f"for_{node.span.start_line}"]["gates"] = []
                # process loop body
                inner_context = self.visit(
                    node.block,
                    context["scopes"]["loops"][f"for_{node.span.start_line}"]
                )
                for gate in inner_context["gates"]:  # we only want to execute the gates in the loop's scope
                    gate()  # updates vars in sub context... need to propagate these to outer context
                context["vals"] = inner_context["vals"] if "vals" in inner_context else None
                context["wires"] += inner_context["wires"] if "wires" in inner_context else None
                return context

            context["gates"].append(loop)

        # we unroll the loop in the following case when we don't have a range since qml.for_loop() only
        # accepts (start, stop, step) and nto a list of values.
        elif isinstance(loop_params, ArrayLiteral):
            iter = [self.eval_expr(literal, context) for literal in loop_params.values]
        elif isinstance(loop_params, Iterable):  # it's an array literal that's been eval'd before TODO: unify these reprs?
            iter = [val for val in loop_params]
            def unrolled():
                for i in iter:
                    context["scopes"]["loops"][f"for_{node.span.start_line}"]["vars"][node.identifier.name] = i
                    context["scopes"]["loops"][f"for_{node.span.start_line}"]["gates"] = []
                    # visit the nodes once per loop iteration
                    self.visit(node.block, context["scopes"]["loops"][f"for_{node.span.start_line}"])
                    for gate in context["scopes"]["loops"][f"for_{node.span.start_line}"]:
                        gate()  # updates vars in sub context if any measurements etc. occur

            context["gates"].append(unrolled)
        elif loop_params is None:  # could be func param... then it's a value that will be evaluated at "runtime" (when calling the QNode)
            print(f"Uninitialized iterator in loop {f'for_{node.span.start_line}'}.")

    def return_statement(self, node: QASMNode, context: dict):
        """
        Registers a return statement. Points to the var that needs to be set in an outer scope when this
        subroutine is called.
        """
        context["return"] = node.expression.name

    def quantum_measurement_statement(self, node: QASMNode, context: dict):
        """
        Registers a quantum measurement.
        """
        self._init_gates_list(context)
        if isinstance(node.measure.qubit, Identifier):
            meas = partial(measure, node.measure.qubit.name)

        elif isinstance(node.measure.qubit, IntegerLiteral):  # TODO: are all these cases necessary
            meas = partial(measure, node.measure.qubit.value)

        elif isinstance(node.measure.qubit, list):
            for qubit in node.measure.qubit:
                if isinstance(qubit, Identifier):
                    meas = partial(measure, qubit.name)

                elif isinstance(qubit, IntegerLiteral):
                    meas = partial(measure, qubit.value)

        # handle data flow. Note: data flow dependent on quantum operations deferred? Promises?
        def set_local_var():
            name = node.target.name if isinstance(node.target.name, str) else node.target.name.name  # str or Identifier
            res = meas()
            context["vars"][name]["val"] = res
            context["vars"][name]["line"] = node.span.start_line

        # references to an unresolved value see a func for now
        name = node.target.name if isinstance(node.target.name, str) else node.target.name.name  # str or Identifier
        context["vars"][name]["val"] = set_local_var
        context["gates"].append(set_local_var)

    def quantum_reset(self, node: QASMNode, context: dict):
        """
        Registers a reset of a quantum gate.
        """
        self._init_gates_list(context)
        if isinstance(node.qubits, Identifier):
            context["gates"].append(
                partial(measure, node.qubits.name, reset=True)
            )
        elif isinstance(node.qubits, IntegerLiteral):  # TODO: are all these cases necessary / supported
            context["gates"].append(
                partial(measure, node.qubits.value, reset=True)
            )
        elif isinstance(node.qubits, list):
            for qubit in node.qubits:
                if isinstance(qubit, Identifier):
                    context["gates"].append(
                        partial(measure, qubit.name, reset=True)
                    )
                elif isinstance(qubit, IntegerLiteral):
                    context["gates"].append(
                        partial(measure, qubit.value, reset=True)
                    )

    def subroutine_definition(self, node: QASMNode, context: dict):
        """
        Registers a subroutine definition. Maintains a namespace in the context, starts populating it with
        its parameters.
        """
        self._init_subroutine_scope(node, context)

        # register the params
        for param in node.arguments:
            if not isinstance(param, QuantumArgument):
                context["scopes"]["subroutines"][node.name.name]["vars"][param.name.name] = {
                    'ty': param.__class__.__name__,
                    'val': None,
                    'line': param.span.start_line
                }
            else:
                context["scopes"]["subroutines"][node.name.name]["wires"].append(param.name.name)

        # process the subroutine body. Note we don't call the gates in the outer context until the subroutine is called.
        context["scopes"]["subroutines"][node.name.name] = self.visit(
            node.body,
            context["scopes"]["subroutines"][node.name.name]
        )

        # Should we visit now or when the function is called with arguments?
        # Now is fine b/c we evaluate vars at the end, and visit only constructs partials that
        # reference them during a visit.

    def _get_wires_helper(self, curr: dict, wires: list):
        """
        We need a device with enough wires to support all the qubit declarations in every sub-context.
        We need to instantiate a device with enough wires to support all qubit declarations, with names
        that give enough specificity to identify them when they are in different scopes but share the same
        name in the QASM file, for example.
        """
        if "scopes" in curr:  # TODO: raise warning when a variable is shadowed
            contexts = curr["scopes"]
            for context_type, typed_contexts in contexts.items():
                for typed_context_name, typed_context in typed_contexts.items():
                    if context_type != "switches" and context_type != "branches":
                        wires += [f'{contexts[context_type][typed_context_name]["name"]}_{w}'
                                  for w in contexts[context_type][typed_context_name]["wires"]]
                        wires = self._get_wires_helper(typed_context, wires)
                    else:
                        # TODO: account for: we don't need new wires for scopes that don't have their own namespaces
                        for cond in typed_context.keys():
                            wires += [f'{typed_context[cond]["name"]}_{w}'
                                      for w in typed_context[cond]["wires"]]
                            wires = self._get_wires_helper(typed_context[cond], wires)
        return wires

    def construct_qnode(self, context: dict):
        if "device" not in context:
            wires = [w for w in context["wires"]]
            curr = context
            if "scopes" in curr:
                wires = self._get_wires_helper(curr, wires)
            context["device"] = device("default.qubit", wires=wires)
        # TODO: pass context through gate calls during second pass
        context["qnode"] = QNode(lambda: [gate() for gate in context["gates"]], device=context["device"])

    @staticmethod
    def identifier(node: QASMNode, context: dict):
        """
        Registers an identifier in the current context.

        Args:
            node (QASMNode): The Identifier QASMNode.
            context (dict): The current context.
        """
        if not hasattr(context, "identifiers"):
            context["identifiers"] = []
        context["identifiers"].append(node.name.name)

    @staticmethod
    def qubit_declaration(node: QASMNode, context: dict):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"].

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        if "wires" not in context:
            context["wires"] = []
        context["wires"].append(node.qubit.name)

    def classical_declaration(self, node: QASMNode, context: dict):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.

        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (dict): The current context.
        """
        if "vars" not in context:
            context["vars"] = {}
        if node.init_expression is not None:
            # TODO: store AST objects in context instead of these objects?
            if isinstance(node.init_expression, BitstringLiteral):
                context["vars"][node.identifier.name] = {
                    'ty': node.type.__class__.__name__,
                    'val': self.eval_expr(node.init_expression, context),
                    'size': node.init_expression.width,
                    'line': node.init_expression.span.start_line
                }
            elif not isinstance(node.init_expression, ArrayLiteral):
                context["vars"][node.identifier.name] = {
                    'ty': node.type.__class__.__name__,
                    'val': self.eval_expr(node.init_expression, context),
                    'line': node.init_expression.span.start_line
                }
            else:
                context["vars"][node.identifier.name] = {
                    'ty': node.type.__class__.__name__,
                    'val': [self.eval_expr(literal, context) for literal in node.init_expression.values],
                    'line': node.init_expression.span.start_line
                }
        else:
            # the var is declared but uninitialized
            context["vars"][node.identifier.name] = {
                'ty': node.type.__class__.__name__,
                'val': None,
                'line': node.span.start_line
            }

    def quantum_gate(self, node: QASMNode, context: dict):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.
        """
        gate = None
        self._init_gates_list(context)
        if node.name.name.upper() in SINGLE_QUBIT_GATES:
            gate = self.non_parameterized_gate(SINGLE_QUBIT_GATES, node, context)
        elif node.name.name.upper() in PARAMETERIZED_SIGNLE_QUBIT_GATES:
            gate = self.param_single_qubit_gate(node, context)
        elif node.name.name.upper() in TWO_QUBIT_GATES:
            gate = self.non_parameterized_gate(TWO_QUBIT_GATES, node, context)
        elif node.name.name.upper() in MULTI_QUBIT_GATES:
            gate = self.non_parameterized_gate(MULTI_QUBIT_GATES, node, context)
        else:
            print(f"Unsupported gate encountered in QASM: {node.name}")  # TODO: change to warning
        if len(node.modifiers) > 0:
            gate = self.modifiers(gate, node, context)

        context["gates"].append(gate)

    def modifiers(self, gate: Callable, node: QASMNode, context: dict):
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
            if mod.modifier.name == "inv":
                wrapper = adjoint
            elif mod.modifier.name == "pow":
                if re.search("Literal", mod.argument.__class__.__name__) is not None:
                    wrapper = partial(pow, z=mod.argument.value)
                elif mod.argument.name in context["vars"]:
                    wrapper = partial(pow, z=self.retrieve_variable(mod.argument.name, context)["val"])
            elif mod.modifier.name == 'ctrl':
                wrapper = partial(ctrl, control=gate.keywords["wires"][0:-1])
            call_stack = [wrapper] + call_stack

        def call():
            res = None
            for callable in call_stack[::-1]:
                # checks there is a control in the stack
                if ('partial' == call_stack[0].__class__.__name__ and 'control' in call_stack[0].keywords):
                    # checks we are processing the control now
                    if 'control' in callable.keywords:
                        res.keywords["wires"] = [res.keywords["wires"][-1]]  # ctrl on all wires but the target
                    # i.e. qml.ctrl(qml.RX, (1))(2, wires=0)
                    res = callable(res.func)(**res.keywords) if res is not None else callable
                else:
                    # i.e. qml.pow(qml.RX(1.5, wires=0), z=4)
                    res = callable(res) if res is not None else callable()

        return call

    def param_single_qubit_gate(self, node: QASMNode, context: dict):
        """
        Registers a parameterized single qubit gate application. Builds a Callable partial
        that can be executed when the QNode is called. The gate will be executed aat that time
        with the appropriate arguments.

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The Callable partial that will execute the gate with the appropriate arguments at
                "runtime".

        Raises:
            NameError: If an argument is not found in the current context.
        """
        args = []
        for arg in node.arguments:
            if hasattr(arg, "name"):
                # the context at this point should reflect the states of the
                # variables as evaluated in the correct (current) scope.
                # But what about deferred evaluations?
                args.append(self.retrieve_variable(arg.name, context)["val"])
            elif re.search('Literal', arg.__class__.__name__) is not None:
                args.append(arg.value)
        return partial(
            PARAMETERIZED_SIGNLE_QUBIT_GATES[node.name.name.upper()],
            *args,
            wires=wires.Wires([
                self.eval_expr(node.qubits[0], context)
            ])
        )

    def non_parameterized_gate(self, gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a multi qubit gate application. Builds a Callable partial that
        will execute the gate with the appropriate arguments at "runtime".

        Args:
            gates_dict (dict): A mapping from QASM std lib naming to Pennylane gate class.
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The Callable partial that will execute the gate on the appropriate qubits.
        """
        return partial(
            gates_dict[node.name.name.upper()],
            wires=[
                wires.Wires([
                    self.eval_expr(node.qubits[q], context)
                ]) for q in range(len(node.qubits))
            ]
        )
