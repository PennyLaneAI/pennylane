import re

from functools import partial
from typing import Callable

from openqasm3.ast import QuantumArgument, Identifier, IntegerLiteral, BinaryExpression, Cast, RangeDefinition, \
    ArrayLiteral, UnaryExpression, WhileLoop
from openqasm3.visitor import QASMVisitor, QASMNode
from pygments.lexers.robotframework import ForLoop

import pennylane
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
    for_loop
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
        """
        # CamelCase -> snake_case
        handler_name = re.sub(r'(?<!^)(?=[A-Z])', '_', node.__class__.__name__).lower()
        if node.__class__ == list:
            for sub_node in node:
                self.visit(sub_node, context)
        elif hasattr(self, handler_name):
            getattr(self, handler_name)(node, context)
        else:
            print(f"An unrecognized QASM instruction {node.__class__.__name__} "  # TODO: change to warning
                  f"was encountered on line {node.span.start_line}.")
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
        super().generic_visit(node, context)
        self.construct_qnode(context)
        return context

    @staticmethod
    def retrieve_variable(name: str, context: dict): # TODO: update this to account for deferred evals?
        """
        Attempts to retrieve a variable from the current context by name.
        """
        if name in context["vars"]:
            return context["vars"][name]
        else:
            raise NameError(
                f"Uninitialized variable {name} encountered in QASM."
            )

    def eval_expr(self, node: QASMNode, context: dict):
        """
        Evaluates an expression.
        """
        res = None
        if isinstance(node, BinaryExpression):
            if re.search('Literal', node.lhs.__class__.__name__) is not None:
                lhs = node.lhs.value
            elif isinstance(node.lhs, Cast):
                lhs = self.retrieve_variable(node.lhs.argument.name, context)["val"]
            else:
                lhs = self.retrieve_variable(node.lhs.name, context)["val"]
            if re.search('Literal', node.rhs.__class__.__name__) is not None:
                rhs = node.rhs.value
            elif isinstance(node.rhs, Cast):
                rhs = self.retrieve_variable(node.rhs.argument.name, context)["val"]
            else:
                rhs = self.retrieve_variable(node.rhs.name, context)["val"]
            res = eval(f'{lhs}{node.op.name}{rhs}')
        elif isinstance(node, UnaryExpression):
            res = eval(f'{node.op.name}{self.eval_expr(node.expression, context)}')
        elif re.search('Literal', node.__class__.__name__):
            res = node.value
        # TODO: include all other cases here, such as references to vars in the current scope (Identifier)
        return res

    def _init_gates_list(self, context: dict):
        """
        Inits the gates list on the curren context.
        """
        if "gates" not in context:
            context["gates"] = []

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
            context["scopes"]["loops"][f"while_{node.span.start_line}"] = {
                "vars": context["vars"],
                "wires": context["wires"],
                "name": f'{context["name"]}_while_{node.span.start_line}'
            }
        elif isinstance(node, ForLoop):
            context["scopes"]["loops"][f"for_{node.span.start_line}"] = {
                "vars": context["vars"],
                "wires": context["wires"],
                "name": f'{context["name"]}_for_{node.span.start_line}'
            }

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

        context["scopes"]["subroutines"][node.name.name] = {
            "vars": context["vars"],  # outer scope variables are available to inner scopes... but not vice versa!
            "wires": context["wires"],
            "name": f'{context["name"]}_{node.name.name}'  # names prefixed with outer scope names for specificity
        }

    def while_loop(self, node: QASMNode, context: dict):
        """
        Registers a while loop.
        """
        self._init_gates_list(context)
        self._init_loops_scope(node, context)

        @while_loop(partial(self.eval_expr, node.while_condition))  # traces data dep through context
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
            context["vals"] = inner_context["vals"]
            context["wires"] = inner_context["wires"]
            return context

        context["gates"].append(loop)

    def for_loop(self, node: QASMNode, context: dict):
        """
        Registers a for loop.
        """
        self._init_gates_list(context)
        self._init_loops_scope(node, context)

        loop_params = node.set_declaration

        # de-referencing
        if isinstance(loop_params, Identifier):
            # TODO: could be a ref to a range? support AST types, etc. in context instead of python types?
            loop_params = self.retrieve_variable(loop_params.name, context)["val"]

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
                context["vals"] = inner_context["vals"]
                context["wires"] = inner_context["wires"]
                return context

            context["gates"].append(loop)

        # we unroll the loop in the following case when we don't have a range since qml.for_loop() only
        # accepts (start, stop, step) and nto a list of values.
        elif isinstance(loop_params, ArrayLiteral):
            iter = [self.eval_expr(literal, context) for literal in loop_params.values]
        elif isinstance(loop_params, list):  # it's an array literal that's been eval'd before TODO: unify these reprs?
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
            measure = partial(pennylane.measure, node.measure.qubit.name)

        elif isinstance(node.measure.qubit, IntegerLiteral):  # TODO: are all these cases necessary
            measure = partial(pennylane.measure, node.measure.qubit.value)

        elif isinstance(node.measure.qubit, list):
            for qubit in node.measure.qubit:
                if isinstance(qubit, Identifier):
                    measure = partial(pennylane.measure, qubit.name)

                elif isinstance(qubit, IntegerLiteral):
                    measure = partial(pennylane.measure, qubit.value)

        # handle data flow. Note: data flow dependent on quantum operations deferred? Promises?
        def set_local_var():
            res = measure()
            context["vars"][node.target.name]["val"] = res

        context["vars"][node.target.name]["val"] = set_local_var  # references to an unresolved value see a func for now
        context["gates"].append(set_local_var)

    def quantum_reset(self, node: QASMNode, context: dict):
        """
        Registers a reset of a quantum gate.
        """
        self._init_gates_list(context)
        if isinstance(node.qubits, Identifier):
            context["gates"].append(
                partial(pennylane.measure, node.qubits.name, reset=True)
            )
        elif isinstance(node.qubits, IntegerLiteral):  # TODO: are all these cases necessary / supported
            context["gates"].append(
                partial(pennylane.measure, node.qubits.value, reset=True)
            )
        elif isinstance(node.qubits, list):
            for qubit in node.qubits:
                if isinstance(qubit, Identifier):
                    context["gates"].append(
                        partial(pennylane.measure, qubit.name, reset=True)
                    )
                elif isinstance(qubit, IntegerLiteral):
                    context["gates"].append(
                        partial(pennylane.measure, qubit.value, reset=True)
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
                # wire mapping is all messed up now, should be named wires
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
        if "scopes" in curr:
            contexts = curr["scopes"]
            for context_type, typed_contexts in contexts.items():
                for typed_context_name, typed_context in typed_contexts.items():
                    wires += [f'{contexts["scopes"][context_type][typed_context_name]["name"]}_{w}'
                              for w in contexts["scopes"][context_type][typed_context_name]["wires"]]
                    wires += self._get_wires_helper(typed_context, wires)
        return wires

    def construct_qnode(self, context: dict):
        if "device" not in context:
            wires = [w for w in context["wires"]]
            curr = context
            wires = self._get_wires_helper(curr, wires)
            context["device"] = device("default.qubit", wires=wires)
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
        in context["wires"]. TODO: this should be changed to have greater specificity. Coming in a follow-up PR.

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
            if not isinstance(node.init_expression, ArrayLiteral):
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
                wrapper = pennylane.adjoint
            elif mod.modifier.name == "pow":
                if re.search("Literal", mod.argument.__class__.__name__) is not None:
                    wrapper = partial(pennylane.pow, z=mod.argument.value)
                elif mod.argument.name in context["vars"]:
                    wrapper = partial(pennylane.pow, z=self.retrieve_variable(mod.argument.name, context)["val"])
            elif mod.modifier.name == 'ctrl':
                wrapper = partial(pennylane.ctrl, control=gate.keywords["wires"][0:-1])
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
            wires=pennylane.wires.Wires([node.qubits[0].name])
        )

    @staticmethod
    def non_parameterized_gate(gates_dict: dict, node: QASMNode, context: dict):
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
                pennylane.wires.Wires([node.qubits[q].name]) for q in range(len(node.qubits))
            ]
        )
