from copy import copy
from functools import partial
from typing import Sequence

import jax
from jax.extend import core

from pennylane.operation import Operator
from pennylane.queuing import QueuingManager

from ..base_interpreter import PlxprInterpreter
from ..primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    qnode_prim,
    while_loop_prim,
)
from .pydot_graph_builder import (
    AdjointCluster,
    ControlCluster,
    ControlFlowCluster,
    DeviceNode,
    MeasurementNode,
    OperatorNode,
    PyDotGraphBuilder,
    QNodeCluster,
)


class PlxprVisualizer(PlxprInterpreter):
    """A visualizer for PennyLane expressions.

    This class extends the `PlxprInterpreter` to provide visualization capabilities for PennyLane expressions.
    It can be used to render the structure of quantum circuits and operations in a human-readable format.
    """

    def __init__(self, plxpr_graph: PyDotGraphBuilder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store the graph representation of the circuit
        self.plxpr_graph = plxpr_graph

        # holds ASCII representations of the environment
        self._env_ascii = {}

        self._cond_eqns = {}

    def __copy__(self):
        """Create a copy of the visualizer with the same graph."""
        new_visualizer = PlxprVisualizer(copy(self.plxpr_graph))

        new_visualizer._env_ascii = copy(self._env_ascii)

        return new_visualizer

    def read_ascii(self, var):
        """Read the ASCII representation of a variable."""
        if isinstance(var, core.Literal):
            # If the variable is a literal, return its value directly
            return var.val
        return self._env_ascii[var]

    def _bold_str(self, text: str) -> str:
        """Returns a bold representation of the text for visualization purposes."""
        return f"<b>{text}</b>"

    def _int_to_ascii(self, num: int) -> str:
        """Converts an integer to its ASCII representation."""

        from string import ascii_lowercase as letters

        if num == 0:
            return "a"

        val = []
        while num > 0:
            num, r = divmod(num, 26)
            val.append(r)
        return "".join([letters[v].lower() for v in val[::-1]])

    def _convert_var_to_ascii(self, var: core.Var) -> str:
        """Converts a JAX variable to its ASCII representation."""

        var_id = id(var)
        if var_id in self._env_ascii:
            return self._env_ascii[var_id]

        num_of_unique_ascii = len(set(list(self._env_ascii.values())))
        ascii = self._int_to_ascii(num_of_unique_ascii)
        return ascii

    def eval(self, jaxpr: core.Jaxpr, consts: Sequence, *args, ascii_context: dict = None) -> list:
        print(self.plxpr_graph.current_cluster.get_name())

        self._env = {}
        if ascii_context:
            # If an ASCII context is provided, use it to update the known
            # mappings of variables to their ASCII representations.
            self._env_ascii |= ascii_context
        self.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            if ascii_context and invar in ascii_context:
                # If the variable is in the ASCII context, use that representation
                invar_ascii = ascii_context[invar]
            else:
                invar_ascii = self._convert_var_to_ascii(invar)
            self._env[invar] = arg
            self._env_ascii[invar] = invar_ascii

        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            if ascii_context and constvar in ascii_context:
                # If the constant variable is in the ASCII context, use that representation
                constvar_ascii = ascii_context[constvar]
            else:
                constvar_ascii = self._convert_var_to_ascii(constvar)
            self._env[constvar] = const
            self._env_ascii[constvar] = constvar_ascii

        for eqn in jaxpr.eqns:
            primitive = eqn.primitive
            custom_handler = self._primitive_registrations.get(primitive, None)

            if custom_handler:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = custom_handler(self, *invals, **eqn.params, eqn=eqn)
            elif getattr(primitive, "prim_type", "") == "operator":
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = self.interpret_operation_eqn(eqn)
            elif getattr(primitive, "prim_type", "") == "measurement":
                outvals = self.interpret_measurement_eqn(eqn)
            else:
                print(
                    f"Warning: No custom handler for primitive {primitive.name}. Using default handling."
                )
                # These primitives might be used in cond expressions so we
                # need to record them down
                for outvar in eqn.outvars:
                    self._cond_eqns[outvar] = eqn

                invals = [self.read(invar) for invar in eqn.invars]
                subfuns, params = primitive.get_bind_params(eqn.params)
                outvals = primitive.bind(*subfuns, *invals, **params)

            if not primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                self._env[outvar] = outval

        # Read the final result of the Jaxpr from the environment
        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if isinstance(outval, Operator):
                outvals.append(self.interpret_operation(outval))
            else:
                outvals.append(outval)
        self.cleanup()
        self._env = {}
        return outvals

    def interpret_operation_eqn(self, eqn: core.JaxprEqn):
        """Interprets a Jaxpr equation that represents a Quantum operation."""

        invals = (self.read(invar) for invar in eqn.invars)
        invals_ascii = [self.read_ascii(invar) for invar in eqn.invars]

        with QueuingManager.stop_recording():
            # Use the primitive's implementation to interpret the operation
            op = eqn.primitive.impl(*invals, **eqn.params)

        if isinstance(eqn.outvars[0], jax.core.DropVar):
            # This is a stand-alone operation that does not return a value
            num_wires = len(op.wires)

            # Invals could contain multiple things so we need to extract out the wires
            op_wires = invals_ascii[-num_wires:]
            op_misc = invals_ascii[:-num_wires]

            # Create operator node
            op_node = OperatorNode(
                wires=op_wires,
                name=op.name,
                label=f"<{op.name} {', '.join(list(map(str, op_misc)))}: ({', '.join(list(map(self._bold_str, op_wires)))})>",
            )
            self.plxpr_graph.add_quantum_node_to_graph(
                op_node,
                auto_connect=True,
            )

            return self.interpret_operation(op)

        if isinstance(self.plxpr_graph.current_cluster, (ControlCluster, AdjointCluster)):
            # This is a stand-alone operation that does not return a value
            num_wires = len(op.wires)

            # Invals could contain multiple things so we need to extract out the wires
            op_wires = invals_ascii[-num_wires:]
            op_misc = invals_ascii[:-num_wires]

            # Create operator node
            op_node = OperatorNode(
                wires=op_wires,
                name=op.name,
                label=f"<{op.name} {', '.join(list(map(str, op_misc)))}: ({', '.join(list(map(self._bold_str, op_wires)))})>",
            )
            self.plxpr_graph.add_quantum_node_to_graph(
                op_node,
                auto_connect=True,
            )

            return self.interpret_operation(op)
        # This operator is part of a bigger thing and is returning something
        # to be processed later (i.e. control or measurement)
        assert len(eqn.outvars) == 1, "Expected a single output variable for the operation."
        self._env_ascii[eqn.outvars[0]] = invals_ascii[0]
        return op

    def interpret_measurement_eqn(self, eqn: core.JaxprEqn):
        """Interprets a Jaxpr equation that represents a measurement operation."""

        invals = (self.read(invar) for invar in eqn.invars)
        invals_ascii = [self.read_ascii(invar) for invar in eqn.invars]

        with QueuingManager.stop_recording():
            # Use the primitive's implementation to interpret the measurement
            mp = eqn.primitive.impl(*invals, **eqn.params)

        mp_wires = invals_ascii
        # TODO: Shouldn't have _shortname here but its conveninent
        mp_name = mp._shortname
        if hasattr(mp, "obs") and mp.obs:
            # If the measurement has an observable, we can use that to label the node
            op_name = mp.obs.name
        else:
            # If the measurement does not have an observable, we use the primitive name
            op_name = ""

        meas_node = MeasurementNode(
            wires=mp_wires,
            name=mp_name,
            label=f"<{mp_name} : {op_name}({', '.join(list(map(self._bold_str, mp_wires)))})>",
        )
        self.plxpr_graph.add_quantum_node_to_graph(
            meas_node,
            auto_connect=True,
        )

        return self.interpret_measurement(mp)


@PlxprVisualizer.register_primitive(qnode_prim)
def handle_qnode(
    self, *invals, shots, qnode, device, execution_config, qfunc_jaxpr, n_consts, eqn=None
):
    """Handle a qnode primitive."""

    # Carry over context from the `eval` that called this primitive.
    ascii_context = {}
    inner_invars = qfunc_jaxpr.invars
    outer_invars = eqn.invars[n_consts:]
    for inner_invar, outer_invar in zip(inner_invars, outer_invars, strict=True):
        ascii_context[inner_invar] = self.read_ascii(outer_invar)

    inner_constvars = qfunc_jaxpr.constvars
    outer_constvars = eqn.invars[:n_consts]
    for inner_constvar, outer_constvar in zip(inner_constvars, outer_constvars, strict=True):
        ascii_context[inner_constvar] = self.read_ascii(outer_constvar)

    consts = invals[:n_consts]
    args = invals[n_consts:]

    # Create QNode cluster
    qnode_cluster = QNodeCluster(info_label="")
    self.plxpr_graph.add_cluster_to_graph(qnode_cluster)

    # Create a node for the QNode
    args_ascii = [self.read_ascii(invar) for invar in self._env_ascii]

    # We don't know if the arguments could be used as wires ...
    all_wires = device.wires + args_ascii
    qnode_node = OperatorNode(
        wires=all_wires,
        name=qnode.__name__,
        label=f"{qnode.__name__}({', '.join(list(map(str, args_ascii)))})",
        shape="rectangle",
        fillcolor="cornsilk",
        color="cornsilk4",
    )

    # Create device node just for visualization purposes
    device = qnode.device
    device_node = DeviceNode(
        wires=[],
        name=device.name,
        label=f"{device.name} : {device.wires.labels}",
    )

    # Don't need to connect device node to anything and we don't
    # want to use it as a source of wires
    self.plxpr_graph.add_quantum_node_to_graph(
        device_node,
        cluster=qnode_cluster,
        auto_connect=False,
    )

    # We want this node to be the source of all_wires so we connect
    self.plxpr_graph.add_quantum_node_to_graph(
        qnode_node,
        cluster=qnode_cluster,
        auto_connect=True,
    )

    self.plxpr_graph.add_edge(
        device_node,
        qnode_node,
        color="cornsilk4",
    )

    interpreter_copy = copy(self)
    interpreter_copy.plxpr_graph.current_cluster = qnode_cluster
    return interpreter_copy.eval(qfunc_jaxpr, consts, *args, ascii_context=ascii_context)


@PlxprVisualizer.register_primitive(for_loop_prim)
def handle_for_loop(
    self,
    start,
    stop,
    step,
    *args,
    jaxpr_body_fn,
    consts_slice,
    args_slice,
    abstract_shapes_slice,
    eqn=None,
):
    """Handle a for loop primitive."""
    consts = args[consts_slice]
    init_state = args[args_slice]
    abstract_shapes = args[abstract_shapes_slice]

    ascii_context = {}

    # Labelled as invar but these are actually constants
    outer_consts = eqn.invars[3:]  # Skip start, stop, step
    inner_consts = jaxpr_body_fn.constvars
    for inner_const, outer_const in zip(inner_consts, outer_consts, strict=True):
        recorded_ascii = self.read_ascii(outer_const)
        ascii_context[inner_const] = recorded_ascii

    # Will encounter a new variable (the loop variable)
    inner_loop_var = jaxpr_body_fn.invars[0]
    inner_loop_var_ascii = self._convert_var_to_ascii(inner_loop_var)
    ascii_context[inner_loop_var] = inner_loop_var_ascii
    # Need to add to _env_ascii so the counter is correct
    self._env_ascii[inner_loop_var] = inner_loop_var_ascii

    start_label, stop_label, step_label = map(self.read_ascii, eqn.invars[:3])

    # Create a cluster for the for loop
    label = f"<for {self._bold_str(inner_loop_var_ascii)} in range({self._bold_str(start_label)}, {self._bold_str(stop_label)}, {self._bold_str(step_label)})>"
    for_loop_cluster = ControlFlowCluster(info_label=label)
    self.plxpr_graph.add_cluster_to_graph(for_loop_cluster)

    interpreter_copy = copy(self)
    interpreter_copy.plxpr_graph.current_cluster = for_loop_cluster
    return interpreter_copy.eval(
        jaxpr_body_fn, consts, *abstract_shapes, start, *init_state, ascii_context=ascii_context
    )


@PlxprVisualizer.register_primitive(while_loop_prim)
def handle_while_loop(
    self,
    *invals,
    jaxpr_body_fn,
    jaxpr_cond_fn,
    body_slice,
    cond_slice,
    args_slice,
    eqn=None,
):
    """Handle a while loop primitive."""

    consts_body = invals[body_slice]
    consts_cond = invals[cond_slice]
    init_state = invals[args_slice]

    ascii_context = {}

    # Handle context for any constants being passed to the body
    outer_consts = eqn.invars[body_slice]
    inner_body_consts = jaxpr_body_fn.constvars
    for inner_const, outer_const in zip(inner_body_consts, outer_consts, strict=True):
        ascii_context[inner_const] = self.read_ascii(outer_const)

    # # The cond variable could be used in the body, so we need to record it
    for cond_invar, body_invar in zip(jaxpr_cond_fn.invars, jaxpr_body_fn.invars, strict=True):
        ascii = self._convert_var_to_ascii(cond_invar)
        ascii_context[body_invar] = ascii
        # Need to add to _env_ascii so the counter is correct
        self._env_ascii[body_invar] = ascii

    # Get label for the while loop condition
    cond_invar_ascii = ascii_context.get(jaxpr_body_fn.invars[0], "unknown condition")
    prim_to_sym = {
        "lt": "&lt;",
        "le": "&le;",
        "gt": "&gt;",
        "ge": "&ge;",
        "eq": "==",
        "ne": "!=",
    }

    label = f"<while ({self._bold_str(cond_invar_ascii)} {prim_to_sym.get(jaxpr_cond_fn.eqns[-1].primitive.name, 'unknown primitive')} {jaxpr_cond_fn.eqns[-1].invars[-1]})>"
    while_loop_cluster = ControlFlowCluster(info_label=label)
    self.plxpr_graph.add_cluster_to_graph(while_loop_cluster)

    interpreter_copy = copy(self)
    interpreter_copy.plxpr_graph.current_cluster = while_loop_cluster
    return interpreter_copy.eval(
        jaxpr_body_fn,
        consts_body,
        *init_state,
        ascii_context=ascii_context,
    )


def jaxpr_to_jaxpr(
    interpreter: PlxprVisualizer, jaxpr: core.Jaxpr, consts, *args
) -> core.ClosedJaxpr:
    """A convenience utility for converting jaxpr to a new jaxpr via an interpreter."""

    f = partial(interpreter.eval, jaxpr, consts)

    return jax.make_jaxpr(f)(*args)


@PlxprVisualizer.register_primitive(cond_prim)
def handle_cond(self, *invals, jaxpr_branches, consts_slices, args_slice, eqn=None):
    """Handle a cond primitive."""
    args = invals[args_slice]

    new_jaxprs = []
    new_consts = []
    new_consts_slices = []
    end_const_ind = len(jaxpr_branches)

    cond_eqn_invars = eqn.invars[: len(jaxpr_branches)]
    cond_eqn_invars[-1] = None  # This is the "else" branch
    jaxpr_cond_eqns = [self._cond_eqns.get(invar, None) for invar in cond_eqn_invars]

    # Create cond cluster
    cond_cluster = ControlFlowCluster(info_label="")
    self.plxpr_graph.add_cluster_to_graph(cond_cluster)

    label_map = {1: "if", 2: "elif", 0: "else"}
    get_branch_label = lambda x: label_map.get(x % len(jaxpr_branches), "unknown")
    branch_counter = 1

    # Since each jaxpr is a "parallel" branch
    # we need to keep track of the wires to nodes mapping for each branch
    branch_wires_to_nodes = {}
    for const_slice, jaxpr, cond_jaxpr in zip(
        consts_slices, jaxpr_branches, jaxpr_cond_eqns, strict=True
    ):

        consts = invals[const_slice]
        if jaxpr is None:
            new_jaxprs.append(None)
            new_consts_slices.append(slice(0, 0))
        else:

            # Create branch cluster
            branch_label = f"<{get_branch_label(branch_counter)} ({cond_jaxpr or ''})>"
            branch_cluster = ControlFlowCluster(info_label=branch_label)
            self.plxpr_graph.add_cluster_to_graph(branch_cluster, graph=cond_cluster)

            interpreter_copy = copy(self)
            # Copy so we don't modify the original interpreter's graph
            interpreter_copy.plxpr_graph.wires_to_nodes = copy(self.plxpr_graph.wires_to_nodes)
            interpreter_copy.plxpr_graph.current_cluster = branch_cluster

            new_jaxpr = jaxpr_to_jaxpr(interpreter_copy, jaxpr, consts, *args)

            # Record what we saw so we know how to connect things after the cond
            for wire, node in interpreter_copy.plxpr_graph.wires_to_nodes.items():
                if wire not in branch_wires_to_nodes:
                    branch_wires_to_nodes[wire] = []
                if branch_wires_to_nodes[wire] != node:
                    branch_wires_to_nodes[wire].extend(node)

            new_jaxprs.append(new_jaxpr.jaxpr)
            new_consts.extend(new_jaxpr.consts)
            new_consts_slices.append(slice(end_const_ind, end_const_ind + len(new_jaxpr.consts)))
            end_const_ind += len(new_jaxpr.consts)

        branch_counter += 1

    # Update the interpreter's graph with the wires to nodes mapping
    self.plxpr_graph.wires_to_nodes = branch_wires_to_nodes
    new_args_slice = slice(end_const_ind, None)
    return cond_prim.bind(
        *invals[: len(jaxpr_branches)],
        *new_consts,
        *args,
        jaxpr_branches=new_jaxprs,
        consts_slices=new_consts_slices,
        args_slice=new_args_slice,
    )


@PlxprVisualizer.register_primitive(ctrl_transform_prim)
def handle_ctrl_transform(
    self, *invals, n_control, jaxpr, control_values, work_wires, n_consts, eqn=None
):
    """Interpret a ctrl transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]

    ascii_context = {}

    controls = eqn.invars[-n_control:]
    # TODO: Not sure why the slicing is off here, but it works for now
    outer_invars = eqn.invars[-n_consts - n_control : -n_control]
    outer_constants = eqn.invars[: -n_consts - n_control]

    controls_ascii = [self.read_ascii(control) for control in controls]

    if not outer_invars:
        # This is if you give an operator type to qml.ctrl
        for outer_const, inner_const in zip(outer_invars, jaxpr.constvars, strict=True):
            assert (
                inner_const not in ascii_context
            ), f"Variable {inner_const} already exists in ascii_context."
            ascii_context[inner_const] = self.read_ascii(outer_const)

        for outer_var, inner_var in zip(outer_constants, jaxpr.invars, strict=True):
            assert (
                inner_var not in ascii_context
            ), f"Variable {inner_var} already exists in ascii_context."
            ascii_context[inner_var] = self.read_ascii(outer_var)

    else:
        # This is if you give a qfunc to qml.ctrl
        for outer_const, inner_const in zip(outer_constants, jaxpr.constvars, strict=True):
            assert (
                inner_const not in ascii_context
            ), f"Variable {inner_const} already exists in ascii_context."
            ascii_context[inner_const] = self.read_ascii(outer_const)

        for outer_var, inner_var in zip(outer_invars, jaxpr.invars, strict=True):
            assert (
                inner_var not in ascii_context
            ), f"Variable {inner_var} already exists in ascii_context."
            ascii_context[inner_var] = self.read_ascii(outer_var)

    # Create ctrl cluster
    ctrl_cluster = ControlCluster(
        info_label=f"<control : {', '.join(map(self._bold_str,controls_ascii))}>"
    )
    self.plxpr_graph.add_cluster_to_graph(ctrl_cluster)

    interpreter_copy = copy(self)
    interpreter_copy.plxpr_graph.current_cluster = ctrl_cluster
    _ = interpreter_copy.eval(jaxpr, consts, *args, ascii_context=ascii_context)

    return []


@PlxprVisualizer.register_primitive(adjoint_transform_prim)
def handle_adjoint_transform(self, *invals, jaxpr, lazy, n_consts, eqn=None):
    """Interpret an adjoint transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]

    ascii_context = {}

    outer_invars = eqn.invars[n_consts:]
    outer_constants = eqn.invars[:n_consts]

    if not outer_invars:
        # This is if you give an operator type to qml.ctrl
        for outer_const, inner_const in zip(outer_invars, jaxpr.constvars, strict=True):
            assert (
                inner_const not in ascii_context
            ), f"Variable {inner_const} already exists in ascii_context."
            ascii_context[inner_const] = self.read_ascii(outer_const)

        for outer_var, inner_var in zip(outer_constants, jaxpr.invars, strict=True):
            assert (
                inner_var not in ascii_context
            ), f"Variable {inner_var} already exists in ascii_context."
            ascii_context[inner_var] = self.read_ascii(outer_var)

    else:
        # This is if you give a qfunc to qml.ctrl
        for outer_const, inner_const in zip(outer_constants, jaxpr.constvars, strict=True):
            assert (
                inner_const not in ascii_context
            ), f"Variable {inner_const} already exists in ascii_context."
            ascii_context[inner_const] = self.read_ascii(outer_const)

        for outer_var, inner_var in zip(outer_invars, jaxpr.invars, strict=True):
            assert (
                inner_var not in ascii_context
            ), f"Variable {inner_var} already exists in ascii_context."
            ascii_context[inner_var] = self.read_ascii(outer_var)

    adjoint_cluster = AdjointCluster(info_label="adjoint")
    self.plxpr_graph.add_cluster_to_graph(adjoint_cluster)

    interpreter_copy = copy(self)
    interpreter_copy.plxpr_graph.current_cluster = adjoint_cluster
    _ = interpreter_copy.eval(jaxpr, consts, *args, ascii_context=ascii_context)

    return []
