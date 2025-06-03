from copy import copy
from functools import partial
from typing import Callable

import jax
import pydot

import pennylane as qml

from .base_interpreter import PlxprInterpreter
from .primitives import cond_prim, for_loop_prim, qnode_prim, while_loop_prim


class PlxprGraph:
    def __init__(self):
        self.graph = pydot.Dot(graph_type="digraph", rankdir="TB", overlap="false")
        self.operator_attr = {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "lightblue",
        }
        self.measurement_attr = {
            "shape": "diamond",
            "style": "filled",
            "fillcolor": "lightcoral",
        }
        # Set up data structures for tracking operations and measurements
        self.current_cluster = self.graph
        self.unique_id_counter = 0
        self.last_seen_dynamic_wire = None
        self.dyn_vars = {}
        self.map_wire_to_node_uid = {}
        self.map_operator_outvar_to_mp_invar = {}

    def convert_wires_to_ascii(self, wires: list):
        """Convert a list of wires to an ASCII representation."""
        if not wires:
            return "[]"

        wires_ascii = list(map(self.convert_dyn_wire_to_ascii, wires))
        return wires_ascii

    def get_ascii(self, id):

        from string import ascii_lowercase as letters

        if id == 0:
            return "a"

        val = []
        while id > 0:
            id, r = divmod(id, 26)
            val.append(r)
        return "".join([letters[v] for v in val[::-1]])

    def convert_dyn_wire_to_ascii(self, wire):

        print("Converting wire to ASCII: {}".format(wire))
        if "Var" not in str(wire):
            print(f"Wire {wire} is not a dynamic wire, returning as string")
            return str(wire)

        print("Dynamic tracking so far: ", self.dyn_vars)
        dyn_stuff_counter = len(self.dyn_vars)
        if str(wire) not in self.dyn_vars:
            # never seen before, record and increment
            self.dyn_vars[str(wire)] = self.get_ascii(dyn_stuff_counter)
            return self.dyn_vars[str(wire)]

        print(f"Found {wire} in dyn_vars, returning its ASCII representation")
        return self.dyn_vars[str(wire)]

    def add_operator_node(self, operator_node_uid, label):
        """Add an operator node to the current cluster."""
        self.current_cluster.add_node(
            pydot.Node(operator_node_uid, label=label, **self.operator_attr)
        )

    def add_measurement_node(self, measurement_node_uid, label):
        """Add a measurement node to the current cluster."""
        self.current_cluster.add_node(
            pydot.Node(measurement_node_uid, label=label, **self.measurement_attr)
        )

    def convert_name_to_uid(self, name):
        uid = f"{name}_{self.unique_id_counter}"
        self.unique_id_counter += 1
        return uid

    def __copy__(self):
        copied_graph = PlxprGraph.__new__(PlxprGraph)

        copied_graph.operator_attr = self.operator_attr
        copied_graph.measurement_attr = self.measurement_attr
        copied_graph.graph = self.graph
        copied_graph.current_cluster = self.current_cluster
        copied_graph.unique_id_counter = copy(self.unique_id_counter)
        copied_graph.dyn_vars = self.dyn_vars.copy()
        copied_graph.map_wire_to_node_uid = self.map_wire_to_node_uid.copy()
        copied_graph.last_seen_dynamic_wire = copy(self.last_seen_dynamic_wire)
        copied_graph.map_operator_outvar_to_mp_invar = self.map_operator_outvar_to_mp_invar.copy()

        return copied_graph

    def update(self, other):
        """Update the graph with another PlxprGraph instance."""
        self.unique_id_counter = copy(other.unique_id_counter)
        self.last_seen_dynamic_wire = copy(other.last_seen_dynamic_wire)
        self.dyn_vars |= other.dyn_vars.copy()
        self.map_wire_to_node_uid |= other.map_wire_to_node_uid.copy()
        self.map_operator_outvar_to_mp_invar |= other.map_operator_outvar_to_mp_invar.copy()


class PlxprVisualizerNew(PlxprInterpreter):

    def __init__(self, graph_obj: PlxprGraph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_obj = graph_obj

    def __copy__(self):
        copied_interpreter = PlxprVisualizerNew.__new__(PlxprVisualizerNew)

        copied_interpreter.graph_obj = copy(self.graph_obj)

        return copied_interpreter

    def interpret_operation_eqn(self, eqn):

        print("Interpreting operation equation: ", eqn)
        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            op = eqn.primitive.impl(*invals, **eqn.params)

        # Use the equation parameters to determine the wires
        operator_invars = eqn.invars
        operator_wires = None
        if n_wires := eqn.params["n_wires"]:
            # If the operation has wires, we need to convert them to ASCII
            operator_wires = operator_invars[-n_wires:]
        assert operator_wires
        print("Determined this operator acts on wires: ", operator_wires)

        if isinstance(eqn.outvars[0], jax.core.DropVar):

            # Create node in the graph for the operation
            operator_node_uid = self.graph_obj.convert_name_to_uid("operator_" + op.name)
            op_wires_ascii = self.graph_obj.convert_wires_to_ascii(operator_wires)
            print("Determined the operator wires to be of ascii form: ", op_wires_ascii)
            self.graph_obj.add_operator_node(
                operator_node_uid,
                label=f"{op.name} : {op_wires_ascii}",
            )

            for wire, wire_ascii in zip(op.wires, op_wires_ascii, strict=True):

                if qml.math.is_abstract(wire):
                    print("Detected a dynamic wire in operation: {}".format(wire))
                    self.graph_obj.last_seen_dynamic_wire = wire_ascii
                    # Detected a dynamic wire, need to connect all seen nodes to this node
                    # and purge the tracking of wires
                    for (
                        seen_wire_ascii,
                        seen_node_uid,
                    ) in self.graph_obj.map_wire_to_node_uid.items():
                        cluster_condition_name = self.graph_obj.current_cluster.get_node_list()[
                            0
                        ].get_name()
                        can_not_connect = (
                            "operator_" not in seen_node_uid
                            and "measurement_" not in seen_node_uid
                            and "qnode_"
                            not in seen_node_uid  # work around since it is a source of wires
                            and cluster_condition_name != seen_node_uid
                        )
                        if can_not_connect:
                            # Only connect operators and measurements to the for loop condition
                            continue
                        if not self.graph_obj.graph.get_edge(seen_node_uid, operator_node_uid):
                            if seen_node_uid == operator_node_uid:
                                # Seems to happen it it's a multi wire operation with multiple dynamic wires
                                continue
                            print("Connecting {} to {}".format(seen_node_uid, operator_node_uid))
                            edge = pydot.Edge(seen_node_uid, operator_node_uid, style="dotted")
                            self.graph_obj.graph.add_edge(edge)

                    # Clear the map of wires to node UIDs
                    # This is because we are now treating this node as the source of all future wires
                    # since it is dynamic and could be anything
                    self.graph_obj.map_wire_to_node_uid.clear()
                    self.graph_obj.map_wire_to_node_uid[wire_ascii] = operator_node_uid
                    continue

                # Determine last seen node that had dynamic wires
                last_dynamic_node_uid = self.graph_obj.map_wire_to_node_uid.get(
                    self.graph_obj.last_seen_dynamic_wire
                )
                # Determine the last seen node that acted on this wire
                previous_node_uid_on_this_wire = self.graph_obj.map_wire_to_node_uid.get(str(wire))
                if not previous_node_uid_on_this_wire:
                    # If no previous node was found, use the last dynamic node
                    # since we don't know the concrete wire value
                    previous_node_uid_on_this_wire = last_dynamic_node_uid
                print("Last node on this wire: ", previous_node_uid_on_this_wire)

                if previous_node_uid_on_this_wire == operator_node_uid:
                    # Seems to happen it it's a multi wire operation with dynamic / concrete wire mix
                    continue

                print(
                    "Connecting {} to {}".format(previous_node_uid_on_this_wire, operator_node_uid)
                )
                edge = pydot.Edge(
                    previous_node_uid_on_this_wire,
                    operator_node_uid,
                )
                self.graph_obj.graph.add_edge(edge)

                # Map the wire to the node UID as the new source of this wire
                self.graph_obj.map_wire_to_node_uid[str(wire)] = operator_node_uid

            print("Done.")
            return self.interpret_operation(op)

        # Detected an operator that returns a value.
        # Store information about the operator and the wires
        # it acts on.
        print(
            f"Detected an operation {op} that returns a value, most likely involved in a measurement"
        )
        # Map the outvar to the operator wires so we can later connect it to the measurement
        for outvar in eqn.outvars:
            self.graph_obj.map_operator_outvar_to_mp_invar[str(outvar)] = {"wires": operator_wires}
        return op

    def interpret_measurement_eqn(self, eqn):
        print("Interpreting measurement equation: ", eqn)

        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            mp = eqn.primitive.impl(*invals, **eqn.params)

        # Find out what wires the measurement operator acts on
        mp_invars = eqn.invars
        measurement_operator_wires = self.graph_obj.map_operator_outvar_to_mp_invar.get(
            str(mp_invars[0])
        )["wires"]
        measurement_operator_wires_ascii = self.graph_obj.convert_wires_to_ascii(
            measurement_operator_wires
        )

        mp_name = str(mp._shortname)
        measurement_node_uid = self.graph_obj.convert_name_to_uid("measurement_" + mp_name)
        self.graph_obj.add_measurement_node(
            measurement_node_uid,
            label=f"{mp_name} : {mp.obs.name} : {measurement_operator_wires_ascii}",
        )

        for wire, wire_ascii in zip(mp.obs.wires, measurement_operator_wires_ascii, strict=True):

            if qml.math.is_abstract(wire):
                print("Detected a dynamic wire in measurement operator: {}".format(wire))
                # Detected a dynamic wire, need to connect all seen nodes to this node
                # and purge the tracking of wires
                for (
                    seen_wire_ascii,
                    seen_node_uid,
                ) in self.graph_obj.map_wire_to_node_uid.items():
                    can_not_connect = (
                        "operator_" not in seen_node_uid
                        and "measurement_" not in seen_node_uid
                        and "qnode_"
                        not in seen_node_uid  # work around since it is a source of wires
                    )
                    if can_not_connect:
                        # Only connect operators and measurements to the for loop condition
                        continue
                    if not self.graph_obj.graph.get_edge(seen_node_uid, measurement_node_uid):
                        if seen_node_uid == measurement_node_uid:
                            # Seems to happen it it's a multi wire operation with multiple dynamic wires
                            continue
                        print("Connecting {} to {}".format(seen_node_uid, measurement_node_uid))
                        edge = pydot.Edge(seen_node_uid, measurement_node_uid, style="dotted")
                        self.graph_obj.graph.add_edge(edge)

                continue

            # Determine last seen node that had dynamic wires
            last_dynamic_node_uid = self.graph_obj.map_wire_to_node_uid.get(
                self.graph_obj.last_seen_dynamic_wire
            )
            # Determine the last seen node that acted on this wire
            previous_node_uid_on_this_wire = self.graph_obj.map_wire_to_node_uid.get(str(wire))
            if not previous_node_uid_on_this_wire:
                # If no previous node was found, use the last dynamic node
                # since we don't know the concrete wire value
                previous_node_uid_on_this_wire = last_dynamic_node_uid
            print("Last node on this wire: ", previous_node_uid_on_this_wire)

            can_not_connect = (
                "operator_" not in previous_node_uid_on_this_wire
                and "measurement_" not in previous_node_uid_on_this_wire
                and "qnode_"
                not in previous_node_uid_on_this_wire  # work around since it is a source of wires
            )
            if can_not_connect:
                # Only connect operators and measurements
                previous_node_uid_on_this_wire = last_dynamic_node_uid

            if previous_node_uid_on_this_wire == measurement_node_uid:
                # Seems to happen it it's a multi wire operation with dynamic / concrete wire mix
                print("Skipping connection to self")
                continue

            edge = pydot.Edge(
                previous_node_uid_on_this_wire,
                measurement_node_uid,
                style=(
                    "dotted" if previous_node_uid_on_this_wire == last_dynamic_node_uid else "solid"
                ),
            )
            self.graph_obj.graph.add_edge(edge)

        return self.interpret_measurement(mp)


def jaxpr_to_jaxpr(
    interpreter: "PlxprInterpreter", jaxpr: "jax.extend.core.Jaxpr", consts, *args
) -> "jax.extend.core.ClosedJaxpr":
    """A convenience utility for converting jaxpr to a new jaxpr via an interpreter."""

    f = partial(interpreter.eval, jaxpr, consts)

    return jax.make_jaxpr(f)(*args)


# pylint: disable=too-many-arguments
@PlxprVisualizerNew.register_primitive(qnode_prim)
def handle_qnode(self, *invals, shots, qnode, device, execution_config, qfunc_jaxpr, n_consts):
    """Handle a qnode primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]

    # Create a node for the wires of the device
    wires_node_uid = self.graph_obj.convert_name_to_uid("wires_node")
    wires_node = pydot.Node(
        wires_node_uid,
        label=f"{qnode.device.name} : {str(device.wires.labels)}",
        style="filled",
        shape="rectangle",
        fillcolor="lightgoldenrod2",
    )
    # Add the wires node to the current cluster
    self.graph_obj.current_cluster.add_node(wires_node)

    qnode_node_uid = self.graph_obj.convert_name_to_uid("qnode_cluster")
    qnode_cluster = pydot.Cluster(
        qnode_node_uid,
        label="",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )

    # Find out what arguments the qnode takes and convert them to ASCII
    # This is used to label the qnode node in the graph
    qnode_invars = map(self.graph_obj.convert_dyn_wire_to_ascii, qfunc_jaxpr.invars)
    qnode_node_uid = self.graph_obj.convert_name_to_uid("qnode_node")
    qnode_cluster.add_node(
        pydot.Node(
            qnode_node_uid,
            label=f"{qnode.__name__}({','.join(qnode_invars)})",
            style="dotted",
            shape="rectangle",
        )
    )
    self.graph_obj.current_cluster.add_subgraph(qnode_cluster)

    # Connect the wires node to the qnode node
    self.graph_obj.graph.add_edge(pydot.Edge(wires_node_uid, qnode_node_uid, style="solid"))

    # Record all wires of the device in the graph
    # Treat the qnode_node as the "source" node for these wires
    for wire in device.wires:
        self.graph_obj.map_wire_to_node_uid[str(wire)] = qnode_node_uid

    copy_ = copy(self)
    copy_.graph_obj.current_cluster = qnode_cluster
    new_qfunc_jaxpr = jaxpr_to_jaxpr(copy_, qfunc_jaxpr, consts, *args)

    return qnode_prim.bind(
        *new_qfunc_jaxpr.consts,
        *args,
        shots=shots,
        qnode=qnode,
        device=device,
        execution_config=execution_config,
        qfunc_jaxpr=new_qfunc_jaxpr.jaxpr,
        n_consts=len(new_qfunc_jaxpr.consts),
    )


@PlxprVisualizerNew.register_primitive(for_loop_prim)
def handle_for_loop(
    self, start, stop, step, *args, jaxpr_body_fn, consts_slice, args_slice, abstract_shapes_slice
):
    """Handle a for loop primitive."""
    consts = args[consts_slice]
    init_state = args[args_slice]
    abstract_shapes = args[abstract_shapes_slice]

    # Create a cluster for the for loop
    for_loop_uid = self.graph_obj.convert_name_to_uid("for_loop")
    for_loop_cluster = pydot.Cluster(
        for_loop_uid,
        label="",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    self.graph_obj.current_cluster.add_subgraph(for_loop_cluster)

    # Store for loop invars (specifically the loop index) as a seen dynamic wire
    for_loop_invars = jaxpr_body_fn.invars
    for_loop_invars_ascii = list(map(self.graph_obj.convert_dyn_wire_to_ascii, for_loop_invars))
    self.graph_obj.dyn_vars[str(for_loop_invars[0])] = for_loop_invars_ascii[0]

    # Create a node for the for loop condition
    for_node_uid = self.graph_obj.convert_name_to_uid("condition")
    for_loop_cluster.add_node(
        pydot.Node(
            for_node_uid,
            label=f"for {for_loop_invars_ascii[0]} in range({start}, {stop}, {step})",
            style="dotted",
            shape="rectangle",
            fillcolor="white",
        )
    )

    # Since we created the node, we need to connect all previously seen nodes to this node
    print("seen wire to node uid mapping: {}".format(self.graph_obj.map_wire_to_node_uid))
    for (
        seen_wire_ascii,
        seen_node_uid,
    ) in self.graph_obj.map_wire_to_node_uid.items():
        can_not_connect = (
            "operator_" not in seen_node_uid
            and "measurement_" not in seen_node_uid
            and "qnode_" not in seen_node_uid  # work around since it is a source of wires
        )
        if can_not_connect:
            # Only connect operators and measurements to the for loop condition
            continue

        if not self.graph_obj.graph.get_edge(seen_node_uid, for_node_uid):
            print("Connected {} to {}".format(seen_node_uid, for_node_uid))
            # if seen_node_uid == for_node_uid:
            #     # Seems to happen it it's a multi wire operation with multiple dynamic wires
            #     continue
            edge = pydot.Edge(seen_node_uid, for_node_uid, style="dotted")
            self.graph_obj.graph.add_edge(edge)
        # Since we connected the node, we need to update the mapping
        # to treat this node as the source for the wire
        self.graph_obj.map_wire_to_node_uid[seen_wire_ascii] = for_node_uid

    copy_ = copy(self)
    copy_.graph_obj.current_cluster = for_loop_cluster
    new_jaxpr_body_fn = jaxpr_to_jaxpr(
        copy_, jaxpr_body_fn, consts, *abstract_shapes, start, *init_state
    )
    self.graph_obj.update(copy_.graph_obj)

    consts_slice = slice(0, len(new_jaxpr_body_fn.consts))
    abstract_shapes_slice = slice(consts_slice.stop, consts_slice.stop + len(abstract_shapes))
    args_slice = slice(abstract_shapes_slice.stop, None)
    return for_loop_prim.bind(
        start,
        stop,
        step,
        *new_jaxpr_body_fn.consts,
        *abstract_shapes,
        *init_state,
        jaxpr_body_fn=new_jaxpr_body_fn.jaxpr,
        consts_slice=consts_slice,
        args_slice=args_slice,
        abstract_shapes_slice=abstract_shapes_slice,
    )


@PlxprVisualizerNew.register_primitive(while_loop_prim)
def handle_while_loop(
    self,
    *invals,
    jaxpr_body_fn,
    jaxpr_cond_fn,
    body_slice,
    cond_slice,
    args_slice,
):
    """Handle a while loop primitive."""
    consts_body = invals[body_slice]
    consts_cond = invals[cond_slice]
    init_state = invals[args_slice]

    def get_condition_label(cond_jaxpr):
        # Mapping of internal JAXPR variables to their string representation
        # Start with input variables
        # Example: invars: [a:i32[], b:f32[]]
        # We'll map 'a' to 'carry_0', 'b' to 'carry_1'
        var_map = {cond_jaxpr.invars[idx]: f"invar_{idx}" for idx in range(len(cond_jaxpr.invars))}

        # Process constants (literals)
        # Literals are special in JAXPR: they are `core.Literal` objects.
        # You can get their value via `literal.val`.
        # For now, we'll just use their string representation.
        # We won't map them in `var_map` directly as they're usually just values.

        # This is a simplification. Real JAXPR parsing for general expressions
        # would be a recursive or stack-based approach.
        # For common conditions, it's usually a sequence of comparisons and logical ops.

        # If there's only one equation, it's likely a simple comparison or a logical op.

        if len(cond_jaxpr.eqns) == 1:
            eqn = cond_jaxpr.eqns[0]
            print(eqn)
            primitive = eqn.primitive
            print(primitive)
            print(eqn.invars)
            invars = [
                v.val if isinstance(v, jax.extend.core.Literal) else var_map.get(v, None)
                for v in eqn.invars
            ]
            outvar = eqn.outvars[0]  # Assuming single output for simplicity

            label_parts = []

            # Handle binary operations (comparison, logical AND/OR)
            if primitive.name in ["lt", "gt", "le", "ge", "eq", "ne", "and", "or"]:
                op_symbol = {
                    "lt": "<",
                    "gt": ">",
                    "le": "<=",
                    "ge": ">=",
                    "eq": "==",
                    "ne": "!=",
                    "and": "&&",
                    "or": "||",
                }.get(
                    primitive.name, primitive.name
                )  # Fallback to primitive name

                # Make sure we have enough inputs for a binary op
                if len(invars) >= 2:
                    left_operand = invars[0]
                    right_operand = invars[1]

                    # Update var_map for the output of this equation
                    # This is important if there are chained operations (e.g., (A < B) && C)
                    var_map[outvar] = f"({left_operand} {op_symbol} {right_operand})"
                    label_parts.append(var_map[outvar])
                else:
                    label_parts.append(f"{primitive.name}({', '.join(invars)})")

            elif primitive.name == "not":
                if invars:
                    var_map[outvar] = f"!({invars[0]})"
                    label_parts.append(var_map[outvar])
                else:
                    label_parts.append(f"not()")
            else:
                # Fallback for unknown primitives in condition, or complex expressions
                label_parts.append(f"{primitive.name}({', '.join(invars)})")
                var_map[outvar] = f"{primitive.name}({', '.join(invars)})"  # Store it

            # The final output variable of the cond_jaxpr is the result
            final_result_var = cond_jaxpr.outvars[0]
            return var_map.get(final_result_var, "Complex Condition")

        else:
            # For multiple equations, you'd need to trace the flow more deeply.
            # This can get very complex for arbitrary JAXPRs.
            # A simplified approach might be to just show the last operation,
            # or indicate that it's a "Multi-step Condition".
            # You'd need to build a graph of the cond_jaxpr and then try to
            # find the "root" expression.
            return "Multi-step Condition (expand for details)"

    print(jaxpr_cond_fn)
    print("getting condition label for jaxpr_cond_fn")

    label = get_condition_label(jaxpr_cond_fn)

    # Visualization part
    while_loop_uid = self.graph_obj.convert_name_to_uid("while_loop")
    while_loop_cluster = pydot.Cluster(
        while_loop_uid,
        label="",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    while_node_uid = self.graph_obj.convert_name_to_uid("condition")
    while_loop_cluster.add_node(
        pydot.Node(
            while_node_uid,
            label=f"while {label}",
            style="dotted",
            shape="rectangle",
            fillcolor="white",
        )
    )
    for (
        wire_ascii,
        seen_node_uid,
    ) in self.graph_obj.map_wire_to_node_uid.items():
        # can_not_connect = (
        #     "operator_" not in seen_node_uid
        #     and "measurement_" not in seen_node_uid
        #     and "qnode_" not in seen_node_uid  # work around since it is a source of wires
        # )
        # if can_not_connect:
        #     # Only connect operators and measurements to the while loop condition
        #     continue
        if not self.graph_obj.graph.get_edge(seen_node_uid, while_node_uid):
            print("Connected {} to {}".format(seen_node_uid, while_node_uid))
            if seen_node_uid == while_node_uid:
                # Seems to happen it it's a multi wire operation with multiple dynamic wires
                continue
            edge = pydot.Edge(seen_node_uid, while_node_uid, style="dotted")
            self.graph_obj.graph.add_edge(edge)
        self.graph_obj.map_wire_to_node_uid[wire_ascii] = while_node_uid

    self.graph_obj.current_cluster.add_subgraph(while_loop_cluster)

    copy_ = copy(self)
    copy_.graph_obj.current_cluster = while_loop_cluster
    new_jaxpr_body_fn = jaxpr_to_jaxpr(copy_, jaxpr_body_fn, consts_body, *init_state)
    self.graph_obj.update(copy_.graph_obj)

    copy_ = copy(self)
    copy_.current_cluster = while_loop_cluster
    new_jaxpr_cond_fn = jaxpr_to_jaxpr(copy_, jaxpr_cond_fn, consts_cond, *init_state)
    self.graph_obj.update(copy_.graph_obj)

    body_consts = slice(0, len(new_jaxpr_body_fn.consts))
    cond_consts = slice(body_consts.stop, body_consts.stop + len(new_jaxpr_cond_fn.consts))
    args_slice = slice(cond_consts.stop, None)

    return while_loop_prim.bind(
        *new_jaxpr_body_fn.consts,
        *new_jaxpr_cond_fn.consts,
        *init_state,
        jaxpr_body_fn=new_jaxpr_body_fn.jaxpr,
        jaxpr_cond_fn=new_jaxpr_cond_fn.jaxpr,
        body_slice=body_consts,
        cond_slice=cond_consts,
        args_slice=args_slice,
    )


@PlxprVisualizerNew.register_primitive(cond_prim)
def handle_cond(self, *invals, jaxpr_branches, consts_slices, args_slice):
    """Handle a cond primitive."""
    args = invals[args_slice]

    new_jaxprs = []
    new_consts = []
    new_consts_slices = []
    end_const_ind = len(jaxpr_branches)

    print("Creating cond cluster!")
    cond_node_uid = self.graph_obj.convert_name_to_uid("cond")
    cond_cluster = pydot.Cluster(
        cond_node_uid,
        label="cond",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    self.graph_obj.current_cluster.add_subgraph(cond_cluster)

    branch_count = 0
    temp = {}
    for const_slice, jaxpr in zip(consts_slices, jaxpr_branches):
        consts = invals[const_slice]
        if jaxpr is None:
            new_jaxprs.append(None)
            new_consts_slices.append(slice(0, 0))
        else:
            branch_uid = self.graph_obj.convert_name_to_uid(f"branch_{branch_count}")
            branch_cluster = pydot.Cluster(
                branch_uid,
                label=f"Branch {branch_count}",
                style="dotted",
                fillcolor="lightgrey",
                color="black",
            )
            cond_cluster.add_subgraph(branch_cluster)
            copy_ = copy(self)
            copy_.graph_obj.current_cluster = branch_cluster
            new_jaxpr = jaxpr_to_jaxpr(copy_, jaxpr, consts, *args)
            temp |= copy_.graph_obj.map_wire_to_node_uid.copy()
            new_jaxprs.append(new_jaxpr.jaxpr)
            new_consts.extend(new_jaxpr.consts)
            new_consts_slices.append(slice(end_const_ind, end_const_ind + len(new_jaxpr.consts)))
            end_const_ind += len(new_jaxpr.consts)

            import pprint

            pprint.pprint(vars(copy_.graph_obj))

        branch_count += 1

    self.graph_obj.map_wire_to_node_uid |= temp
    print(self.graph_obj.map_wire_to_node_uid)

    new_args_slice = slice(end_const_ind, None)
    return cond_prim.bind(
        *invals[: len(jaxpr_branches)],
        *new_consts,
        *args,
        jaxpr_branches=new_jaxprs,
        consts_slices=new_consts_slices,
        args_slice=new_args_slice,
    )
