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
            "fillcolor": "lightgreen",
        }
        self.measurement_attr = {
            "shape": "diamond",
            "style": "filled",
            "fillcolor": "lightblue",
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

        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            op = eqn.primitive.impl(*invals, **eqn.params)
        if isinstance(eqn.outvars[0], jax.core.DropVar):
            # Create node in the graph for the operation
            operator_node_uid = self.graph_obj.convert_name_to_uid(op.name)
            print("Wires to node uid mapping: {}".format(self.graph_obj.map_wire_to_node_uid))
            print(f"Creating node {operator_node_uid}")

            print(eqn)
            real_wires_with_id = eqn.invars[-eqn.params["n_wires"] :]
            print("real wires with id: {}".format(real_wires_with_id))

            op_wires_ascii = self.graph_obj.convert_wires_to_ascii(real_wires_with_id)
            print("real wires but ascii: {}".format(op_wires_ascii))
            print(
                "adding node to current cluster: {}".format(
                    self.graph_obj.current_cluster.get_label()
                )
            )
            self.graph_obj.current_cluster.add_node(
                pydot.Node(
                    operator_node_uid,
                    label=f"{op.name} : {op_wires_ascii}",
                    **self.graph_obj.operator_attr,
                )
            )

            for wire, wire_ascii in zip(op.wires, op_wires_ascii, strict=True):

                if qml.math.is_abstract(wire):
                    print(
                        "Updating last seen dynamic wire to: {} (which is ascii {})".format(
                            wire, wire_ascii
                        )
                    )
                    self.graph_obj.last_seen_dynamic_wire = wire_ascii
                    # Detected a dynamic wire, need to connect all seen nodes to this node
                    for _, seen_node_uid in self.graph_obj.map_wire_to_node_uid.items():
                        if not self.graph_obj.graph.get_edge(seen_node_uid, operator_node_uid):
                            print("Connected {} to {}".format(seen_node_uid, operator_node_uid))
                            if seen_node_uid == operator_node_uid:
                                # Seems to happen it it's a multi wire operation with multiple dynamic wires
                                continue
                            edge = pydot.Edge(seen_node_uid, operator_node_uid, style="dotted")
                            self.graph_obj.graph.add_edge(edge)
                    self.graph_obj.map_wire_to_node_uid.clear()
                    self.graph_obj.map_wire_to_node_uid[wire_ascii] = operator_node_uid
                    continue

                print(self.graph_obj.map_wire_to_node_uid)
                last_dynamic_node_uid = self.graph_obj.map_wire_to_node_uid.get(
                    self.graph_obj.last_seen_dynamic_wire, None
                )
                previous_node_uid_on_this_wire = self.graph_obj.map_wire_to_node_uid.get(
                    str(wire), last_dynamic_node_uid
                )

                print(
                    "Connecting {} to {}".format(previous_node_uid_on_this_wire, operator_node_uid)
                )
                if previous_node_uid_on_this_wire == operator_node_uid:
                    # Seems to happen it it's a multi wire operation with dynamic / concrete wire mix
                    continue
                edge = pydot.Edge(
                    previous_node_uid_on_this_wire,
                    operator_node_uid,
                    style=(
                        "dotted"
                        if previous_node_uid_on_this_wire == last_dynamic_node_uid
                        else "solid"
                    ),
                )
                self.graph_obj.graph.add_edge(edge)

                # Map the wire to the node UID
                self.graph_obj.map_wire_to_node_uid[str(wire)] = operator_node_uid

            return self.interpret_operation(op)

        # Detected an operator that returns a value.
        # Store information about the operator and the wires
        # it acts on.
        print(
            f"Detected an operation {op} that returns a value, most likely involved in a measurement"
        )
        print(eqn.outvars, eqn.invars, eqn.params)
        for outvar in eqn.outvars:
            self.graph_obj.map_operator_outvar_to_mp_invar[str(outvar)] = {
                "wires": eqn.invars[-eqn.params["n_wires"] :]
            }
        return op

    def interpret_measurement_eqn(self, eqn):
        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            mp = eqn.primitive.impl(*invals, **eqn.params)
        print(f"Detected a measurement {mp} being applied on {mp.obs}, with wires {mp.obs.wires}")
        print(eqn.outvars, eqn.invars, eqn.params)
        measurement_node_uid = self.graph_obj.convert_name_to_uid(str(mp._shortname))

        real_wires = self.graph_obj.map_operator_outvar_to_mp_invar.get(str(eqn.invars[0]))["wires"]
        print("Real wires for measurement: ", real_wires)
        mp_wires_ascii = self.graph_obj.convert_wires_to_ascii(real_wires)
        self.graph_obj.current_cluster.add_node(
            pydot.Node(
                measurement_node_uid,
                label=f"{mp._shortname} : {mp.obs.name} : {mp_wires_ascii}",
                **self.graph_obj.measurement_attr,
            )
        )

        for wire, wire_ascii in zip(mp.obs.wires, mp_wires_ascii, strict=True):

            if qml.math.is_abstract(wire):
                print("Detected a dynamic wire in measurement observable: {}".format(wire))
                # self.last_seen_dynamic_wire = wire_ascii
                # Detected a dynamic wire, need to connect all seen nodes to this node
                for _, seen_node_uid in self.graph_obj.map_wire_to_node_uid.items():
                    if not self.graph_obj.graph.get_edge(seen_node_uid, measurement_node_uid):
                        print("Connected {} to {}".format(seen_node_uid, measurement_node_uid))
                        if seen_node_uid == measurement_node_uid:
                            # Seems to happen it it's a multi wire operation with multiple dynamic wires
                            continue
                        edge = pydot.Edge(seen_node_uid, measurement_node_uid, style="dotted")
                        self.graph_obj.graph.add_edge(edge)
                # self.graph_obj.map_wire_to_node_uid.clear()
                # self.graph_obj.map_wire_to_node_uid[wire_ascii] = measurement_node_uid
                continue

            print("Inside measurement and mapping dict: ", self.graph_obj.map_wire_to_node_uid)
            print("Dynamic wire seen: ", self.graph_obj.dyn_vars)
            print("Last seen dynamic wire: ", self.graph_obj.last_seen_dynamic_wire)
            last_dynamic_node_uid = self.graph_obj.map_wire_to_node_uid.get(
                self.graph_obj.last_seen_dynamic_wire, None
            )
            previous_node_uid_on_this_wire = self.graph_obj.map_wire_to_node_uid.get(
                str(wire), last_dynamic_node_uid
            )

            print(
                "Connecting {} to {}, with last dynamic node {}".format(
                    previous_node_uid_on_this_wire, measurement_node_uid, last_dynamic_node_uid
                )
            )
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

        # Map the wire to the node UID
        # self.graph_obj.map_wire_to_node_uid[str(wire)] = measurement_node_uid

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

    # Visualization part
    wires_node_uid = self.graph_obj.convert_name_to_uid("wires")
    wires_node = pydot.Node(
        wires_node_uid,
        label=f"{qnode.device.name} : {str(device.wires.labels)}",
        style="filled",
        fillcolor="red",
    )
    self.graph_obj.current_cluster.add_node(wires_node)
    for wire in device.wires:
        self.graph_obj.map_wire_to_node_uid[str(wire)] = wires_node_uid

    invars = map(self.graph_obj.convert_dyn_wire_to_ascii, qfunc_jaxpr.invars)
    qnode_node_uid = self.graph_obj.convert_name_to_uid(qnode.__name__)
    qnode_cluster = pydot.Cluster(
        qnode_node_uid,
        label=f"{qnode.__name__}({','.join(invars)})",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    self.graph_obj.current_cluster.add_subgraph(qnode_cluster)

    print(self.graph_obj.current_cluster.get_label())
    copy_ = copy(self)
    print(copy_.graph_obj.current_cluster.get_label())
    copy_.graph_obj.current_cluster = qnode_cluster
    print(copy_.graph_obj.current_cluster.get_label())
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

    # Visualization part
    for_loop_uid = self.graph_obj.convert_name_to_uid("for_loop")
    loop_var = jaxpr_body_fn.invars
    print(consts)
    loop_var_ascii = self.graph_obj.convert_wires_to_ascii(loop_var)
    print("Something inside for loop depends on outside variable: {}".format(consts))
    self.graph_obj.dyn_vars[str(loop_var[0])] = loop_var_ascii[0]
    for_loop_cluster = pydot.Cluster(
        for_loop_uid,
        label=f"for {loop_var_ascii[0]} in range({start}, {stop}, {step})",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    self.graph_obj.current_cluster.add_subgraph(for_loop_cluster)

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

    # Visualization part
    while_loop_uid = self.graph_obj.convert_name_to_uid("while_loop")
    while_loop_cluster = pydot.Cluster(
        while_loop_uid,
        label="while",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
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
