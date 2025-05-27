from copy import copy
from functools import partial
from typing import Callable

import jax
import pydot

import pennylane as qml

from .base_interpreter import PlxprInterpreter
from .primitives import for_loop_prim, qnode_prim


class PlxprVisualizer:
    wires: dict
    _primitive_registrations: dict["jax.extend.core.Primitive", Callable] = {}

    def __init_subclass__(cls) -> None:
        cls._primitive_registrations = copy(cls._primitive_registrations)

    @classmethod
    def register_primitive(cls, primitive) -> Callable[[Callable], Callable]:
        def decorator(f: Callable) -> Callable:
            cls._primitive_registrations[primitive] = f
            return f

        return decorator

    def get_ascii(self, id):

        from string import ascii_lowercase as letters

        if id == 0:
            return "a"

        val = []
        while id > 0:
            id, r = divmod(id, 26)
            val.append(r)
        return "".join([letters[v] for v in val[::-1]])

    def convert_id_to_ascii(self, id: str):

        if "Var" not in id:
            return id

        if id not in self.seen_dynamic_ids:
            # never seen before, record and increment
            self.seen_dynamic_ids[id] = self.get_ascii(self.id_counter)
            self.id_counter += 1
            return self.seen_dynamic_ids[id]

        return self.seen_dynamic_ids[id]

    def __init__(self):
        self.interpreter = qml.capture.PlxprInterpreter()

        self.unique_id = 0
        self.wires = {}
        self.id_counter = 0
        self.wires_source = {}
        self.seen_dynamic_ids = {}
        self.last_seen_dynamic_wire = None

        graph_attr = {
            "rankdir": "TB",
            "overlap": "false",
        }
        self.graph = pydot.Dot(graph_type="digraph", **graph_attr)
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
        super().__init__()

    def convert_name_to_uid(self, name):
        uid = f"{name}_{self.unique_id}"
        self.unique_id += 1
        return uid

    def parse(self, jaxpr, consts, *args, cluster=None, interpreter=None) -> list:
        if not interpreter:
            interpreter = self.interpreter

        if not cluster:
            cluster = self.graph

        interpreter._env = {}
        interpreter.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            interpreter._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            interpreter._env[constvar] = const

        for eqn in jaxpr.eqns:

            primitive = eqn.primitive
            custom_handler = self._primitive_registrations.get(primitive, None)

            if custom_handler:
                invals = [interpreter.read(invar) for invar in eqn.invars]
                custom_handler(self, cluster, *invals, **eqn.params)

            elif getattr(primitive, "prim_type", "") == "operator":
                operator_wires = eqn.invars[-eqn.params["n_wires"] :]

                if str(eqn.outvars) != "[_]":
                    # store information about the operator
                    # and the wires it is connected to
                    # and will be used to connect to the measurement
                    for outvar in eqn.outvars:
                        self.wires_source[str(outvar)] = {
                            "name": primitive.name,
                            "eqn": eqn,
                        }
                    continue

                operator_uid = self.convert_name_to_uid(primitive.name)
                operator_wires_ascii = list(map(self.convert_id_to_ascii, map(str, operator_wires)))
                cluster.add_node(
                    pydot.Node(
                        operator_uid,
                        label=f"{primitive.name} : {operator_wires_ascii}",
                        **self.operator_attr,
                    )
                )

                for op_wire in operator_wires:

                    if "Var" in str(op_wire):
                        print("encountered dynamic wire", op_wire)
                        self.last_seen_dynamic_wire = op_wire

                        # connect all seen nodes to this operator
                        for seen_wire, _ in self.wires.items():
                            seen_node = self.wires[str(seen_wire)]
                            if not self.graph.get_edge(seen_node, operator_uid):
                                # only add dotted edge if not already connected
                                print("connecting", self.wires[str(seen_wire)], operator_uid)
                                edge = pydot.Edge(seen_node, operator_uid, style="dotted")
                                self.graph.add_edge(edge)

                        # clear seen wires and replace with dynamic wire "bus"
                        self.wires.clear()
                        self.wires[str(op_wire)] = operator_uid
                        continue

                    last_dynamic_node = self.wires.get(str(self.last_seen_dynamic_wire), None)
                    previous_op = self.wires.get(str(op_wire), last_dynamic_node)
                    print("connecting", previous_op, operator_uid)
                    edge = pydot.Edge(previous_op, operator_uid)
                    self.graph.add_edge(edge)

                    # record this operator node as last seen on this wire
                    self.wires[str(op_wire)] = operator_uid

            elif getattr(primitive, "prim_type", "") == "measurement":
                measurement_uid = self.convert_name_to_uid(primitive.name)
                node_involved = self.wires_source[str(eqn.invars[0])]
                name_op = node_involved["name"]
                eqn_op = node_involved["eqn"]
                cluster.add_node(
                    pydot.Node(
                        measurement_uid,
                        label=f"{primitive.name} : {name_op} : {eqn_op.invars}",
                        **self.measurement_attr,
                    )
                )

                source_node = self.wires.get(str(eqn_op.invars[0]), None)
                print(
                    "connecting",
                    source_node,
                    measurement_uid,
                )
                if not source_node:
                    # connect all seen nodes to this measurement
                    for seen_wire, _ in self.wires.items():
                        print("connecting", self.wires[str(seen_wire)], measurement_uid)
                        seen_node = self.wires[str(seen_wire)]
                        edge = pydot.Edge(seen_node, measurement_uid, style="dotted")
                        self.graph.add_edge(edge)
                    continue
                edge = pydot.Edge(
                    self.wires.get(str(eqn_op.invars[0]), list(self.wires.values())[-1]),
                    measurement_uid,
                )
                self.graph.add_edge(edge)

            else:
                invals = [interpreter.read(invar) for invar in eqn.invars]
                subfuns, params = primitive.get_bind_params(eqn.params)
                primitive.bind(*subfuns, *invals, **params)


def jaxpr_to_jaxpr(
    interpreter: "PlxprInterpreter", jaxpr: "jax.extend.core.Jaxpr", consts, *args
) -> "jax.extend.core.ClosedJaxpr":
    """A convenience utility for converting jaxpr to a new jaxpr via an interpreter."""

    f = partial(interpreter.eval, jaxpr, consts)

    return jax.make_jaxpr(f)(*args)


# pylint: disable=unused-argument, too-many-arguments
@PlxprVisualizer.register_primitive(qnode_prim)
def handle_qnode(
    self, cluster, *invals, shots, qnode, device, execution_config, qfunc_jaxpr, n_consts
):
    """Handle a qnode primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]

    copied_interpreter = copy(self.interpreter)
    new_qfunc_jaxpr = jaxpr_to_jaxpr(copied_interpreter, qfunc_jaxpr, consts, *args)

    # Visualization part
    wires_uid = self.convert_name_to_uid("device_wires")
    wires_node = pydot.Node(
        wires_uid,
        label=f"{device.wires}",
        style="filled",
        fillcolor="red",
    )
    cluster.add_node(wires_node)
    for wire in device.wires:
        self.wires[str(wire)] = wires_uid

    qnode_cluster = pydot.Cluster(
        self.convert_name_to_uid(qnode.__name__),
        label=qnode.__name__,
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    cluster.add_subgraph(qnode_cluster)

    self.parse(
        new_qfunc_jaxpr.jaxpr,
        new_qfunc_jaxpr.consts,
        *args,
        cluster=qnode_cluster,
        interpreter=copied_interpreter,
    )

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


@PlxprVisualizer.register_primitive(for_loop_prim)
def handle_for_loop(
    self,
    cluster,
    start,
    stop,
    step,
    *args,
    jaxpr_body_fn,
    consts_slice,
    args_slice,
    abstract_shapes_slice,
):
    """Handle a for loop primitive."""
    consts = args[consts_slice]
    init_state = args[args_slice]
    abstract_shapes = args[abstract_shapes_slice]

    copied_interpreter = copy(self.interpreter)
    new_jaxpr_body_fn = jaxpr_to_jaxpr(
        copied_interpreter, jaxpr_body_fn, consts, *abstract_shapes, start, *init_state
    )

    # Visualization part
    uid = self.convert_name_to_uid("for_loop")
    loop_var = str(new_jaxpr_body_fn.jaxpr.invars[0])
    for_loop_cluster = pydot.Cluster(
        uid,
        label=f"for {self.convert_id_to_ascii(loop_var)} in [{start}, {stop}, {step}]",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    cluster.add_subgraph(for_loop_cluster)
    self.parse(
        new_jaxpr_body_fn.jaxpr,
        new_jaxpr_body_fn.consts,
        *args,
        cluster=for_loop_cluster,
        interpreter=copied_interpreter,
    )

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
