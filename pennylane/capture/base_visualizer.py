from .base_interpreter import PlxprInterpreter
from .primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)

from copy import copy
from typing import Callable


import pennylane as qml
import pydot, jax
from functools import partial


class PlxprVisualizer:
    wires: dict
    _env: dict
    _primitive_registrations: dict["jax.extend.core.Primitive", Callable] = {}

    def __init_subclass__(cls) -> None:
        cls._primitive_registrations = copy(cls._primitive_registrations)

    @classmethod
    def register_primitive(cls, primitive) -> Callable[[Callable], Callable]:
        def decorator(f: Callable) -> Callable:
            cls._primitive_registrations[primitive] = f
            return f

        return decorator

    def make_id_into_ascii(self, id):
        if "Var" not in id:
            return id
        import re

        match = re.search(r"id=(\d+)", id)

        id_str = match.group(1)
        extracted_id = int(id_str)

        from string import ascii_lowercase as letters

        if extracted_id == 0:
            return "a"

        val = []
        while extracted_id > 0:
            extracted_id, r = divmod(extracted_id, 26)
            val.append(r)

        return "".join([letters[v] for v in val[::-1]])

    def __init__(self):
        self.interpreter = qml.capture.PlxprInterpreter()

        self.unique_id = 0
        self.wires = {}
        self._env = {}
        self.wires_source = {}

        graph_attr = {
            "rankdir": "TB",
            "overlap": "false",
        }
        self.graph = pydot.Dot(graph_type="digraph", **graph_attr)
        self.graph.set_node_defaults(shape="box", style="filled", fillcolor="white")
        self.graph.set_edge_defaults(color="black", arrowhead="normal", arrowsize=0.5)
        self.last_seen_cluster = self.graph
        super().__init__()

    def read(self, var):
        return var.val if isinstance(var, jax.extend.core.Literal) else self._env[var]

    def get_unique_id(self, name):
        self.unique_id += 1
        return f"{name}_{self.unique_id}"

    def parse(self, jaxpr, consts, *args, cluster=None) -> list:
        if not cluster:
            cluster = self.graph
            self.last_seen_cluster = cluster

        in_new_cluster = False
        if cluster != self.last_seen_cluster:
            print("not the same!!")
            in_new_cluster = True

        for eqn in jaxpr.eqns:
            primitive = eqn.primitive
            custom_handler = self._primitive_registrations.get(primitive, None)

            if custom_handler:
                invals = [self.read(invar) for invar in eqn.invars]
                custom_handler(self, cluster, *invals, **eqn.params)
            elif getattr(primitive, "prim_type", "") == "operator":
                operator_uid = self.get_unique_id(primitive.name)
                operator_wires = eqn.invars[-eqn.params["n_wires"] :]
                str_operator_wires = map(str, operator_wires)
                operator_wires = list(map(self.make_id_into_ascii, str_operator_wires))

                if str(eqn.outvars) != "[_]":
                    # output wires most likely participates in measurement?
                    for outvar in eqn.outvars:
                        self.wires_source[str(outvar)] = {
                            "name": primitive.name,
                            "eqn": eqn,
                        }
                    continue

                cluster.add_node(
                    pydot.Node(
                        operator_uid,
                        label=f"{primitive.name} : {operator_wires}",
                        shape="ellipse",
                        style="filled",
                        fillcolor="lightgreen",
                    )
                )

                # Figure out how to connect previous nodes to this node
                for op_wire in operator_wires:
                    if str(op_wire) not in self.wires:
                        print("wire not in env", op_wire)
                        if in_new_cluster:
                            print("in a new cluster")
                            # connect all wires to this operator
                            for seen_wire, _ in self.wires.items():
                                print("connecting", self.wires[str(seen_wire)], operator_uid)
                                edge = pydot.Edge(
                                    self.wires[str(seen_wire)], operator_uid, style="dotted"
                                )
                                self.graph.add_edge(edge)
                                # only need to draw one (bus of wires)
                            self.wires.clear()
                            self.wires[str(op_wire)] = operator_uid
                            print("cleared and new env is ", self.wires)
                            continue
                        else:
                            self.wires[str(op_wire)] = operator_uid

                    print(
                        "connecting",
                        self.wires[str(op_wire)],
                        operator_uid,
                        f"inside cluster: {cluster.get_name()}",
                    )
                    edge = pydot.Edge(self.wires[str(op_wire)], operator_uid)
                    self.graph.add_edge(edge)
                    self.wires[str(op_wire)] = operator_uid

            elif getattr(primitive, "prim_type", "") == "measurement":
                measurement_uid = self.get_unique_id(primitive.name)
                node_involved = self.wires_source[str(eqn.invars[0])]
                name_op = node_involved["name"]
                eqn_op = node_involved["eqn"]
                cluster.add_node(
                    pydot.Node(
                        measurement_uid,
                        label=f"{primitive.name} : {name_op} : {eqn_op.invars}",
                        shape="ellipse",
                        style="filled",
                        fillcolor="lightblue",
                    )
                )

                edge = pydot.Edge(
                    self.wires.get(str(eqn_op.invars[0]), list(self.wires.values())[-1]),
                    measurement_uid,
                )
                self.graph.add_edge(edge)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
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

    # device_cluster = pydot.Cluster(
    #     name="Device" + str(len(cluster.get_nodes())),
    #     label=f"{device.name}",
    #     style="filled",
    #     fillcolor="lightgrey",
    #     color="black",
    # )
    wires_uid = self.get_unique_id("wires")
    wires_node = pydot.Node(
        wires_uid,
        label=f"{device.wires}",
        shape="diamond",
        style="filled",
        fillcolor="red",
    )
    cluster.add_node(wires_node)
    # cluster.add_subgraph(device_cluster)

    for wire in device.wires:
        self.wires[str(wire)] = wires_uid

    qnode_cluster = pydot.Cluster(
        self.get_unique_id(qnode.__name__),
        label=qnode.__name__,
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )
    cluster.add_subgraph(qnode_cluster)

    new_qfunc_jaxpr = jaxpr_to_jaxpr(self.interpreter, qfunc_jaxpr, consts, *args)

    self.parse(new_qfunc_jaxpr, new_qfunc_jaxpr.consts, *args, cluster=qnode_cluster)

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

    new_jaxpr_body_fn = jaxpr_to_jaxpr(
        self.interpreter, jaxpr_body_fn, consts, *abstract_shapes, start, *init_state
    )
    uid = self.get_unique_id("for_loop")
    loop_var = str(new_jaxpr_body_fn.jaxpr.invars[0])
    loop_var_ascii = self.make_id_into_ascii(loop_var)
    for_loop_cluster = pydot.Cluster(
        uid,
        label=f"for {loop_var_ascii} in [{start}, {stop}, {step}]",
        style="filled",
        fillcolor="lightgrey",
        color="black",
    )

    cluster.add_subgraph(for_loop_cluster)

    self.parse(new_jaxpr_body_fn, new_jaxpr_body_fn.consts, *args, cluster=for_loop_cluster)

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
