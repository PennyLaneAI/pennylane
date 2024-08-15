from functools import partial, wraps

import jax
from jax.tree_util import tree_flatten, tree_unflatten

from pennylane.compiler.qjit_api import (
    _get_for_loop_qfunc_prim,
    _get_while_loop_qfunc_prim,
    for_loop,
    while_loop,
)
from pennylane.devices.qubit import apply_operation, create_initial_state, measure
from pennylane.measurements.mid_measure import MidMeasureMP, _create_mid_measure_primitive
from pennylane.ops.op_math.condition import _get_cond_qfunc_prim
from pennylane.tape import QuantumScript

from .base_interpreter import PlxprInterpreter

for_prim = _get_for_loop_qfunc_prim()
midmeasure_prim = _create_mid_measure_primitive()
cond_prim = _get_cond_qfunc_prim()


class DefaultQubitInterpreter(PlxprInterpreter):

    def __init__(self, num_wires=None, state=None):
        self.num_wires = num_wires
        self.state = state

    @property
    def statevector(self):
        return None if self.state is None else self.state["statevector"]

    @statevector.setter
    def statevector(self, val):
        if self.state is None:
            self.state = {"statevector": val}
        else:
            self.state["statevector"] = val

    def setup(self):
        if self.statevector is None:
            self.statevector = create_initial_state(range(self.num_wires))

    def cleanup(self):
        self.state = None

    def interpret_operation(self, op):
        self.statevector = apply_operation(op, self.statevector)
        return op

    def interpret_measurement_eqn(self, eqn):
        invals = [self.read(invar) for invar in eqn.invars]
        mp = eqn.primitive.impl(*invals, **eqn.params)
        return measure(mp, self.statevector)


@DefaultQubitInterpreter.register_primitive(for_prim)
def handle_for_loop(self, *invals, jaxpr_body_fn, n_consts):
    start, stop, step = invals[0], invals[1], invals[2]
    consts = invals[3 : 3 + n_consts]
    init_state = invals[3 + n_consts :]

    res = None
    for i in range(start, stop, step):
        res = type(self)(state=self.state).eval(jaxpr_body_fn.jaxpr, consts, i, *init_state)

    return res


@DefaultQubitInterpreter.register_primitive(midmeasure_prim)
def handle_mm(self, *invals, reset, postselect):
    mp = MidMeasureMP(invals, reset=reset, postselect=postselect)
    mid_measurements = {}
    self.statevector = apply_operation(mp, self.statevector, mid_measurements=mid_measurements)
    return mid_measurements[mp]


@DefaultQubitInterpreter.register_primitive(cond_prim)
def handle_cond(self, *invals, jaxpr_branches, n_consts_per_branch, n_args):
    n_branches = len(jaxpr_branches)
    conditions = invals[:n_branches]
    consts_flat = invals[n_branches + n_args :]
    args = invals[n_branches : n_branches + n_args]

    if conditions[0]:
        return type(self)(state=self.state).eval(
            jaxpr_branches[0].jaxpr, consts_flat[: n_consts_per_branch[0]], *args
        )
    return type(self)(state=self.state).eval(
        jaxpr_branches[-1].jaxpr, consts_flat[: n_consts_per_branch[0]], *args
    )


from pennylane_lightning.lightning_qubit._state_vector import (
    LightningMeasurements,
    LightningStateVector,
)


class LightningInterpreter(DefaultQubitInterpreter):

    def setup(self):
        if self.statevector is None:
            self.statevector = LightningStateVector(self.num_wires)

    def interpret_operation(self, op):
        self.statevector._apply_lightning([op])

    def interpret_measurement(self, m):
        return LightningMeasurements(self.statevector).measurement(m)


class DecompositionInterpreter(PlxprInterpreter):
    """
    >>> def f(x):
    ...     qml.IsingXX(x, wires=(0,1))
    ...     qml.Rot(0.5, x, 1.5, wires=1)
    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> DecompositionInterpreter().call_jaxpr(jaxpr.jaxpr, jaxpr.consts)(0.5)
    { lambda ; a:f32[]. let
        _:AbstractOperator() = CNOT[n_wires=2] 0 1
        _:AbstractOperator() = RX[n_wires=1] a 0
        _:AbstractOperator() = CNOT[n_wires=2] 0 1
        _:AbstractOperator() = RZ[n_wires=1] 0.5 1
        _:AbstractOperator() = RY[n_wires=1] a 1
        _:AbstractOperator() = RZ[n_wires=1] 1.5 1
    in () }

    """

    def interpret_operation(self, op):
        if op.has_decomposition:
            op.decomposition()
        else:
            vals, structure = tree_flatten(op)
            tree_unflatten(structure, vals)


class ConvertToTape(PlxprInterpreter):
    """

    >>> def f(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> ConvertToTape()(jaxpr.jaxpr, jaxpr.consts, 1.2).circuit
    [RX(1.2, wires=[0]), expval(Z(0))]

    """

    def setup(self):
        if self.state is None:
            self.state = {"ops": [], "measurements": []}

    def interpret_operation(self, op):
        self.state["ops"].append(op)

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        invals = [self.read(invar) for invar in eqn.invars]
        mp = eqn.primitive.bind(*invals, **eqn.params)
        self.state["measurements"].append(mp)
        return mp

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            jaxpr = jax.make_jaxpr(partial(f, **kwargs))(*args)
            _ = self.eval(jaxpr.jaxpr, jaxpr.consts, *args)
            return QuantumScript(self.state["ops"], self.state["measurements"])

        return wrapper


@ConvertToTape.register_primitive(for_prim)
def handle_for_loop(self, *invals, jaxpr_body_fn, n_consts):
    start, stop, step = invals[0], invals[1], invals[2]
    consts = invals[3 : 3 + n_consts]
    init_state = invals[3 + n_consts :]

    res = None
    for i in range(start, stop, step):
        res = type(self)(state=self.state).eval(jaxpr_body_fn.jaxpr, consts, i, *init_state)

    return res


class CancelInverses(PlxprInterpreter):
    """

    >>> def f(x):
    ...     qml.X(0)
    ...     qml.X(0)
    ...     qml.Hadamard(0)
    ...     qml.Y(1)
    ...     qml.RX(x, 0)
    ...     qml.adjoint(qml.RX(x, 0))
    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> CancelInverses().call_jaxpr(jaxpr.jaxpr, jaxpr.consts)(0.5)
    { lambda ; a:f64[]. let
        _:AbstractOperator() = Hadamard[n_wires=1] 0
        _:AbstractOperator() = PauliY[n_wires=1] 1
    in () }

    """

    _last_op_on_wires = None

    def setup(self):
        self._last_op_on_wires = {}

    def interpret_operation(self, op):
        if len(op.wires) != 1:
            for w in op.wires:
                self._last_op_on_wires[w] = None
            vals, structure = tree_flatten(op)
            tree_unflatten(structure, vals)
            return

        w = op.wires[0]
        if w in self._last_op_on_wires:
            if _are_inverses(self._last_op_on_wires[w], op):
                self._last_op_on_wires[w] = None
                return
            previous_op = self._last_op_on_wires[w]
            if previous_op is not None:
                vals, structure = tree_flatten(previous_op)
                tree_unflatten(structure, vals)
        self._last_op_on_wires[w] = op
        return

    def interpret_measurement(self, m):
        vals, structure = tree_flatten(m)
        return tree_unflatten(structure, vals)

    def cleanup(self):
        for _, op in self._last_op_on_wires.items():
            if op is not None:
                vals, structure = tree_flatten(op)
                tree_unflatten(structure, vals)


class MergeRotations(PlxprInterpreter):
    """

    >>> def g(x):
    ...     qml.RX(x, 0)
    ...     qml.RX(2*x, 0)
    ...     qml.RX(-4*x, 0)
    ...     qml.X(0)
    ...     qml.RX(0.5, 0)
    >>> plxpr = jax.make_jaxpr(g)(1.0)
    >>> MergeRotations().call_jaxpr(plxpr.jaxpr, plxpr.consts)(1.0)
    { lambda ; a:f64[]. let
        b:f64[] = mul 2.0 a
        c:f64[] = add b a
        d:f64[] = mul -4.0 a
        e:f64[] = add d c
        _:AbstractOperator() = RX[n_wires=1] e 0
        _:AbstractOperator() = PauliX[n_wires=1] 0
        _:AbstractOperator() = RX[n_wires=1] 0.5 0
    in () }

    """

    _last_op_on_wires = None

    def setup(self):
        self._last_op_on_wires = {}

    def interpret_operation(self, op):
        if len(op.wires) != 1:
            for w in op.wires:
                self._last_op_on_wires[w] = None
            vals, structure = tree_flatten(op)
            tree_unflatten(structure, vals)
            return

        w = op.wires[0]
        if w in self._last_op_on_wires:
            previous_op = self._last_op_on_wires[w]
            if op.name == previous_op.name and op.wires == previous_op.wires:
                new_data = [d1 + d2 for d1, d2 in zip(op.data, previous_op.data)]
                self._last_op_on_wires[w] = op._primitive.impl(*new_data, wires=op.wires)
                return
            if previous_op is not None:
                vals, structure = tree_flatten(previous_op)
                tree_unflatten(structure, vals)
        self._last_op_on_wires[w] = op
        return

    def interpret_measurement(self, m):
        vals, structure = tree_flatten(m)
        return tree_unflatten(structure, vals)

    def cleanup(self):
        for _, op in self._last_op_on_wires.items():
            if op is not None:
                vals, structure = tree_flatten(op)
                tree_unflatten(structure, vals)
