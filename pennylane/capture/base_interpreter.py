import copy
from functools import partial, wraps
from typing import Optional

import jax
from jax.tree_util import tree_flatten, tree_unflatten

from pennylane import cond
from pennylane.tape import QuantumScript
from pennylane.transforms.optimization.cancel_inverses import _are_inverses
from pennylane.workflow import qnode

import pennylane as qml
from .primitives import _get_abstract_measurement, _get_abstract_operator
from .capture_qnode import _get_qnode_prim
from pennylane.compiler.qjit_api import _get_for_loop_qfunc_prim, for_loop, while_loop, _get_while_loop_qfunc_prim
from pennylane.ops.op_math.condition import _get_cond_qfunc_prim

for_prim = _get_for_loop_qfunc_prim()
while_prim = _get_while_loop_qfunc_prim()
cond_prim = _get_cond_qfunc_prim()
qnode_prim = _get_qnode_prim()

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()


class PlxprInterpreter:

    _env: dict
    _op_math_cache: dict
    _primitive_registrations = {}
    
    def __init_subclass__(cls) -> None:
        cls._primitive_registrations = copy.copy(PlxprInterpreter._primitive_registrations)

    def __init__(self, state=None):
        self._env = {}
        self._op_math_cache = {}
        self.state = state

    @classmethod
    def register_primitive(cls, primitive):
        def decorator(f):
            cls._primitive_registrations[primitive] = f
            return f
        return decorator
    
    def read(self, var):
        """Extract the value corresponding to a variable."""
        if self._env is None:
            raise ValueError("_env not yet initialized.")
        return var.val if type(var) is jax.core.Literal else self._env[var]

    def setup(self):
        pass

    def cleanup(self):
        pass

    def interpret_operation(self, op: "pennylane.operation.Operator"):
        raise NotImplementedError

    def interpret_operation_eqn(self, eqn: "jax.core.JaxprEqn"):

        invals = [self.read(invar) for invar in eqn.invars]
        op = eqn.primitive.impl(*invals, **eqn.params)
        if not isinstance(eqn.outvars[0], jax.core.DropVar):
            self._op_math_cache[eqn.outvars[0]] = eqn
            self._env[eqn.outvars[0]] = op
            return
        return self.interpret_operation(op)

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        invals = [self.read(invar) for invar in eqn.invars]
        return eqn.primitive.bind(*invals, **eqn.params)

    def eval(self, jaxpr, consts, *args):
        self._env = {}
        self._op_math_cache = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars):
            self._env[constvar] = const

        measurements = []
        for eqn in jaxpr.eqns:
            custom_handler= self._primitive_registrations.get(eqn.primitive, None)
            if custom_handler:
                custom_handler(self, eqn.outvars, *eqn.invars, **eqn.params)
            elif isinstance(eqn.outvars[0].aval, AbstractOperator):
                self.interpret_operation_eqn(eqn)

            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                measurement = self.interpret_measurement_eqn(eqn)
                self._env[eqn.outvars[0]] = measurement
                measurements.append(measurement)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = eqn.primitive.bind(*invals, **eqn.params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals):
                    self._env[outvar] = outval

        self.cleanup()
        # Read the final result of the Jaxpr from the environment
        return [self._env[outvar] for outvar in jaxpr.outvars]

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            jaxpr = jax.make_jaxpr(partial(f, **kwargs))(*args)
            return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        return wrapper

@PlxprInterpreter.register_primitive(for_prim)
def handle_for_loop(self, outvars, *invars, jaxpr_body_fn, n_consts):
    invals = [self.read(invar) for invar in invars]
    start, stop, step = invals[0], invals[1], invals[2]
    consts = invals[3:3+n_consts]

    @for_loop(start, stop, step)
    def g(i, *init_state):
        return type(self)().eval(jaxpr_body_fn.jaxpr, consts, i, *init_state)

    res = g(*invals[3+n_consts:])

    for outvar, outval in zip(outvars, res):
        self._env[outvar] = outval

@PlxprInterpreter.register_primitive(cond_prim)
def handle_cond(self, outvars, *invars, jaxpr_branches, n_consts_per_branch, n_args):
    n_branches = len(jaxpr_branches)
    invals = [self.read(var) for var in invars]
    conditions = invals[:n_branches]
    consts_flat = invals[n_branches + n_args :]

    @cond(conditions[0])
    def true_fn(*args):
        return type(self)().eval(jaxpr_branches[0].jaxpr, consts_flat[:n_consts_per_branch[0]], *args)

    @true_fn.otherwise
    def _(*args):
        return type(self)().eval(jaxpr_branches[-1].jaxpr, consts_flat[n_consts_per_branch[0]:], *args)

    res = true_fn(*invals[n_branches : n_branches + n_args])

    for outvar, outval in zip(outvars, res):
        self._env[outvar] = outval


@PlxprInterpreter.register_primitive(while_prim)
def handle_while_loop(self, outvars, *invars, jaxpr_body_fn, jaxpr_cond_fn, n_consts_body, n_consts_cond):
    invals = [self.read(invar) for invar in invars]
    consts_body = invals[:n_consts_body]
    consts_cond = invals[n_consts_body: n_consts_body+n_consts_cond]
    init_state = invals[n_consts_body+n_consts_cond:]

    def cond_fn(*args):
        return jax.core.eval_jaxpr(jaxpr_cond_fn.jaxpr, consts_cond, *args)

    @while_loop(cond_fn)
    def loop(*args):
        return type(self)().eval(jaxpr_body_fn.jaxpr, consts_body, *args)

    res = loop(*init_state)

    for outvar, outval in zip(outvars, res):
        self._env[outvar] = outval


@PlxprInterpreter.register_primitive(qnode_prim)
def handle_qnode(self, outvars, *invars, shots, qnode, device, qnode_kwargs, qfunc_jaxpr, n_consts):
    invals = [self.read(invar) for invar in invars]
    consts = invals[:n_consts]

    @qml.qnode(device, **qnode_kwargs)
    def new_qnode(*args):
        return type(self)().eval(qfunc_jaxpr, consts, *args)

    res = new_qnode(invals[n_consts:])

    for outvar, outval in zip(outvars, res):
        self._env[outvar] = outval
