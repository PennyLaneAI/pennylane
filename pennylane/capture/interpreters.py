from functools import partial
from typing import Optional

import jax
from jax.tree_util import tree_flatten, tree_unflatten

from pennylane.devices.qubit import apply_operation, create_initial_state, measure
from pennylane.tape import QuantumScript
from pennylane.transforms.optimization.cancel_inverses import _are_inverses

from .primitives import _get_abstract_measurement, _get_abstract_operator

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()


class PlxprInterpreter:

    _env: Optional[dict] = None
    _op_math_cache: Optional[dict] = None

    def _read(self, var):
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

        invals = [self._read(invar) for invar in eqn.invars]
        op = eqn.primitive.impl(*invals, **eqn.params)
        if not isinstance(eqn.outvars[0], jax.core.DropVar):
            self._op_math_cache[eqn.outvars[0]] = eqn
            self._env[eqn.outvars[0]] = op
            return
        return self.interpret_operation(op)

    def interpret_measurement(self, measurement: "pennylane.measurements.MeasurementProcess"):
        raise NotImplementedError

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        invals = [self._read(invar) for invar in eqn.invars]
        mp = eqn.primitive.impl(*invals, **eqn.params)
        return self.interpret_measurement(mp)

    def call_jaxpr(self, jaxpr, consts):
        partial_call = partial(self, jaxpr, consts)
        return jax.make_jaxpr(partial_call)

    def __call__(self, jaxpr, consts, *args):
        self._env = {}
        self._op_math_cache = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars):
            self._env[constvar] = const

        measurements = []
        for eqn in jaxpr.eqns:
            if isinstance(eqn.outvars[0].aval, AbstractOperator):
                self.interpret_operation_eqn(eqn)

            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                measurement = self.interpret_measurement_eqn(eqn)
                measurements.append(measurement)
            else:
                invals = [self._read(invar) for invar in eqn.invars]
                outvals = eqn.primitive.bind(*invals, **eqn.params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals):
                    self._env[outvar] = outval

        self.cleanup()
        # Read the final result of the Jaxpr from the environment
        return measurements


class DefaultQubitInterpreter(PlxprInterpreter):

    _state = None

    def __init__(self, num_wires):
        self.num_wires = num_wires

    def setup(self):
        self._state = create_initial_state(range(self.num_wires))

    def cleanup(self):
        self._state = None

    def interpret_operation(self, op):
        self._state = apply_operation(op, self._state)

    def interpret_measurement(self, m):
        return measure(m, self._state)


from pennylane_lightning.lightning_qubit._state_vector import (
    LightningMeasurements,
    LightningStateVector,
)


class LightningInterpreter(PlxprInterpreter):

    def __init__(self, num_wires):
        self._num_wires = num_wires

    def setup(self):
        self._state = LightningStateVector(self._num_wires)

    def interpret_operation(self, op):
        self._state._apply_lightning([op])

    def interpret_measurement(self, m):
        return LightningMeasurements(self._state).measurement(m)


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

    def interpret_measurement(self, m):
        vals, structure = tree_flatten(m)
        return tree_unflatten(structure, vals)


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
        self._ops = []
        self._measurements = []

    def interpret_operation(self, op):
        self._ops.append(op)

    def interpret_measurement(self, m):
        self._measurements.append(m)
        return m

    def __call__(self, jaxpr, consts, *args):
        out = super().__call__(jaxpr, consts, *args)
        return QuantumScript(self._ops, self._measurements)


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
