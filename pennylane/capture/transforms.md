This document provides a brief introduction to JAX `Trace`s and `Tracer`s and explains how we use
them in PennyLane to apply PennyLane transforms to PLxPR without having to create tapes. A lot of
details here have been derived and reused from the
[JAX Autodidax tutorial](https://jax.readthedocs.io/en/latest/autodidax.html), which you can refer
to for more details about JAX's core system.

A `Tracer` represents a boxed-up value, and a `Trace` handles boxing up values into `Tracer`s and
handles applying/transforming primitives.

```python
import jax
import pennylane as qml

qml.capture.enable()
```

# Trace basics

JAX represents active interpreters as a stack, stored as a list of containers, where each container
has an interpreter type (`trace_type`), an integer corresponding to its height in the stack
(`level`), and an optional field containing metadata for the interpreter. This container is called
a `MainTrace`. Below is a minimal implementation of `MainTrace` taken from the Autodidax tutorial:

```python
from contextlib import contextmanager
from typing import Any

class MainTrace:
    level: int
    trace_type: type["Trace"]
    global_data: Any | None

    def __init__(self, level, trace_type, global_data):
        self.level = level
        self.trace_type = trace_type
        self.global_data = global_data

trace_stack: list[MainTrace] = []

@contextmanager
def new_main(trace_type: type["Trace"], global_data=None):
    """Create a new MainTrace and place is at the top of the trace_stack"""
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()
```

When applying a transformation, we push another `MainTrace` into the stack using `new_main`. Then,
as we bind primitives, `bind` first gets interpreted by the trace at the top of the stack. If the
trace itself binds new primitives, these `bind` calls will be handled by the next trace in the
stack. Let's define `Trace` and `Tracer` to demonstrate:

```python
class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val):
        """Wrap a value inside a Tracer"""
        assert False  # must override

    def lift(self, val):
        """Wrap a Tracer belonging to a lower-level Trace inside a Tracer
        of the current level"""
        assert False  # must override

    def process_primitive(self, primitive, tracers, params):
        """Evaluate a primitive based on what the transformation is supposed
        to do. Here, we can bind new primitives, which might be interpreted by
        Traces lower in the stack."""
        assert False  # must override
```

`pure` and `lift` are for boxing up values in `Tracer`s, and `process_primitive` is the callback we
use to transform primitives. Multiple instances of `Trace` might be created and discarded during
application of a transformation, whereas only a single `MainTrace` instance is created per
application of a transformation.

```python
class Tracer:
    _trace: Trace

    __array_priority__ = 1000

    @property
    def aval(self):
        """Abstract value of the Tracer"""
        assert False  # must override

    def full_lower(self):
        return self  # default implementation

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
```

A `Tracer` carries an abstract value, and the rest is up to the transformation.

Now, let's set up minimal versions of other JAX helper functions that are used during
interpretation, namely `jax.core.bind_with_trace` (which is called by `Primitive.bind`),
`jax.core.find_top_trace` (to figure out which `MainTrace`, or interpreter, should handle
the current primitive), `jax.core.full_raise` (to ensure that values are boxed up in the
top `Trace`'s `Tracer`s), and `jax.core.full_lower` (an optional optimization to unbox
values out of `Tracer`s as much as possible).

```python
from typing import Sequence, Union

def find_top_trace(xs: Sequence[Union[Any, Tracer]]) -> Trace:
    """Find the MainTrace corresponding to the highest level interpreter that should be used for
    the given inputs, and return a Trace corresponding to the trace_type of that MainTrace"""
    top_main = max(
        (x._trace.main for x in xs if isinstance(x, Tracer)),
        default=trace_stack[0],
        key=lambda main: main.level,
    )

    return top_main.trace_type(top_main)
```
`find_top_trace` returns the highest-level interpreter associated with the `Tracer`s on its inputs,
and otherwise returns the interpreter at the bottom of the stack. This last interpreter does the
standard evaluation on the primitive we are binding (more on that later). Note that we are only
applying an interpreter when the input arguments to a primitive are boxed in a `Tracer`
corresponding to that interpreter. This is an optimization that JAX uses to skip usage of
unnecessary interpreters, so that transformations follow data-dependence (except for the special
level-0 interpreter, which interprets everything). This data-dependence will cause issues with
PennyLane transforms, so we will have to work around it (more on that later).

```python
def full_lower(val: Any):
    if isinstance(val, Tracer):
        return val.full_lower()
    else:
        return val


def full_raise(trace: Trace, val: Any) -> Tracer:
    if not isinstance(val, Tracer):
        return trace.pure(val)

    level = trace.main.level
    if val._trace.main is trace.main:
        return val
    elif val._trace.main.level < level:
        return trace.lift(val)
    elif val._trace.main.level > level:
        raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
    else:  # val._trace.level == level
        raise Exception(f"Different traces at same level: {val._trace}, {trace}.")
```

```python
def bind(prim, *args, **params):
    print("Call bind")
    top_trace = find_top_trace(args)
    tracers = [full_raise(top_trace, arg) for arg in args]
    outs = top_trace.process_primitive(prim, tracers, params)
    return [full_lower(out) for out in outs]
```

Finally, let's define a minimal version of the JAX `Primitive` to integrate with our framework:

```python
from typing import Callable

class Primitive:
    name: str
    impl: Callable
    abstract_eval: Callable

    def __init__(self, name: str):
        self.name = name

    def def_impl(self, impl):
        self.impl = impl
        return self.impl

    def def_abstract_eval(self, abstract_eval):
        self.abstract_eval = abstract_eval
        return self.abstract_eval

    def bind(*args, **kwargs):
        return bind(*args, **kwargs)
```

Now, we have everything we need to start implementing interpreters. Let's start with the level-0
interpreter, called the Evaluation Interpreter.

##  Evaluation Interpreter (EvalTrace)

```python
class EvalTrace(Trace):
    pure = lift = lambda self, x: x  # no boxing in Tracers needed

    def process_primitive(self, primitive, tracers, params):
        """Just do standard evaluation of the primitive being binded (bound?)"""
        print("Binding primitive with standard evaluation")
        return [primitive.impl(*tracers, **params)]
```

With this interpreter, we can evaluate functions. Let's try it out. First, we define a simple
`Primitive`:

```python
def sin(x):
    prim = Primitive("sin")

    @prim.def_impl
    def _(*args, **kwargs):
        return np.sin(args[0])

    @prim.def_abstract_eval
    def _(*args, **kwargs):
        return jax.core.ShapedArray(args[0].shape, args[0].dtype)

    return prim.bind(x)[0]
```

Now, let's create a simple user function that uses our `sin` `Primitive` and bring it all together:

```python
def f(x):
    return sin(x)
```
```pycon
>>> print(f(3.5))
Call bind
Binding primitive with standard evaluation
-0.35078322768961984
>>> print(np.sin(3.5))
-0.35078322768961984
```

Now that we have a good understanding of how JAX's core works, we can start creating `Trace`s for
our purposes.

# PennyLane Transforms

As mentioned earlier, JAX has a built in optimization to avoid processing primitives with
unnecessary interpreters by leveraging data-dependence: we only apply an interpreter when
the input arguments to a `primitive.bind` call are boxed in a `Tracer` corresponding to
the interpreter. However, this optimization is unfavourable for our purposes. We want to
interpret _all_ operator and measurement primitives using our "pennylane transform
interpreter", and we cannot enforce the boxing of inputs to these primitives inside custom
`Tracer`s.

To circumvent this, we can use a `PlxprInterpreter` to selectively box inputs of operator and
measurement primitives inside custom `Tracer`s, thus giving us the ability to transform them
as needed.

Check out `pennylane/capture/transforms.py` to see how the `TransformInterpreter`, a child class of
`PlxprInterpreter`, is implemented, and how it uses `TransformTrace` and `TransformTracer` to
transform PennyLane primitives.

## How to process primitives for custom transforms

To create new transforms for PLxPR, or to register already-existing transforms for natively
transforming PLxPR, use the `custom_plxpr_transform` decorator of `TransformDispatcher`:

```python
@qml.transforms.core.transform
def convert_rx_to_ry(tape):
    new_ops = [
        qml.RY(op.data[0], op.wires) if isinstance(op, qml.RX) else op for op in tape.operations
    ]
    new_tape = qml.tape.QuantumScript(
        new_ops, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params
    )
    return [new_tape], lambda results: results[0]


@convert_rx_to_ry.custom_plxpr_transform
def _(self, primitive, tracers, params, targs, tkwargs, state):
    from pennylane.capture import TransformTracer

    # Step 1: Transform primitive
    primitive = qml.RY._primitive if primitive.name == "RX" else primitive
    # Step 2: Update tracers
    tracers = [
        TransformTracer(t._trace, t.val, t.idx + 1) if isinstance(t, TransformTracer) else t
        for t in tracers
    ]
    # Step 3: Return the result of the transformation
    return primitive.bind(*tracers, **params)
```

Here, we transform an `RX` primitive into an `RY` primitive. Note step 2, which increments the `idx`
of all `TransformTracer`s. This is required for the `TransformTrace` to know which transform in the
`TransformProgram` we are currently applying.

We can now use this transform to transform PLxPR:

```python
def qfunc(x):
    qml.X(0)
    qml.RX(x, 0)
    qml.RZ(2.5, 1)
    qml.CNOT([0, 1])
    qml.RX(x - 1, 1)
    qml.RY(x + 1, 1)
    return qml.expval(qml.Z(1))
```
```pycon
>>> print(jax.make_jaxpr(qfunc)(6.1))
{ lambda ; a:f32[]. let
    _:AbstractOperator() = PauliX[n_wires=1] 0
    _:AbstractOperator() = RX[n_wires=1] a 0
    _:AbstractOperator() = RZ[n_wires=1] 2.5 1
    _:AbstractOperator() = CNOT[n_wires=2] 0 1
    b:f32[] = sub a 1.0
    _:AbstractOperator() = RX[n_wires=1] b 1
    c:f32[] = add a 1.0
    _:AbstractOperator() = RY[n_wires=1] c 1
    d:AbstractOperator() = PauliZ[n_wires=1] 1
    e:AbstractMeasurement(n_wires=None) = expval_obs d
  in (e,) }
```

```python
program = qml.transforms.core.TransformProgram()
program.add_transform(convert_rx_to_ry)
transformer = TransformInterpreter(program)

transformed_qfunc = transformer(qfunc)
```
```pycon
>>> print(jax.make_jaxpr(transformed_qfunc)(6.1))
{ lambda ; a:f32[]. let
    _:AbstractOperator() = PauliX[n_wires=1] 0
    _:AbstractOperator() = RY[n_wires=1] a 0
    _:AbstractOperator() = RZ[n_wires=1] 2.5 1
    _:AbstractOperator() = CNOT[n_wires=2] 0 1
    b:f32[] = sub a 1.0
    _:AbstractOperator() = RY[n_wires=1] b 1
    c:f32[] = add a 1.0
    _:AbstractOperator() = RY[n_wires=1] c 1
    d:AbstractOperator() = PauliZ[n_wires=1] 1
    e:AbstractMeasurement(n_wires=None) = expval_obs d
  in (e,) }
```

All the `RX` primitives have been transformed into `RY` primitives. Woohoo!

## Constraints

This implementation has many constraints. Hopefully some of these can be mitigated in the future.
But, for now, they're listed below:

1. Only transforms that return a single tape can be implemented to transform PLxPR natively.
2. Only transforms with trivial processing functions can be implemented to transform PLxPR natively.
3. Transforms that do multiple passes over the tape cannot be implemented using this strategy.
4. Transforms that traverse the tape in reverse cannot be implemented using this strategy.

Some of the above constraints can be mitigated. We can re-think at least some transforms to be
impementable using the above strategy using the `state` input. For example, we can store
intermediate primitives in `state` without applying them until a certain condition is met. This can
be useful for mitigating point 4.

A reminder that this is an ongoing project, and will be subject to change without a deprecation.
