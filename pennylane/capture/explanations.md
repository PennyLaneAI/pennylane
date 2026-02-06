This documentation explains the principles behind `qp.capture.CaptureMeta` and higher order primitives.


```python
import jax
```

# Primitive basics


```python
my_func_prim = jax.extend.core.Primitive("my_func")

@my_func_prim.def_impl
def _(x):
    return x**2

@my_func_prim.def_abstract_eval
def _(x):
    return jax.core.ShapedArray((1,), x.dtype)

def my_func(x):
    return my_func_prim.bind(x)
```


```python
>>> jaxpr = jax.make_jaxpr(my_func)(0.1)
>>> jaxpr
{ lambda ; a:f32[]. let b:f32[1] = my_func a in (b,) }
>>> jaxpr.jaxpr.eqns
[a:f32[1] = my_func b]
```

# Higher Order Primitives and nested jaxpr

Higher order primitives are essentially function transforms. They are functions that accept other
functions. Our higher order primitives will include `adjoint`, `ctrl`, `for_loop`, `while_loop`, `cond`, `grad`,
and `jacobian`.

Jax describes two separate ways of defining higher order derivatives:

1) *On the fly processing*: the primitive binds the function itself as metadata

2) *Staged processing*: the primitive binds the function's jaxpr as metadata.

Jax also has a [`CallPrimitive`](https://github.com/google/jax/blob/23ad313817f20345c60281fbf727cf4f8dc83181/jax/_src/core.py#L2366)
but using this seems to be more trouble than its worth so far. Notably, this class is rather private and undocumented.

We will proceed with using *staged processing* for now. This choice is more straightforward to implement, follows Catalyst's choice of representation, and is more
explicit in the contents. On the fly isn't as much "program capture" as deferring capture till later. We want to immediately capture all aspects of the jaxpr.


Suppose we have a transform that repeats a function n times

```python
def repeat(func: Callable, n: int) -> Callable:
    def new_func(*args, **kwargs):
        for _ in range(n):
            args = func(*args, **kwargs)
        return args
    return new_func
```

We can start by creating the primitive itself:

```python
repeat_prim = jax.extend.core.Primitive("repeat")
repeat_prim.multiple_results = True
```

Instead of starting with the implementation and abstract evaluation, let's write out the function that will
bind the primitive first.  This will showcase what the args and keyword args for our bind call will look like:

```python
from functools import partial
from typing import Callable

def repeat(func: Callable, n: int) -> Callable:
    def new_func(*args, **kwargs):
        func_bound_kwargs = partial(func, **kwargs)
        jaxpr = jax.make_jaxpr(func_bound_kwargs)(*args)
        n_consts = len(jaxpr.consts)
        return repeat_prim.bind(n, *jaxpr.consts, *args, jaxpr=jaxpr.jaxpr, n_consts=n_consts)
    return new_func
```

Several things to notice about this code.

First, we have to make the jaxpr from a function with any keyword arguments
already bound.  `jax.make_jaxpr` does not currently accept keyword arguments for the function, so we need to pre-bind them.

Next, we decided to make the integer `n` a traceable parameter instead of metadata. We could have chosen to make
`n` metadata instead.  This way, we can compile our function once for different integers `n`, and it is in line with how
catalyst treats `for_loop` and `while_loop`.  If the function produced outputs of different types and shapes for different `n`,
we would have to treat `n` like metadata and re-compile for different integers `n`.

Finally, we promote the `jaxpr.consts` to being actual positional arguments. The consts
contain any closure variables that the function implicitly depends on that are not present
in the actual call signature. For example: `def f(x): return x+y`. `y` here would be a
`const` pulled from the global environment. `f` implicitly depends on it, and it is
required to reproduce the full behavior of `f`. To separate the normal positional
arguments from the consts, we then also need a `n_consts` keyword argument.

Now we can define the implementation for our primitive.

```python
@repeat_prim.def_impl
def _(n, *args, jaxpr, n_consts):
    consts = args[:n_consts]
    args = args[n_consts:]
    for _ in range(n):
        args = jax.core.eval_jaxpr(jaxpr, consts, *args)
    return args
```

Here we use `jax.core.eval_jaxpr` to execute the jaxpr with concrete arguments. If we had instead used
*on the fly processing*, we could have simply executed the stored function, but when using *staged processing*, we need
to directly evaluate the jaxpr instead.

In addition, we need to define the abstract evaluation. As the function in our case returns outputs that match the inputs in number, shape and type, we can simply extract the abstract values of the `args`.

```python
@repeat_prim.def_abstract_eval
def _(n, *args, jaxpr, n_consts):
    return args[n_consts:]
```

Now that we have all the parts, we can see it in action:

```pycon
>>> a = jax.numpy.array(1)
>>> def func(x, y, y_coeff=1):
...     return (x + a, y_coeff * y)
>>> def workflow(x):
...     return repeat(func, 2)(x, 2.0, y_coeff=2.0)
>>> workflow(0.5)
[Array(2.5, dtype=float32, weak_type=True),
 Array(8., dtype=float32, weak_type=True)]
>>> jax.make_jaxpr(workflow)(0.5)
{ lambda a:i32[]; b:f32[]. let
    c:f32[] d:f32[] = repeat[
      jaxpr={ lambda e:i32[]; f:f32[] g:f32[]. let
          h:f32[] = convert_element_type[new_dtype=float32 weak_type=True] e
          i:f32[] = add f h
          j:f32[] = mul 2.0 g
        in (i, j) }
      n_consts=1
    ] 2 a b 2.0
  in (c, d) }
>>> jax.make_jaxpr(workflow)(0.5).consts
[Array(1, dtype=int32, weak_type=True)]
```

Some notes here about how read this. `a:i32[]` is the global integer variable `a` that is
a constant.  The arguments to the `repeat` primitive are `n (const a) x (hardcoded 2.0=y)`.
You can also see the const variable `a` as argument `e:i32[]` to the inner nested jaxpr as well.

### Pytree handling

Evaluating a jaxpr requires accepting and returning a flat list of tensor-like inputs and outputs.
These long lists can be hard to manage and are very restrictive on the allowed functions, but we
can take advantage of pytrees to allow handling arbitrary functions.

To start, we import the `FlatFn` helper. This class converts a function to one that caches
the result pytree into `flat_fn.out_tree` when executed. This can be used to repack the
results into the correct shape. It also returns flattened results. This does not particularly
matter for program capture, as we will only be producing jaxpr from the function, not calling
it directly.

The following demonstrates how the tree utilities and the `FlatFn` class
can be used together to unpack and repack the variables:

```pycon
>>> from pennylane.capture.flatfn import FlatFn
>>> def f(x): # define a function with pytree inputs and outputs
...     return {"a": x[0], "b": x[1]+1}
>>> args = ([0.1, 0.2],) # the arguments to the function
>>> flatfn = FlatFn(f)
>>> flatfn.out_tree is None # initialized to None
True
>>> results = flatfn(*args)
>>> results
[0.1, 1.2]
>>> flatfn.out_tree # set once function is called
PyTreeDef({'a': *, 'b': *})
>>> jax.tree_util.tree_unflatten(flatfn.out_tree, results)
{'a': 0.1, 'b': 1.2}
```

Using these tools, we can now redefine our wrapper around the `repeat` primitive. This
code just extends what we did with the `repeat` primitive above.

```python
def repeat(func, n: int):
    def new_func(*args, **kwargs):

        func_bound_kwargs = partial(func, **kwargs)
        flat_fn = FlatFn(func_bound_kwargs)

        jaxpr = jax.make_jaxpr(flat_fn)(*args)
        flat_args, _ = jax.tree_util.tree_flatten(args)
        n_consts = len(jaxpr.consts)
        results = repeat_prim.bind(n, *jaxpr.consts, *flat_args, jaxpr=jaxpr.jaxpr, n_consts=n_consts)

        # repack the results back into the cached pytree
        assert flat_fn.out_tree is not None
        return jax.tree_util.tree_unflatten(flat_fn.out_tree, results)
    return new_func
```

And now the workflow supports arbitrary inputs and outputs.

```pycon
>>> a = {"x": jax.numpy.array(1), "y": 2.0}
>>> def func(arg, y_coeff=1):
...     return {"x": arg['x'] + 2.0, "y": y_coeff * arg['y']}
>>> def workflow(arg):
...     return repeat(func, 2)(arg, y_coeff=2.0)
>>> workflow(a)
{'x': Array(5., dtype=float32, weak_type=True),
 'y': Array(8., dtype=float32, weak_type=True)}
```

## Metaprogramming


```python
class MyMetaClass(type):

    def __init__(cls, *args, **kwargs):
        print(f"Creating a new type {cls} with {args}, {kwargs}. ")

        # giving every class a property
        cls.a = "a"

    def __call__(cls, *args, **kwargs):
        print(f"creating an instance of type {cls} with {args}, {kwargs}. ")
        inst = cls.__new__(cls, *args, **kwargs)
        inst.__init__(*args, **kwargs)
        return inst
```

Now let's define a class with this meta class.

You can see that when we *define* the class, we have called `MyMetaClass.__init__` to create the new type


```python
class MyClass(metaclass=MyMetaClass):

    def __init__(self, *args, **kwargs):
        print("now creating an instance in __init__")
        self.args = args
        self.kwargs = kwargs
```

    Creating a new type <class '__main__.MyClass'> with ('MyClass', (), {'__module__': '__main__', '__qualname__': 'MyClass', '__init__': <function MyClass.__init__ at 0x11c59cae0>}), {}.


And that we have set a class property `a`


```python
>>> MyClass.a
'a'
```

But can we actually create instances of these classes?


```python
>> obj = MyClass(0.1, a=2)
>>> obj
creating an instance of type <class '__main__.MyClass'> with (0.1,), {'a': 2}.
now creating an instance in __init__
<__main__.MyClass at 0x11c5a2810>
```


So far, we've just added print statements around default behavior.  Let's try something more radical


```python
class MetaClass2(type):

    def __call__(cls, *args, **kwargs):
        return 2.0

class MyClass2(metaclass=MetaClass2):

    def __init__(self, *args, **kwargs):
        print("Am I here?")
        self.args = args
```

You can see now that instead of actually getting an instance of `MyClass2`, we just get `2.0`.

Using a metaclass, we can hijack what happens when a type is called.


```python
>>> out = MyClass2(1.0)
>>> out, out == 2.0
(2.0, True)
```

## Putting Primitives and Metaprogramming together

We have two goals that we need to accomplish with our meta class.

1. Create an associated primitive every time we define a new class type
2. Hijack creating a new instance to use `primitive.bind` instead


```python
class PrimitiveMeta(type):

    def __init__(cls, *args, **kwargs):
        # here we set up the primitive
        primitive = jax.extend.core.Primitive(cls.__name__)

        @primitive.def_impl
        def _(*inner_args, **inner_kwargs):
            # just normal class creation if not tracing
            return type.__call__(cls, *inner_args, **inner_kwargs)

        @primitive.def_abstract_eval
        def _(*inner_args, **inner_kwargs):
            # here we say that we just return an array of type float32 and shape (1,)
            # other abstract types could be used instead
            return jax.core.ShapedArray((1,), jax.numpy.float32)

        cls._primitive = primitive

    def __call__(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)
```


```python
class PrimitiveClass(metaclass=PrimitiveMeta):

    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return f"PrimitiveClass({self.a})"
```

What happens if we just create a class normally as is?


```python
>>> PrimitiveClass(1.0)
PrimitiveClass(1.0)
```

But now it can also be used in tracing as well


```python
>>> jax.make_jaxpr(PrimitiveClass)(1.0)
{ lambda ; a:f32[]. let b:f32[1] = PrimitiveClass a in (b,) }
```

Great!👍

Now you can see that the problem is that we lied in our definition of abstract evaluation.  Jax thinks that `PrimitiveClass` returns something of shape `(1,)` and type `float32`.

But jax doesn't have an abstract type that really describes "PrimitiveClass".  So we need to define and register our own.


```python
class AbstractPrimitiveClass(jax.core.AbstractValue):

    def __eq__(self, other):
        return isinstance(other, AbstractPrimitiveClass)

    def __hash__(self):
        return hash("AbstractPrimitiveClass")
```

Now we can redefine our class to use this abstract class


```python
class PrimitiveMeta2(type):

    def __init__(cls, *args, **kwargs):
        # here we set up the primitive
        primitive = jax.extend.core.Primitive(cls.__name__)

        @primitive.def_impl
        def _(*inner_args, **inner_kwargs):
            # just normal class creation if not tracing
            return type.__call__(cls, *inner_args, **inner_kwargs)

        @primitive.def_abstract_eval
        def _(*inner_args, **inner_kwargs):
            # here we say that we just return an array of type float32 and shape (1,)
            # other abstract types could be used instead
            return AbstractPrimitiveClass()

        cls._primitive = primitive

    def __call__(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

class PrimitiveClass2(metaclass=PrimitiveMeta2):

    def __init__(self, a):
        self.a = a

    def __repr__(self):
        return f"PrimitiveClass({self.a})"
```

Now in our jaxpr, we can see thet `PrimitiveClass2` returns something of type `AbstractPrimitiveClass`.


```python
>>> jax.make_jaxpr(PrimitiveClass2)(0.1)
{ lambda ; a:f32[]. let b:AbstractPrimitiveClass() = PrimitiveClass2 a in (b,) }
```

# Non-interpreted primitives

**WARNING:** THIS EXPLANATION IS SPECIFIC TO JAX 0.4.28 AND NO LONGER APPLIES


Some of the primitives in the capture module have a somewhat non-standard requirement for the
behaviour under differentiation or batching: they should ignore that an input is a differentiation
or batching tracer and just execute the standard implementation on them.

We will look at an example to make the necessity for such a non-interpreted primitive clear.

Consider a finite-difference differentiation routine together with some test function `fun`.

```python
def finite_diff_impl(x, fun, delta):
    """Finite difference differentiation routine. Only supports differentiating
    a function `fun` with a single scalar argument, for simplicity."""

    out_plus = fun(x + delta)
    out_minus = fun(x - delta)
    return tuple((out_p - out_m) / (2 * delta) for out_p, out_m in zip(out_plus, out_minus))

def fun(x):
    return (x**2, 4 * x - 3, x**23)
```

Now suppose we want to turn this into a primitive. We could just promote it to a standard
`jax.extend.core.Primitive` as

```python
import jax

fd_prim = jax.extend.core.Primitive("finite_diff")
fd_prim.multiple_results = True
fd_prim.def_impl(finite_diff_impl)

def finite_diff(x, fun, delta=1e-5):
    return fd_prim.bind(x, fun, delta)
```

This allows us to use the forward pass as usual (to compute the first-order derivative):

```pycon
>>> finite_diff(1., fun, delta=1e-6)
(2.000000000002, 3.999999999892978, 23.000000001216492)
```

Now if we want to make this primitive differentiable (with automatic
differentiation/backprop, not by using a higher-order finite difference scheme),
we need to specify a JVP rule. (Note that there are multiple rather simple fixes for this example
that we could use to implement a finite difference scheme that is readily differentiable. This is
somewhat beside the point because we did not identify the possibility of using any of those
alternatives in the PennyLane code).

However, the finite difference rule is just a standard
algebraic function making use of calls to `fun` and some elementary operations, so ideally
we would like to just use the chain rule as it is known to the automatic differentiation framework. A JVP rule would
then just manually re-implement this chain rule, which we'd rather not do.

Instead, we define a non-interpreted type of primitive and create such a primitive
for our finite difference method. We also create the usual method that binds the
primitive to inputs.

```python
class NonInterpPrimitive(jax.extend.core.Primitive):
    """A subclass to JAX's Primitive that works like a Python function
    when evaluating JVPTracers."""

    def bind_with_trace(self, trace, args, params):
        """Bind the ``NonInterpPrimitive`` with a trace.
        If the trace is a ``JVPTrace``, it falls back to a standard Python function call.
        Otherwise, the bind call of JAX's standard Primitive is used."""
        if isinstance(trace, jax.interpreters.ad.JVPTrace):
            return self.impl(*args, **params)
        return super().bind_with_trace(trace, args, params)

fd_prim_2 = NonInterpPrimitive("finite_diff_2")
fd_prim_2.multiple_results = True
fd_prim_2.def_impl(finite_diff_impl) # This also defines the behaviour with a JVP tracer

def finite_diff_2(x, fun, delta=1e-5):
    return fd_prim_2.bind(x, fun, delta)
```

Now we can use the primitive in a differentiable workflow, without defining a JVP rule
that just repeats the chain rule:

```pycon
>>> # Forward execution of finite_diff_2 (-> first-order derivative)
>>> finite_diff_2(1., fun, delta=1e-6)
(2.000000000002, 3.999999999892978, 23.000000001216492)
>>> # Differentiation of finite_diff_2 (-> second-order derivative)
>>> jax.jacobian(finite_diff_2)(1., fun, delta=1e-6)
(Array(1.9375, dtype=float32, weak_type=True), Array(0., dtype=float32, weak_type=True), Array(498., dtype=float32, weak_type=True))
```

In addition to the differentiation primitives for `qp.jacobian` and `qp.grad`, quantum operators
have non-interpreted primitives as well. This is because their differentiation is performed
by the surrounding QNode primitive rather than through the standard chain rule that acts
"locally" (in the circuit). In short, we only want gates to store their tracers (which will help
determine the differentiability of gate arguments, for example), but not to do anything with them.

