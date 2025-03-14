# Introduction to dynamic shapes in jax


```python
import jax
```

Dynamic shapes are an experimental feature of jax with limited support and feature coverage.
Without the `"jax_dynamic_shapes"` feature, we can't create arrays whose  size depends on an abstract value.

Note that "dynamic shapes" reference an array with a dynamic size for one or more dimensions.  
The number of dimensions must still remain fixed. We cannot do `jax.numpy.ones([3] * n)` for
a tracer `n`.

```python
jax.config.update("jax_dynamic_shapes", False)
```

```python
%xmode Minimal
def f(n):
    return jax.numpy.ones((n,))

jax.make_jaxpr(f)(3)
```

    Exception reporting mode: Minimal



    TypeError: Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>,).
    If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
    The error occurred while tracing the function f at /var/folders/k1/0v_kvphn55lgf_45kntf1hqm0000gq/T/ipykernel_27275/1980236754.py:2 for make_jaxpr. This concrete value was not available in Python because it depends on the value of the argument n.




```python
jax.make_jaxpr(f, static_argnums=0)(3)
```




        { lambda ; . let
            a:f32[3] = broadcast_in_dim[broadcast_dimensions=() shape=(3,)] 1.0
        in (a,) }



But if we make `3` as a static argnum with a fixed value, we can now produce jaxpr.  But now we can see that it has `3` hardcoded into the jaxpr and the jaxpr could not be reused with a different input.

Once we enable the experimental `"jax_dynamic_shapes"` mode we can capture such a function into jaxpr.

Now the shapes of an array can themselves contain dynamic tracers.


```python
jax.config.update("jax_dynamic_shapes", True)
```


```python
jax.make_jaxpr(f)(3)
```

        { lambda ; a:i32[]. let
            b:f32[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
        in (b,) }



With the use of the `abstracted_axes` keyword argument, we can also produce jaxpr for an input with a dynamic shape.

By using the `abstracted_axes` to make the first dimension of our input as dynamic, we can reuse the same jaxpr for different sizes of arrays.


```python
def g(x):
    return jax.numpy.sum(x)

jax.make_jaxpr(g, abstracted_axes=("x",))(jax.numpy.array([1,2,3]))
```




        { lambda ; a:i32[] b:i32[a]. let c:i32[] = reduce_sum[axes=(0,)] b in (c,) }

### Understanding `abstracted_axes`

Suppose we want to have two arrays with dynamic array dimensions `a` and `b`.
`x` has two dynamic axes, with a shape `(a, b)`. This corresponds to an abstracted axes specification of `{0:"a", 1:"b"}`.
`y` has one dynamic axis and one static axis, with a shape `(4, b)`.  This corresponds to an abstracted axes specification of
`{1:"b"}`.  As the `0` dimension is static, it is not included in the dictionary.

The abstracted axes for both `x` and `y` include the string `"b"`. This is because the second dimension of `x` and the second dimension
of `y` should always match and should be represented by a single tracer variable.

```
a = 3
b = 4
x = jnp.zeros((a, b))
y = jnp.zeros((4, b))
x_axes = {0: "a", 1: "b"}
y_axes = {1: "b"}
args = (x, y)
abstracted_axes = (x_axes, y_axes)
jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(*args)
```
```
{ lambda ; a:i32[] b:i32[] c:f32[a,b] d:f32[4,b]. let  in (0,) }
```

The abstracted axes should have the same pytree structure as `args`, but with each tensor replaced by a dictionary indicating which axes
are abstract. Suppose our first argument is instead a dictionary with tensorlike leaves.  Then we should provide an `abstracted_axes` with
the same tree structure.

```
args = ({"x": x, "y": y},)
abstracted_axes = ({"x": x_axes, "y": y_axes},)
jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(*args)
```
```
{ lambda ; a:i32[] b:i32[] c:f32[a,b] d:f32[4,b]. let  in (0,) }
```


## Limitations of dynamic shapes and numerical manipulations

1. Slicing into a dynamically sized array.

Erick has an open PR to fix this issue on the jax github.  Catalyst currently patches this bug on their side by patching the jax source code.


```python
def h(x):
    return x[0]

jax.make_jaxpr(h, abstracted_axes=("x", ) )(jax.numpy.array([0, 1, 2]))
```


    TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[]..
    The error occurred while tracing the function h at /var/folders/k1/0v_kvphn55lgf_45kntf1hqm0000gq/T/ipykernel_27275/2165410745.py:1 for make_jaxpr. This concrete value was not available in Python because it depends on the value of the argument x.
    See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError



2. Executing with `eval_jaxpr`:

No idea how to fix this right now.


```python
def k(n):
    return jax.numpy.ones((n,))

jaxpr = jax.make_jaxpr(k)(3)
jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)
```


    XlaRuntimeError: UNKNOWN: /var/folders/k1/0v_kvphn55lgf_45kntf1hqm0000gq/T/ipykernel_27275/1615670206.py:2:11: error: 'mhlo.dynamic_broadcast_in_dim' op can't be translated to XLA HLO
    /var/folders/k1/0v_kvphn55lgf_45kntf1hqm0000gq/T/ipykernel_27275/1615670206.py:4:8: note: called from
    IPython/core/interactiveshell.py:3577:20: note: called from
                        exec(code_obj, self.user_global_ns, self.user_ns)
                       ^
    IPython/core/interactiveshell.py:3517:19: note: called from
                    if await self.run_code(code, result, async_=asy):
                      ^
    IPython/core/interactiveshell.py:3334:29: note: called from
                    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
                                ^
    IPython/core/async_helpers.py:128:8: note: called from
            coro.send(None)
           ^
    IPython/core/interactiveshell.py:3130:21: note: called from
                result = runner(coro)
                        ^
    IPython/core/interactiveshell.py:3075:21: note: called from
                result = self._run_cell(
                        ^
    ipykernel/zmqshell.py:549:15: note: called from
            return super().run_cell(*args, **kwargs)
                  ^
    ipykernel/ipkernel.py:449:26: note: called from
                        res = shell.run_cell(
                             ^
    /var/folders/k1/0v_kvphn55lgf_45kntf1hqm0000gq/T/ipykernel_27275/1615670206.py:2:11: note: see current operation: %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    



## Extending support to PLXPR Higher Order Primitives (HOP's)

When capturing higher order primitives, we call `jax.make_jaxpr(f)` with arguments whose shapes are tracers.  

When calling `jax.make_jaxpr` inside a traced function, such as we do when using HOP's, we still need to specify the `abstracted_axes`.  Failing to do so leads in an error:


```python
def f(n):
    x = jax.numpy.ones((n,))
    jaxpr = jax.make_jaxpr(jax.numpy.sum)(x)
    return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, n, x)
    
jax.make_jaxpr(f)(3)
```


    AssertionError




```python
def f(n):
    x = jax.numpy.ones((n,))
    jaxpr = jax.make_jaxpr(jax.numpy.sum, abstracted_axes=("n",))(x)
    print("inner jaxpr: ", jaxpr)
    return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, n, x)
    
jax.make_jaxpr(f)(3)
```
```
inner jaxpr:  { lambda ; a:i32[] b:f32[a]. let c:f32[] = reduce_sum[axes=(0,)] b in (c,) }
{ lambda ; a:i32[]. let
    b:f32[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
    c:f32[] = reduce_sum[axes=(0,)] b
  in (c,) }
```



Note in this case that I am passing `n` when evaluating the jaxpr, even though `n` wasn't an argument that produced the jaxpr.

`n` was an implicit argument contained inside of `x`, so `make_jaxpr` promotes it to an explicit input. We can see this in the "inner jaxpr" printed out inside the function. Even though the function that produced it only had `x` as an input, the jaxpr has `a:i32[], b:f32[a]` as two arguments. When re-evaluating the jaxpr later, we need to make sure to pass the value for `n` as well.

To handle generic functions, we must then be able to determine which axes are dynamic from the arguments, and extract the tracer values for all the abstract dimensions.


```python
alphabet = "abcdefghijklmnop"
def determine_abstracted_axes(args):
    
    leaves, structure = jax.tree_util.tree_flatten(args)
    abstracted_axes = []
    abstract_shapes = []
    
    for l in leaves:
        l_shape = []
        for s in l.shape:
            if isinstance(s, int): # not abstract
                l_shape.append(())
            else:
                l_shape.append(alphabet[len(abstract_shapes)])
                abstract_shapes.append(s)
        abstracted_axes.append(tuple(l_shape) if len(l_shape) != 1 else l_shape[0]) # maybe ?
    abstracted_axes = jax.tree_util.tree_unflatten(structure, abstracted_axes)
    return abstracted_axes, abstract_shapes
```


```python
def f(n):
    x = jax.numpy.ones((n,))
    abstracted_axes, abstract_shapes = determine_abstracted_axes((x,))
    jaxpr = jax.make_jaxpr(jax.numpy.sum, abstracted_axes=abstracted_axes)(x)
    return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, x)
    
jax.make_jaxpr(f)(3)
```
```
{ lambda ; a:i32[]. let
    b:f32[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 a
    c:f32[] = reduce_sum[axes=(0,)] b
  in (c,) }
```



We can now take these learnings to make a custom higher order primitive that supports dynamically shaped inputs:


```python
prim = jax.core.Primitive("hop")
prim.multiple_results = True

@prim.def_impl
def _(*args, jaxpr, n_consts):
    return jax.core.eval_jaxpr(jaxpr, args[:n_consts], *args[n_consts:])

@prim.def_abstract_eval
def _(*args, jaxpr, n_consts):
    return [v.aval for v in jaxpr.outvars]

def bind_prim(f, *args):
    abstracted_axes, abstract_shapes  = determine_abstracted_axes(args)
    jaxpr = jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(*args)
    return prim.bind(*jaxpr.consts, *abstract_shapes, *args, jaxpr=jaxpr.jaxpr, n_consts=len(jaxpr.consts))
```


```python
def workflow(x):
    return bind_prim(jax.numpy.sum, x)

jaxpr = jax.make_jaxpr(workflow, abstracted_axes=("a", ))(jax.numpy.array([1,2,3]))
jaxpr
```


```
{ lambda ; a:i32[] b:i32[a]. let
    c:i32[] = hop[
      jaxpr={ lambda ; d:i32[] e:i32[d]. let
          f:i32[] = reduce_sum[axes=(0,)] e
        in (f,) }
      n_consts=0
    ] a b
  in (c,) }
```




```python
jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2, jax.numpy.array([1,1]))
```




    [Array(2, dtype=int32)]



Great! It's working!

At least for that example with `jax.numpy.sum`.

What happens when the higher order primitive returns a dynamic shaped array too?


```python
def workflow2(x):
    return bind_prim(lambda x: 2*x, x)

jaxpr = jax.make_jaxpr(workflow2, abstracted_axes=("a", ))(jax.numpy.array([1,2,3]))
jaxpr
```


    KeyError: Var(id=4694049536):int32[]



It no longer works ;( 

The output shape for the primitive contains a variable that is not in the local environment.  It lived in the environment of the inner jaxpr, and is not present in the outer jaxpr.

Do we have any workarounds?

If we enforce that the HOP has a return shape that *matches* one of the inputs, we are home free.

For example, with for loops and while loops, we can insist that the output shapes are the same as the input shapes:


```python
prim2 = jax.core.Primitive("hop")
prim2.multiple_results = True

@prim2.def_impl
def _(*args, jaxpr, n_consts, in_abstract_inds):
    return jax.core.eval_jaxpr(jaxpr, args[:n_consts], *args[n_consts:])

@prim2.def_abstract_eval
def _(*args, jaxpr, n_consts, n_abstract_inds):
    return args[n_consts+n_abstract_inds:]

def bind_prim2(f, *args):
    abstracted_axes, abstract_shapes  = determine_abstracted_axes(args)
    jaxpr = jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(*args)
    return prim2.bind(*jaxpr.consts, *abstract_shapes, *args,
                      jaxpr=jaxpr.jaxpr,
                      n_consts=len(jaxpr.consts),
                      n_abstract_inds=len(abstract_shapes)
                     )
```


```python
def workflow3(x):
    return bind_prim2(lambda x: 2*x, x)

jaxpr = jax.make_jaxpr(workflow3, abstracted_axes=("a", ))(jax.numpy.array([1,2,3]))
jaxpr
```

```
{ lambda ; a:i32[] b:i32[a]. let
    c:i32[a] = hop[
      jaxpr={ lambda ; d:i32[] e:i32[d]. let
          f:i32[d] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 2 d
          g:i32[d] = mul f e
        in (g,) }
      n_abstract_inds=1
      n_consts=0
    ] a b
  in (c,) }
```



So once again we are good! Our primitive accepted something of shape `i32[a]` and returned something of shape `i32[a]`. The value of `a` was already present in the local environment, so it could continue to exist in the jaxpr.

What if the shape isn't accessible? What if we wanted to resize one of the inputs, or create a fully new dimension like is done with `jax.numpy.ones`?

That now gets a bit trickier.  The solution has several issues:

1) A bit more difficult to read and follow
2) Relies on unstable componets of jax internals

But why let those concerns stop us now! Let's do it.

What we need to do in this case in hi-jack how `DynamicJaxTracer` creates an equation for the relevant primitive. It will no longer use the default logic relying on the `abstract_eval`, but our own pipeline.

Here we are going to create a primitive that accepts an argument `n`, and returns an array of shape `f32[n,2]`.


```python
prim3 = jax.core.Primitive("dynamic_output")
prim3.multiple_results = True
```




```python
from jax._src.interpreters import partial_eval as pe

def custom_staging_rule(jaxpr_trace, *invars, **params):
    new_shapes = [jax.core.DShapedArray((invars[0],2), jax.numpy.float32.dtype)]
    out_tracers = [pe.DynamicJaxprTracer(jaxpr_trace, o) for o in new_shapes]
    eqn = pe.new_jaxpr_eqn(
        [jaxpr_trace.getvar(x) for x in invars],
        [jaxpr_trace.makevar(o) for o in out_tracers],
        prim3,
        params,
        jax.core.no_effects,
    )
    jaxpr_trace.frame.add_eqn(eqn)
    return out_tracers

pe.custom_staging_rules[prim3] = custom_staging_rule
```




```python
def workflow4(n):
    return prim3.bind(n)

jax.make_jaxpr(workflow4)(2)
```

```
{ lambda ; a:i32[]. let b:f32[a,2] = dynamic_output a in (b,) }
```


This custom staging rule route will be most useful for allowing the shape of `sample` to depend on a dynamic number of shots.
