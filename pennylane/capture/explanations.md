This documentation explains the principles behind `qml.capture.CaptureMeta` and higher order primitives.


```python
import jax
```

# Primitive basics


```python
my_func_prim = jax.core.Primitive("my_func")

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
functions. Our higher order primitives will be `adjoint`, `ctrl`, `for_loop`, `while_loop`, `cond`, `grad`,
and `jacobian`.

Jax describes two separate ways of defining higher order derivatives:

1) *On the fly processing*: the primitive binds the function itself as metadata

2) *Staged processing*: the primitive binds the function's jaxpr as metadata.

Jax also has a [`CallPrimitive`](https://github.com/google/jax/blob/23ad313817f20345c60281fbf727cf4f8dc83181/jax/_src/core.py#L2366)
but using this seems to be more trouble than its worth so far. Notably, this class is rather private and undocumented.

We will proceed with using *staged processing* for now.


Suppose we have a transform that repeats a function n times

```python
def repeat(func : Callable, n:int) -> Callable:
    def new_func(*args, **kwargs):
        for _ in range(n):
            args = func(*args, **kwargs)
        return args
    return new_func
```

We can start by creating the primitive itself:

```python
repeat_prim = jax.core.Primitive("repeat")
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
        return repeat_prim.bind(n, *args, jaxpr=jaxpr)
    return new_func
```

Several things to notice about this code.  First, we have to make the jaxpr from a function with any keyword arguments
already bound.  `jax.make_jaxpr` does not currently accept keyword arguments for the function, so we need to pre-bind them.
Next, we decided to make the integer `n` a traceable parameter instead of metadata. We could have chosen to make
`n` metadata instead.  This way, we can compile our function once for different integers `n`, and it is in line with how
catalyst treats `for_loop` and `while_loop`.  If the function produced outputs of different types and shapes than the inputs,
we would have to treat `n` like metadata and re-compile for different integers `n`.

Now we can define the implementation for our primitive.

```python
@repeat_prim.def_impl
def _(n, *args, jaxpr):
    for _ in range(n):
        args = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
    return args
```

Here we use `jax.core.eval_jaxpr` to execute the jaxpr with concrete arguments. If we had instead used
*on the fly processing*, we could have simply executed the stored function, but when using *staged processing*, we need
to directly evaluate the jaxpr instead.

```python
@repeat_prim.def_abstract_eval
def _(n, *args, jaxpr):
    return args
```

Now that we have all the parts, we can see it in action:

```pycon
>>> def func(x, y, y_coeff=1):
...     return (x + 1, y_coeff * y)
>>> def workflow(x):
...     return repeat(func, 2)(x, 2.0, y_coeff=2.0)
>>> workflow(0.5)
[Array(2.5, dtype=float32, weak_type=True),
 Array(8., dtype=float32, weak_type=True)]
>>> jax.make_jaxpr(workflow)(0.5)
{ lambda ; a:f32[]. let
    b:f32[] c:f32[] = repeat[
      jaxpr={ lambda ; d:f32[] e:f32[]. let
          f:f32[] = mul d e
          g:f32[] = mul 1.0 e
        in (f, g) }
    ] 2 a 2.0
  in (b, c) }
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
        primitive = jax.core.Primitive(cls.__name__)

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

jax.core.raise_to_shaped_mappings[AbstractPrimitiveClass] = lambda aval, _: aval
```

Now we can redefine our class to use this abstract class


```python
class PrimitiveMeta2(type):

    def __init__(cls, *args, **kwargs):
        # here we set up the primitive
        primitive = jax.core.Primitive(cls.__name__)

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
