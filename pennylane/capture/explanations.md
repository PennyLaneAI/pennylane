This documentation explains the principles behind `qml.capture.CaptureMeta`.


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
