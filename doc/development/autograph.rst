Guide for AutoGraph for plxpr capture
=====================================

When capturing PennyLane programs as a plxpr instance using AutoGraph, you
can represent quantum programs with **structure**. That is, you can use
classical control flow (such as conditionals and loops) with quantum operations
and measurements, and this structure is captured and preserved in the plxpr
representation.

PennyLane provides various high-level functions, such as :func:`~.cond`,
:func:`~.for_loop`, and :func:`~.while_loop`, that work with native PennyLane
quantum operations. However, it can sometimes take a bit of work to rewrite
existing Python code using these specific control flow functions. An experimental
feature of PennyLane capture, AutoGraph, instead allows Pennylane capture to work
with **native Python control flow**, such as if statements and for loops.

Here, we'll aim to provide an overview of AutoGraph, as well as various
restrictions and constraints you may discover.

.. note::

    When converting code in these examples, we will use the `make_plxpr` function,
    which uses AutoGraph by default. When creating the initial plxpr representation,
    we must call the constructor function produced by `make_plxpr` with some initial
    values, which should have the same type and shape as the values we intend to use:

    .. code-block:: python

        def f(x):
            if x > 5:
                x = x ** 2
            return x


    >>> plxpr = make_plxpr(f)(0.0)

    Once the plxpr representation is created, we can evaluate it using

    >>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 5.3)
    [Array(28.09, dtype=float64, weak_type=True)]



Using AutoGraph
---------------

The AutoGraph feature in PennyLane is supported by the ``diastatic-malt`` package, a standalone
fork of the AutoGraph module in TensorFlow (
`official documentation <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md>`_
).

The :func:`~pennylane.capture.make_plxpr` function uses AutoGraph by default.

.. code-block:: python

    dev = qml.device("lightning.qubit", wires=4)

    @qml.qnode(dev)
    def cost(weights, data):
        qml.AngleEmbedding(data, wires=range(4))

        for x in weights:

            for j, p in enumerate(x):
                if p > 0:
                    qml.RX(p, wires=j)
                elif p < 0:
                    qml.RY(p, wires=j)

            for j in range(4):
                qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

        return qml.expval(qml.PauliZ(0) + qml.PauliZ(3))

>>> weights = jnp.linspace(-1, 1, 20).reshape([5, 4])
>>> data = jnp.ones([4])
>>> cost(weights, data)
Array(0.30455313, dtype=float64)

This would be equivalent to writing the following program, without using
AutoGraph, but instead using :func:`~.cond` and :func:`~.for_loop`:

.. code-block:: python

    @qjit(autograph=False)
    @qml.qnode(dev)
    def cost(weights, data):
        qml.AngleEmbedding(data, wires=range(4))

        def layer_loop(i):
            x = weights[i]
            def wire_loop(j):

                @cond(x[j] > 0)
                def trainable_gate():
                    qml.RX(x[j], wires=j)

                @trainable_gate.else_if(x[j] < 0)
                def trainable_gate():
                    qml.RY(x[j], wires=j)

                trainable_gate()

            def cnot_loop(j):
                qml.CNOT(wires=[j, jnp.mod((j + 1), 4)])

            for_loop(0, 4, 1)(wire_loop)()
            for_loop(0, 4, 1)(cnot_loop)()

        for_loop(0, jnp.shape(weights)[0], 1)(layer_loop)()
        return qml.expval(qml.PauliZ(0) + qml.PauliZ(3))

>>> cost(weights, data)
Array(0.30455313, dtype=float64)

We can verify that the control flow is being correctly captured and
converted is to examine the plxpr representation of the compiled
program:

>>> g.jaxpr
{ lambda ; a:f64[] b:i64[]. let
    c:f64[] = for[
      apply_reverse_transform=False
      body_jaxpr={ lambda ; d:i64[] e:f64[]. let
          f:bool[] = gt e 5.0
          g:f64[] = cond[
            branch_jaxprs=[
              { lambda ; a:f64[] b_:f64[]. let c:f64[] = integer_pow[y=2] a in (c,) },
              { lambda ; a_:f64[] b:f64[]. let c:f64[] = integer_pow[y=3] b in (c,) }
            ]
          ] f e e
          h:f64[] = add e g
        in (h,) }
      body_nconsts=0
    ] 0 b 1 0 a
  in (c,) }

Here, we can see the for loop contained within the ``qcond`` operation, and
the two branches of the ``if`` statement represented by the ``branch_jaxprs``
list.

Currently, AutoGraph supports converting the following Python statements:

- ``if`` statements (including ``elif`` and ``else``)
- ``for`` loops
- ``while`` loops

``break`` and ``continue`` statements are currently not supported. The logical operators
``and``, ``or`` and ``not`` are currently unsupported.

Nested functions
----------------

AutoGraph will continue to work even when the function
itself calls nested functions. All functions called within the
top-level function will also have Python control flow captured
and converted by AutoGraph.

In addition, built-in functions from ``jax``, ``pennylane``, and ``catalyst``
are automatically *excluded* from the AutoGraph conversion.

.. code-block:: python

    def f(x):
        if x > 5:
            y = x ** 2
        else:
            y = x ** 3
        return y

    def g(x, n):
        for i in range(n):
            x = x + f(x)
        return x

>>> plxpr = make_plxpr(g)(0.0, 1)  # initialize with arguments of correct type and shape
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.4, 6)
[Array(22.14135448, dtype=float64)]


If statements
-------------

While most ``if`` statements you may write in Python will be automatically
converted, there are some important constraints and restrictions to be aware of.

Return statements
~~~~~~~~~~~~~~~~~

Return statements are generally supported inside of ``if``/``elif``/``else`` statements,
however, the returned values require a matching shape and structure across branches.

For example, consider the following pattern, where two different array dimensions are returned
from each branch:

.. code-block:: python

    def f(x):
        if x > 5:
            return jnp.array([1, 2])
        return 0

This will generate the following error:

>>> make_plxpr(f)(0)
ValueError: Mismatch in output abstract values in false branch #0 at position 1:
ShapedArray(int64[], weak_type=True) vs ShapedArray(int64[2])

Another example is the use of different *structure* across branches. The structure of a function
output is defined by things like the number of results, the containers used like lists or
dictionaries, or more generally any (compile-time) PyTree metadata.

Different branches must assign the same type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different branches of an if statement must always assign variables with the same type across branches,
if those variables are used in the outer scope (external variables). The type must be the same in the sense
that the *structure* of the variable should not change across branches, and the dtypes must match.

In particular, this requires that if an external variable is assigned an array in one
branch, other branches must also assign arrays of the same shape:

>>> def f(x):
...     if x > 1:
...         y = jnp.array([0.1, 0.2])
...     else:
...         y = jnp.array([0.4, 0.5, -0.1])
...     return jnp.sum(y)
>>> make_plxpr(f)(0.5)
ValueError: Mismatch in output abstract values in false branch #0 at position 0: ShapedArray(float64[3]) vs ShapedArray(float64[2])s

>>> def f(x):
...     if x > 1:
...         y = jnp.array([0.1, 0.2, 0.3])
...     else:
...         y = jnp.array([0.4, 0.5, -0.1])
...     return jnp.sum(y)
>>> plxpr = make_plxpr(f)(0.5)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.5)
[Array(0.8, dtype=float64)]

More generally, this also applies to common container classes such as
``dict``, ``list``, and ``tuple``. If one branch assigns an external variable,
then all other branches must also assign the external variable with the same
type, nested structure, number of elements, element types, and array shapes.

Changing a variable type
~~~~~~~~~~~~~~~~~~~~~~~~

We can change the type of an existing variable ``y``, as long as we make sure to change it in all branches.
This means will need to include an ``else`` statement to also change the type:

>>> def f(x):
...     y = -1.0
...     if x > 5:
...         y = 4
...     return y
>>> plxpr = make_plxpr(f)(0.5)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7.0)
ValueError: Mismatch in output abstract values in false branch #0 at position 0: ShapedArray(float64[], weak_type=True) vs ShapedArray(int64[], weak_type=True)

Even if we want to keep the value in the ``else`` condition, we need to update it to the new data type:

>>> def f(x):
...     y = -1.0
...     if x > 5:
...         y = 4
...     else:
...         y = -1
...     return y
>>> plxpr = make_plxpr(f)(0.5)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7.0)
Array(-1, dtype=int64)

Compatible type assignments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within an if statement, variable assignments must include JAX compatible
types (Booleans, Python numeric types, JAX arrays, and PennyLane quantum
operators). Non-compatible types (such as strings) used
after the if statement will result in an error:

>>> def f(x):
...     if x > 5:
...         y = "a"
...     else:
...         y = "b"
...     return y
>>> plxpr = make_plxpr(f)(0.5)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7.0)
TypeError: Value 'a' with type <class 'str'> is not a valid JAX type

For loops
---------

Most ``for`` loop constructs will be properly captured and compiled by AutoGraph.

.. code-block:: python

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def f():
        for x in [0, 1, 2]:
            qml.RY(x * jnp.pi / 4, wires=0)
        return qml.expval(qml.PauliZ(0))

>>> plxpr = make_plxpr(f)()
>>> jax.core.eval_jaxpr(plxpr.jaxpr, jaxpr.consts)
[Array(-0.70710678, dtype=float64)]

This includes automatic unpacking and enumeration through JAX arrays:

>>> def f(weights):
...     z = 0.
...     for i, (x, y) in enumerate(weights):
...         z = i * x + i ** 2 * y
...     return z
>>> weights = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).T
>>> plxpr = make_plxpr(f)(weights)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, [], weights)
Array(8.4, dtype=float64)

The Python ``range`` function is also fully supported by AutoGraph, even when
its input is a **dynamic variable** (i.e., its numeric value is only known at
runtime):

>>> def f(n):
...     x = -jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> plxpr = make_plxpr(f)(0)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, [], 1000)
[Array(0.57771558, dtype=float64, weak_type=True)]

Indexing within a loop
~~~~~~~~~~~~~~~~~~~~~~

Indexing arrays within a for loop will generally work, but care must be taken.

For example, using a for loop with static bounds to index a JAX array is straightforward:

>>> dev = qml.device("default.qubit", wires=3)
... @qml.qnode(dev)
... def f(x):
...     for i in range(3):
...         qml.RX(x[i], wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> weights = jnp.array([0.1, 0.2, 0.3])
>>> plxpr = make_plxpr(f)(weights)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, [], weights)
[Array(0.99500417, dtype=float64)]

However, indexing within a for loop with AutoGraph will require that the object indexed is
a JAX array or dynamic runtime variable.

If the array you are indexing within the for loop is not a JAX array
or dynamic variable, an error will be raised:

... @qml.qnode(dev)
... def f():
...     x = [0.1, 0.2, 0.3]
...     for i in range(3):
...         qml.RX(x[i], wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> plxpr = make_plxpr(f)()
AutoGraphError: Tracing of an AutoGraph converted for loop failed with an exception:
  TracerIntegerConversionError:    The __index__() method was called on traced array with shape int64[]
    The error occurred while tracing the function functional_for at /Users/lillian.frederiksen/pennylane/pennylane/capture/autograph/ag_primitives.py:176 for jit. This concrete value was not available in Python because it depends on the value of the argument i.
    See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerIntegerConversionError

To allow AutoGraph conversion to work in this case, simply convert the list to
a JAX array:

... @qml.qnode(dev)
... def f():
...     x = jnp.array([0.1, 0.2, 0.3])
...     for i in range(3):
...         qml.RX(x[i], wires=i)
...     return qml.expval(qml.PauliZ(0))
>>> plxpr = make_plxpr(f)()
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts)
[Array(0.99500417, dtype=float64)]

If the object you are indexing **cannot** be converted to a JAX array, it is not possible for AutoGraph to capture this for loop.

If you are updating elements of the array, this must be done using the JAX `.at` and `.set` syntax.

>>> def f():
...     my_list = jnp.empty(2, dtype=int)
...     for i in range(2):
...         my_list = my_list.at[i].set(i)  # *not* my_list[i] = i
...     return my_list
>>> plxpr = make_plxpr(f)()
>>> jax.core.eval_jaxpr(plxpr.jaxpr, [])
Array([0, 1], dtype=int64)


Dynamic indexing
~~~~~~~~~~~~~~~~

Indexing into arrays where the for loop has **dynamic bounds** (that is, where
the size of the loop is set by a dynamic runtime variable) will also work, as long
as the object indexed is a JAX array:

>>> @qml.qnode(dev)
... def f(n):
...     x = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
...     for i in range(n):
...         qml.RY(x[i], wires=0)
...     return qml.expval(qml.PauliZ(0))
>>> plxpr = make_plxpr(f)(0)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 2)
Array(0.70710678, dtype=float64)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 3)
Array(-0.70710678, dtype=float64)

However AutoGraph conversion will fail if the object being indexed by the
loop with dynamic bounds is **not** a JAX array, because you cannot index
standard Python objects with dynamic variables. Ensure that all objects that
]are indexed within dynamic for loops are JAX arrays.

Break and continue
~~~~~~~~~~~~~~~~~~

Within a for loop, control flow statements ``break`` and ``continue``
are not currently supported.


Updating and assigning variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For loops that update variables can also be converted with AutoGraph:

>>> def f(x):
...     for y in [0, 4, 5]:
...         x = x + y
...     return x
>>> f(4)
Array(13, dtype=int64)

However, like with conditionals, a similar restriction applies: variables
which are updated across iterations of the loop must have a JAX compilable
type (Booleans, Python numeric types, and JAX arrays).

You can also utilize temporary variables within a for loop:

>>> @qjit(autograph=True)
... def f(x):
...     for y in [0, 4, 5]:
...         c = 2
...         x = x + y * c
...     return x
>>> f(4)
Array(22, dtype=int64)

Temporary variables used inside a loop --- and that are **not** passed to a
function within the loop --- do not have any type restrictions.

While loops
-----------

Most ``while`` loop constructs will be properly captured and compiled by
AutoGraph:

>>> @qjit(autograph=True)
... def f(param):
...     n = 0.
...     while param < 0.5:
...         param *= 1.2
...         n += 1
...     return n
>>> f(0.1)
Array(9., dtype=float64)

Break and continue
~~~~~~~~~~~~~~~~~~

Within a while loop, control flow statements ``break`` and ``continue``
are not currently supported. Usage will result in an error:


Updating and assigning variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As with for loops, while loops that update variables can also be converted with AutoGraph:

>>> @qjit(autograph=True)
... def f(x):
...     while x < 5:
...         x = x + 2
...     return x
>>> f(4)
Array(6.4, dtype=float64)

However, like with conditionals, a similar restriction applies: variables
which are updated across iterations of the loop must have a JAX compilable
type (Booleans, Python numeric types, and JAX arrays).

You can also utilize temporary variables within a while loop:

>>> @qjit(autograph=True)
... def f(x):
...     while x < 5:
...         c = "hi"
...         x = x + 2 * len(c)
...     return x
>>> f(4)
Array(8.4, dtype=float64)

Temporary variables used inside a loop --- and that are **not** passed to a
function within the loop --- do not have any type restrictions.

Logical statements
------------------

AutoGraph in PennyLane currently does not provide support for capturing logical statements that involve dynamic variables --- that is,
statements involving ``and``, ``not``, and ``or`` that return booleans.

Debugging
---------

We've seen examples in the above code where we have used the jaxpr representation
of the compiled function in order to verify that AutoGraph is correctly capturing
the control flow. This can be a useful tool in debugging issues.

In addition, the function :func:`~.autograph_source` is provided,
and allows you to view the converted Python code generated by AutoGraph:

>>> def f(n):
...     x = - jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> plxpr = make_plxpr(f)(0)
>>> print(qml.capture.autograph.autograph_source(f))
def ag__f(n):
    with ag__.FunctionScope('f', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=ag__.Feature.BUILTIN_FUNCTIONS, internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()
        x = -ag__.converted_call(ag__.ld(jnp).log, (ag__.ld(n),), None, fscope)

        def get_state():
            return (x,)

        def set_state(vars_):
            nonlocal x
            x, = vars_

        def loop_body(itr):
            nonlocal x
            k = itr
            x = ag__.ld(x) + 1 / ag__.ld(k)
        k = ag__.Undefined('k')
        ag__.for_stmt(ag__.converted_call(ag__.ld(range), (1, ag__.ld(n) + 1), None, fscope), None, loop_body, get_state, set_state, ('x',), {'iterate_names': 'k'})
        try:
            do_return = True
            retval_ = ag__.ld(x)
        except:
            do_return = False
            raise
        return fscope.ret(retval_, do_return)


Native Python control flow without AutoGraph
--------------------------------------------

It's important to note that native Python control flow --- in cases where the
control flow parameters are static --- will continue to work with
PennyLane **without** AutoGraph. However, if AutoGraph is not enabled, such
control flow will be evaluated at compile time, and not preserved in the
compiled program.


Let's consider an example where a for loop is evaluated at compile time:

>>> @qjit
... def f(x):
...     for i in range(2):
...         print(i, x)
...         x = x / 2
...     return x ** 2
>>> f(2.)
0 Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
1 Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
Array(0.25, dtype=float64)

Here, the for loop is evaluated at compile time (notice the multiple tracers
that have been printed out during program capture --- one for each loop!),
rather than runtime.

For more details, see the :ref:`compile-time vs. runtime <compile_time>`
documentation.


In-place JAX array updates
--------------------------

To update array values when using JAX, the `JAX syntax for array assignment
<https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#array-updates-x-at-idx-set-y>`__
(which uses the array ``at`` and ``set`` methods) must be used:

.. code-block:: python

    @qjit(autograph=True, abstracted_axes=(0,))
    def f(x):
        first_dim = x.shape[0]
        result = jnp.empty((first_dim,), dtype=x.dtype)

        for i in range(first_dim):
            result = result.at[i].set(x[i] * 2)

        return result

>>> f(jnp.array([0.1, 0.2, 0.3]))
Array([0.2, 0.4, 0.6], dtype=float64)

However, if updating a single static index or slice of the array, then Autograph supports conversion
of standard Python array assignment syntax:

.. code-block:: python

    @qjit(autograph=True)
    def f(x, y):
        y[1:10:2] = x  # static slice index
        y[0] = x[-1] ** 2   # single integer index
        return y

>>> x = jnp.linspace(2, 5, 5)
>>> y = jnp.zeros([11])
>>> f(x, y)
Array([25.,  2.,  0.,  2.75,  0.,  3.5,  0.,  4.25,  0., 5.,  0.], dtype=float64)

Under the hood, Catalyst converts anything coming in the latter notation into the former one.

Similarly, to update array values with an operation when using JAX, the JAX syntax for array
update (which uses the array `at` and the `add`, `multiply`, etc. methods) must be used:

>>> @qjit(autograph=True)
... def f(x):
...     first_dim = x.shape[0]
...     result = jnp.copy(x)
...
...     for i in range(first_dim):
...         result = result.at[i].multiply(2)
...
...     return result

Again, if updating a single index or slice of the array, then Autograph supports conversion of
standard Python array operator assignment syntax for the equivalent in-place expressions
listed in the `JAX documentation for jax.numpy.ndarray.at
<https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at>`__:

>>> @qjit(autograph=True)
... def f(x):
...     first_dim = x.shape[0]
...     result = jnp.copy(x)
...
...     for i in range(first_dim):
...         result[i] *= 2
...
...     return result

Under the hood, Catalyst converts anything coming in the latter notation into the former one.

The list of supported operators includes:
- ``=`` (set)
- ``+=`` (add)
- ``-=`` (add with negation)
- ``*=`` (multiply)
- ``/=`` (divide)
- ``**=`` (power)
