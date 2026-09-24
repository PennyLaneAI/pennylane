:orphan:

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
existing Python code using these specific control flow functions. AutoGraph is an experimental
feature of PennyLane capture that allows Pennylane capture to work
with **native Python control flow**, such as ``if`` statements and ``for`` loops.

Here, we'll aim to provide an overview of AutoGraph, as well as various
restrictions and constraints you may discover.

.. note::

    When converting code in these examples, we will use the :func:`~.autograph.make_plxpr` function,
    which uses AutoGraph by default.

    When creating the initial plxpr representation, we must call the constructor function produced
    by :func:`~.autograph.make_plxpr` with some initial values, which should have the same type and
    shape as the values we intend to use when evaluating:

    .. code-block:: python

        from pennylane.capture import make_plxpr

        def f(x):
            if x > 5:
                x = x ** 2
            return x


    >>> plxpr = make_plxpr(f)(0.0)  # x will be a float

    Once the plxpr representation is created, we can evaluate it using

    >>> from jax.core import eval_jaxpr
    >>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 5.3)  # evaluate f(5.3)
    [Array(28.09, dtype=float64, weak_type=True)]



Using AutoGraph
---------------

The AutoGraph feature in PennyLane is supported by the ``diastatic-malt`` `package <https://github.com/PennyLaneAI/diastatic-malt>`_, a standalone
fork of the AutoGraph module in TensorFlow (`official documentation <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md>`_).

The :func:`~pennylane.capture.make_plxpr` function uses AutoGraph by default. Consider a function using
Python control flow:

.. code-block:: python

    dev = qp.device("default.qubit", wires=4)

    @qp.qnode(dev)
    def cost(weights, data):

        for w in dev.wires:
            qp.X(w)

        for x in weights:

            for j, p in enumerate(x):
                if p > 0:
                    qp.RX(p, wires=j)
                elif p < 0:
                    qp.RY(p, wires=j)

            for j in range(4):
                qp.CNOT(wires=[j, jnp.mod((j + 1), 4)])

        return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))

While this function cannot be captured directly because there is control flow that depends on the values of the function's inputs (the inputs are treated as JAX tracers at capture time, which don't have concrete values) it can be captured by converting to native PennyLane syntax
via AutoGraph. This is the default behaviour of :func:`~.autograph.make_plxpr`.

>>> weights = jnp.linspace(-1, 1, 20).reshape([5, 4])
>>> data = jnp.ones([4])
>>> plxpr = make_plxpr(cost)(weights, data)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, weights, data)
[Array(-0.45165857, dtype=float64)]

This would be equivalent to writing the following program, without using
AutoGraph, but instead using :func:`~.cond` and :func:`~.for_loop`:

.. code-block:: python

    @qp.qnode(dev)
    def cost(weights, data):

        @qp.for_loop(0, 4, 1)
        def initialize_loop(w):
            qp.X(w)

        @qp.for_loop(0, jnp.shape(weights)[0], 1)
        def layer_loop(i):
            x = weights[i]

            @qp.for_loop(0, 4, 1)
            def wire_loop(j):

                @qp.cond(x[j] > 0)
                def trainable_gate():
                    qp.RX(x[j], wires=j)

                @trainable_gate.else_if(x[j] < 0)
                def trainable_gate():
                    qp.RY(x[j], wires=j)

                trainable_gate()

            @qp.for_loop(0, 4, 1)
            def cnot_loop(j):
                qp.CNOT(wires=[j, jnp.mod((j + 1), 4)])

            wire_loop()
            cnot_loop()

        initialize_loop()
        layer_loop()
        return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))

Once converted to native PennyLane control flow manually, AutoGraph is no longer needed:

>>> plxpr = make_plxpr(cost, autograph=False)(weights, data)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, weights, data)
[Array(-0.45165857, dtype=float64)]

Currently, AutoGraph supports converting the following Python statements:

- ``if`` statements (including ``elif`` and ``else``)
- ``for`` loops
- ``while`` loops

``break`` and ``continue`` statements are currently not supported. 

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
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.4, 6)
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
        return jnp.array([0])

This will generate the following error:

>>> make_plxpr(f)(0)
ValueError: Mismatch in output abstract values in false branch #0 at position 1:
ShapedArray(int64[1]) vs ShapedArray(int64[2])

This is relevant for any example that uses different *structure* across branches. The structure of a function
output is defined by things like the number of results, the containers used like lists or
dictionaries, or more generally any (compile-time) PyTree metadata.

Different branches must assign the same type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different branches of an ``if`` statement must always assign variables with the same type across branches,
if those variables are used in the outer scope (external variables). The type must be the same in the sense
that the *structure* of the variable should not change across branches, and the dtypes must match.

Consider this function, which differs in the type of the elements in ``y`` in different logic branches:

>>> def f(x):
...     if x > 1:
...         y = jnp.array([1.0, 2.0, 3.0])
...     else:
...         y = jnp.array([4, 5, 6])
...     return jnp.sum(y)
>>> make_plxpr(f)(0.5)
ValueError: Mismatch in output abstract values in false branch #0 at position 0: ShapedArray(int64[3]) vs ShapedArray(float64[3])

Instead, all possible outcomes for ``y`` at the end of the if/else block need to have the same shape, type, etc:

>>> def f(x):
...     if x > 1:
...         y = jnp.array([1.0, 2.0, 3.0])
...     else:
...         y = jnp.array([4.0, 5.0, 6.0])
...     return jnp.sum(y)
>>> plxpr = make_plxpr(f)(0.5)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.5)
[Array(15., dtype=float64)]

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
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7.0)
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
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7.0)
Array(-1, dtype=int64)

Compatible type assignments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within an ``if`` statement, variable assignments must include JAX compatible
types (Booleans, Python numeric types, JAX arrays, and PennyLane quantum
operators). Non-compatible types (such as strings) used
after the ``if`` statement will result in an error:

>>> def f(x):
...     if x > 5:
...         y = "a"
...     else:
...         y = "b"
...     return y
>>> plxpr = make_plxpr(f)(0.5)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7.0)
TypeError: Value 'a' with type <class 'str'> is not a valid JAX type

For loops
---------

Most ``for`` loop constructs will be properly captured and compiled by AutoGraph.

.. code-block:: python

    dev = qp.device("default.qubit", wires=1)

    @qp.qnode(dev)
    def f():
        for x in jnp.array([0, 1, 2]):
            qp.RY(x * jnp.pi / 4, wires=0)
        return qp.expval(qp.PauliZ(0))

>>> plxpr = make_plxpr(f)()
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts)
[Array(-0.70710678, dtype=float64)]

This includes automatic unpacking and enumeration through JAX arrays:

>>> def f(weights):
...     z = 0.
...     for i, (x, y) in enumerate(weights):
...         z = i * x + i ** 2 * y
...     return z
>>> weights = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).T
>>> plxpr = make_plxpr(f)(weights)
>>> eval_jaxpr(plxpr.jaxpr, [], weights)
Array(8.4, dtype=float64)

The Python ``range`` function is also supported by AutoGraph, even when
its input is a **dynamic variable** (i.e., its numeric value is only known at
runtime):

>>> def f(n):
...     x = -jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> plxpr = make_plxpr(f)(0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1000)
[Array(0.57771558, dtype=float64, weak_type=True)]

Indexing within a loop
~~~~~~~~~~~~~~~~~~~~~~

Indexing arrays within a ``for`` loop will generally work, but care must be taken.

For example, using a ``for`` loop with static bounds to index a JAX array is straightforward:

>>> dev = qp.device("default.qubit", wires=3)
... @qp.qnode(dev)
... def f(x):
...     for i in range(3):
...         qp.RX(x[i], wires=i)
...     return qp.expval(qp.PauliZ(0))
>>> weights = jnp.array([0.1, 0.2, 0.3])
>>> plxpr = make_plxpr(f)(weights)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, weights)
[Array(0.99500417, dtype=float64)]

However, indexing within a ``for`` loop with AutoGraph will require that the object indexed is
a JAX array or dynamic runtime variable.

If the array you are indexing within the ``for`` loop is not a JAX array
or dynamic variable, an error will be raised:

>>> @qp.qnode(dev)
... def f():
...     x = [0.1, 0.2, 0.3]
...     for i in range(3):
...         qp.RX(x[i], wires=i)
...     return qp.expval(qp.PauliZ(0))
>>> plxpr = make_plxpr(f)()
AutoGraphError: Tracing of an AutoGraph converted for loop failed with an exception:
  TracerIntegerConversionError:    The __index__() method was called on traced array with shape int64[]
    The error occurred while tracing the function functional_for [...]

To allow AutoGraph conversion to work in this case, simply convert the list to
a JAX array:

>>> @qp.qnode(dev)
... def f():
...     x = jnp.array([0.1, 0.2, 0.3])
...     for i in range(3):
...         qp.RX(x[i], wires=i)
...     return qp.expval(qp.PauliZ(0))
>>> plxpr = make_plxpr(f)()
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts)
[Array(0.99500417, dtype=float64)]

If the object you are indexing **cannot** be converted to a JAX array, it is not possible for AutoGraph to capture this ``for`` loop.

Dynamic indexing
~~~~~~~~~~~~~~~~

Indexing into arrays where the ``for`` loop has **dynamic bounds** (that is, where
the size of the loop is set by a dynamic runtime variable) will also work, as long
as the object indexed is a JAX array:

>>> @qp.qnode(dev)
... def f(n):
...     x = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
...     for i in range(n):
...         qp.RY(x[i], wires=0)
...     return qp.expval(qp.PauliZ(0))
>>> plxpr = make_plxpr(f)(0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 2)
Array(0.70710678, dtype=float64)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 3)
Array(-0.70710678, dtype=float64)

However AutoGraph conversion will fail if the object being indexed by the
loop with dynamic bounds is **not** a JAX array, because you cannot index
standard Python objects with dynamic variables. Ensure that all objects that
are indexed within dynamic ``for`` loops are JAX arrays.

Break and continue
~~~~~~~~~~~~~~~~~~

Within a ``for`` loop, control flow statements ``break`` and ``continue``
are not currently supported.

Updating and assigning variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``for`` loops that update variables can also be converted with AutoGraph:

>>> def f(x):
...     for y in [0, 4, 5]:
...         x = x + y
...     return x
>>> plxpr = make_plxpr(f)(0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 3)
[Array(12, dtype=int64)]

However, like with conditionals, a similar restriction applies: variables
which are updated across iterations of the loop must have a JAX compilable
type (Booleans, Python numeric types, and JAX arrays).

You can also utilize temporary variables within a ``for`` loop:

>>> def f(x):
...     for y in [0, 4, 5]:
...         c = 2
...         x = x + y * c
...     return x
>>> plxpr = make_plxpr(f)(0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 4)
[Array(22, dtype=int64)]

Temporary variables used inside a loop --- and that are **not** passed to a
function within the loop --- do not have any type restrictions.

While loops
-----------

Most ``while`` loop constructs will be properly captured and compiled by
AutoGraph:

>>> def f(param):
...     n = 0.
...     while param < 0.5:
...         param *= 1.2
...         n += 1
...     return n
>>> plxpr = make_plxpr(f)(0.0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 0.1)
[Array(9., dtype=float64, weak_type=True)]


Indexing within a loop
~~~~~~~~~~~~~~~~~~~~~~

Indexing arrays within a ``while`` loop will generally work, but care must be taken.

For example, using a ``while`` loop variable to index a JAX array is straightforward:

>>> dev = qp.device("default.qubit", wires=3)
... @qp.qnode(dev)
... def f(x):
...     i = 0
...     while i < 3:
...         qp.RX(x[i], wires=i)
...         i += 1
...     return qp.expval(qp.PauliZ(0))
>>> weights = jnp.array([0.1, 0.2, 0.3])
>>> plxpr = make_plxpr(f)(weights)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, weights)
[Array(0.99500417, dtype=float64)]

However, indexing within a ``while`` loop with AutoGraph will require that the object indexed is
a JAX array:

>>> @qp.qnode(dev)
... def f():
...     x = [0.1, 0.2, 0.3]
...     i = 0
...     while i < 3:
...         qp.RX(x[i], wires=i)
...         i += 1
...     return qp.expval(qp.PauliZ(0))
>>> plxpr = make_plxpr(f)()
TracerIntegerConversionError: The __index__() method was called on traced array with shape int64[].
The error occurred while tracing the function functional_while at [...]

To allow AutoGraph conversion to work in this case, simply convert the list to a JAX array:

>>> @qp.qnode(dev)
... def f():
...     x = jnp.array([0.1, 0.2, 0.3])
...     i = 0
...     while i < 3:
...         qp.RX(x[i], wires=i)
...         i += 1
...     return qp.expval(qp.PauliZ(0))
>>> plxpr = make_plxpr(f)()
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts)
[Array(0.99500417, dtype=float64)]

If the object you are indexing **cannot** be converted to a JAX array, it is not possible for AutoGraph to capture this ``while`` loop.

Break and continue
~~~~~~~~~~~~~~~~~~

Within a ``while`` loop, control flow statements ``break`` and ``continue``
are not currently supported.


Updating and assigning variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As with ``for`` loops, ``while`` loops that update variables can also be converted with AutoGraph:

>>> def f(x):
...     while x < 5:
...         x = x + 2
...     return x
>>> plxpr = make_plxpr(f)(0.0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 4.4)
[Array(6.4, dtype=float64, weak_type=True)]

However, like with conditionals, a similar restriction applies: variables
which are updated across iterations of the loop must have a JAX compilable
type (Booleans, Python numeric types, and JAX arrays).

You can also utilize temporary variables within a ``while`` loop:

>>> def f(x):
...     while x < 5:
...         c = "hi"
...         x = x + 2 * len(c)
...     return x
>>> plxpr = make_plxpr(f)(0.0)
>>> eval_jaxpr(plxpr.jaxpr, plxpr.consts, 4.4)
[Array(8.4, dtype=float64, weak_type=True)]

Temporary variables used inside a loop—and that are **not** passed to a
function within the loop—do not have any type restrictions.

A caveat regarding updating variables in a ``while`` loop is that it is not possible to
update variables inside the loop test statement. For example, while the following
works in standard Python:

>>> def fn(limit):
...     i = 0
...     y = 0
...     while (i := y) < limit:
...         y += 1
...     return i
>>> fn(10)
10

any updates to the variables inside the ``while`` test function (in this case ``(i := y)``)
will be ignored by AutoGraph:

>>> plxpr = make_plxpr(fn)(0)
>>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 10)
[0]

Debugging
---------

One useful tool in debugging issues is to examine the plxpr representation
of the compiled function, in order to verify that AutoGraph is correctly capturing
the control flow. For example, consider:

.. code-block:: python

    def f(x, n):

        for i in range(n):

            if x > 5:
                y = x ** 2
            else:
                y = x ** 3

            x = x + y

        return x

We can verify that the control flow is being correctly captured and
converted by examining the plxpr representation of the compiled
program:

>>> make_plxpr(f)(0.0, 0)
{ lambda ; a:f64[] b:i64[]. let
    c:f64[] = for_loop[
      args_slice=slice(0, None, None)
      consts_slice=slice(0, 0, None)
      jaxpr_body_fn={ lambda ; d:i64[] e:f64[]. let
          f:bool[] = gt e 5.0
          g:f64[] = cond[
            args_slice=slice(4, None, None)
            consts_slices=[slice(2, 3, None), slice(3, 4, None)]
            jaxpr_branches=[{ lambda a:f64[]; . let b:f64[] = integer_pow[y=2] a in (b,) }, { lambda a:f64[]; . let b:f64[] = integer_pow[y=3] a in (b,) }]
          ] f True e e
          h:f64[] = add e g
        in (h,) }
    ] 0 b 1 a
  in (c,) }

Here, we can see the ``cond`` operation inside the ``for`` loop, and
the two branches of the ``if`` statement represented by the ``jaxpr_branches``
list.

In addition, the function :func:`~.autograph_source` is provided,
and allows you to view the converted Python code generated by AutoGraph:

>>> def f(n):
...     x = - jnp.log(n)
...     for k in range(1, n + 1):
...         x = x + 1 / k
...     return x
>>> plxpr = make_plxpr(f)(0)
>>> print(qp.capture.autograph.autograph_source(f))
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


.. warning::

    Nested functions are only lazily converted by AutoGraph. If the input includes nested
    functions, these won't be converted until the first time the function is traced. This
    is important to be aware of if examining the output of running autograph for debugging
    purposes. In an example like:

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

        ag_fn = make_plxpr(g)

    we can access ``autograph_source(g)``, but we will get an error for ``autograph_source(f)``:

    >>> autograph_source(f)
    AutoGraphError: The given function was not converted by AutoGraph. If you expect the given function to be converted, please submit a bug report.

    This is because it has only been lazily converted. To examine the inner function's Autograph
    conversion, we must trace the output function from `make_plxpr` with values at least once:

    >>> plxpr = ag_fn(0, 0)
    >>> autograph_source(f)
    def ag__f(x):
        with ag__.FunctionScope('f', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=False, optional_features=ag__.Feature.BUILTIN_FUNCTIONS, internal_convert_user_code=True)) as fscope:
        ...

Native Python control flow without AutoGraph
--------------------------------------------

It's important to note that native Python control flow—in cases where the
control flow parameters are static—will continue to work with
PennyLane **without** AutoGraph. However, if AutoGraph is not enabled, such
control flow will be evaluated at compile time, and not preserved in the
compiled program.

Let's consider an example where a ``for`` loop is evaluated at compile time:

>>> def f(x):
...     for i in range(2):
...         print(i, x)
...         x = x / 2
...     return x ** 2
>>> plxpr = make_plxpr(f, autograph=False)(0.0)
0 Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
1 Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>

>>> plxpr
{ lambda ; a:f64[]. let
    b:f64[] = div a 2.0
    c:f64[] = div b 2.0
    d:f64[] = integer_pow[y=2] c
  in (d,) }

Here, the loop is evaluated at compile time, rather than runtime. Notice the multiple tracers that
have been printed out during program capture—one for each loop—as well as the unrolling of the
loop in the resulting plxpr.

With AutoGraph, we instead get a single print of the tracers, and compile with a ``for`` loop that can be
evaluated at runtime:

>>> plxpr = make_plxpr(f, autograph=True)(0.0)
Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace(level=2/0)> Traced<ShapedArray(float64[], weak_type=True)>with<DynamicJaxprTrace(level=2/0)>

>>> plxpr
{ lambda ; a:f64[]. let
    b:f64[] = for_loop[
      args_slice=slice(0, None, None)
      consts_slice=slice(0, 0, None)
      jaxpr_body_fn={ lambda ; c:i64[] d:f64[]. let
          e:f64[] = div d 2.0
        in (e,) }
    ] 0 2 1 a
    f:f64[] = integer_pow[y=2] b
  in (f,) }
