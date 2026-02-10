# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains a utility for handling inputs with dynamically shaped arrays.
"""
from collections.abc import Callable

has_jax = True
try:
    import jax
    from jax._src.interpreters.partial_eval import TracingEqn


except ImportError as e:  # pragma: no cover
    has_jax = False


def _get_shape_for_array(x, abstract_shapes: list, previous_ints: list) -> dict:
    """
    Populate the dictionary of abstract axes for a single tensorlike.

    This dictionary has dimensions as keys, and an integer as the value.

    Examples of shape -> abstract axes:

    * ``(3,4) -> {}``
    * ``(tracer1, ) -> {0: 0}``
    * ``(tracer1, tracer1) -> {0: 0, 1: 0}``
    * ``(3, tracer1) -> {1: 0}``
    * ``(tracer1, 2, tracer2) -> {0: 0, 2: 1}``

    ``abstract_shapes`` contains all the tracers found in shapes.

    """
    dtype = getattr(x, "dtype", "float")
    if getattr(x, "shape", None) == () and jax.numpy.issubdtype(dtype, jax.numpy.integer):
        previous_ints.append(x)
        return {}

    abstract_axes = {}
    for i, s in enumerate(getattr(x, "shape", ())):  # pragma: no cover
        if not isinstance(s, int):  #  if not int, then abstract
            found = False
            # check if the shape tracer is one we have already encountered
            for previous_idx, previous_shape in enumerate(previous_ints):
                if s is previous_shape:
                    abstract_axes[i] = f"{previous_idx}_arg"
                    found = True
                    break
            if not found:
                for previous_idx, previous_shape in enumerate(abstract_shapes):
                    if s is previous_shape:
                        abstract_axes[i] = previous_idx
                        found = True
                        break
            # haven't encountered it, so add it to abstract_axes
            # and use new letter designation
            if not found:
                abstract_axes[i] = len(abstract_shapes)
                abstract_shapes.append(s)

    return abstract_axes


def determine_abstracted_axes(args):
    """Compute the abstracted axes and extract the abstract shapes from the arguments.

    Args:
        args (tuple): the arguments for a higher order primitive

    Returns:
        tuple, tuple: the corresponding abstracted axes and dynamic shapes

    Note that "dynamic shapes" only refers to the size of dimensions, but not the number of dimensions.
    Even with dynamic shapes mode enabled, we cannot change the number of dimensions.

    See the ``intro_to_dynamic_shapes.md`` document for more information on how dynamic shapes work.

    To make jaxpr from arguments with dynamic shapes, the ``abstracted_axes`` keyword argument must be set.
    Then, when calling the jaxpr, variables for the dynamic shapes must be passed.

    .. code-block:: python

        jax.config.update("jax_dynamic_shapes", True)

        def f(n):
            x = jax.numpy.ones((n,))
            abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes((x,))
            jaxpr = jax.make_jaxpr(jax.numpy.sum, abstracted_axes=abstracted_axes)(x)
            return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, x)


    For cases where the shape of an argument matches a previous argument like:

    >>> def f(i, x):
    ...    return x
    >>> def workflow(i):
    ...     args = (i, jax.numpy.ones((i, )))
    ...     abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes(args)
    ...     print("abstracted_axes: ", abstracted_axes)
    ...     print("abstract_shapes: ", abstract_shapes)
    ...     print("jaxpr: ", jax.make_jaxpr(f, abstracted_axes=abstracted_axes)(*args))
    >>> _ = jax.make_jaxpr(workflow)(2)
    abstracted_axes:  ({}, {0: '0_arg'})
    abstract_shapes:  []
    jaxpr:  { lambda ; a:i32[] b:f32[a]. let  in (b,) }

    We allow Jax to identify that the shape of ``b`` matches our first argument, ``a``. This is
    demonstrated by the fact that we do not have any additional ``abstract_shapes``, as it is already
    present in the call signature. The abstracted axis is also ``"0_arg"`` instead of ``0``.
    The ``"_arg"`` at the end indicates that the corresponding abstract axis
    was already in the argument loop.

    """
    assert has_jax
    if not jax.config.jax_dynamic_shapes:
        return None, ()

    args, structure = jax.tree_util.tree_flatten(args)

    abstract_shapes = []
    previous_ints = []
    # note: this function in-place mutates abstract_shapes and previous_ints
    # adding any additional abstract shapes found
    abstracted_axes = [_get_shape_for_array(a, abstract_shapes, previous_ints) for a in args]

    if not any(abstracted_axes):
        return None, ()

    abstracted_axes = jax.tree_util.tree_unflatten(structure, abstracted_axes)  # pragma: no cover
    return abstracted_axes, abstract_shapes  # pragma: no cover


def register_custom_staging_rule(
    primitive,
    get_outvars_from_params: Callable[[dict], list["jax.extend.core.Var"]],
    create_initial_env=None,
) -> None:
    """Register a custom staging rule for a primitive, where the output should match the variables retrieved by
    ``get_outvars_from_params``.

    Args:
        primitive (jax.extend.core.Primitive): a jax primitive we want to register a custom staging rule for
        get_outvars_from_params (Callable[[dict], list[jax.extend.core.Var]]): A function that takes in the equation's ``params``
            and returns ``jax.extend.core.Var`` we need to mimic for the primitives return.

    For example, the ``cond_prim`` will request its custom staging rule like:

    .. code-block:: python

        register_custom_staging_rule(cond_prim, lambda params: params['jaxpr_branches'][0].outvars)

    The return of any ``cond_prim`` will match the output variables of the first jaxpr branch.

    """
    # see https://github.com/jax-ml/jax/blob/9e62994bce7c7fcbb2f6a50c9ef89526cd2c2be6/jax/_src/lax/lax.py#L3538
    # and https://github.com/jax-ml/jax/blob/9e62994bce7c7fcbb2f6a50c9ef89526cd2c2be6/jax/_src/lax/lax.py#L208
    # for reference to how jax is handling staging rules for dynamic shapes in v0.4.28
    # JAX 0.6.2 to 0.7.0 introduced breaking changes in custom staging rules for dynamic shapes:
    # 1. DynamicJaxprTracer constructor now requires the var as 3rd argument (previously created internally)
    # 2. TracingEqn must be used instead of JaxprEqn for trace.frame.add_eqn
    #
    # This implementation creates vars first using trace.frame.newvar() before constructing
    # DynamicJaxprTracer instances, fixing dynamic shape support that was broken in JAX 0.7.0.
    # See pennylane/capture/jax_patches.py for related fixes to JAX's own staging rules.
    # See also capture/intro_to_dynamic_shapes.md for dynamic shapes documentation.

    def _tracer_and_outvar(
        jaxpr_trace,
        outvar: jax.extend.core.Var,
        env: dict[jax.extend.core.Var, jax.extend.core.Var],
    ):
        """
        Create a new tracer and return var from the true branch outvar.
        Returned vars are cached in env for use in future shapes
        """
        if not hasattr(outvar.aval, "shape"):
            # JAX 0.7.0: Create variable first, then pass to DynamicJaxprTracer
            new_var = jaxpr_trace.frame.newvar(outvar.aval)
            out_tracer = jax.interpreters.partial_eval.DynamicJaxprTracer(
                jaxpr_trace, outvar.aval, new_var
            )
            return out_tracer, new_var
        new_shape = [s if isinstance(s, int) else env[s] for s in outvar.aval.shape]
        if all(isinstance(s, int) for s in outvar.aval.shape):
            new_aval = jax.core.ShapedArray(tuple(new_shape), outvar.aval.dtype)
        else:  # pragma: no cover
            new_aval = jax.core.DShapedArray(tuple(new_shape), outvar.aval.dtype)
        # JAX 0.7.0: Create variable first, then pass to DynamicJaxprTracer
        new_var = jaxpr_trace.frame.newvar(new_aval)
        out_tracer = jax.interpreters.partial_eval.DynamicJaxprTracer(
            jaxpr_trace, new_aval, new_var
        )

        if not isinstance(outvar, jax.extend.core.Literal):
            env[outvar] = new_var
        return out_tracer, new_var

    def custom_staging_rule(jaxpr_trace, source_info, *tracers, **params):
        """
        Add new jaxpr equation to the jaxpr_trace and return new tracers.
        """
        if not jax.config.jax_dynamic_shapes:
            # fallback to normal behavior
            return jaxpr_trace.default_process_primitive(
                primitive, tracers, params, source_info=source_info
            )
        outvars = get_outvars_from_params(params)

        env: dict[jax.extend.core.Var, jax.extend.core.Var] = {}  # branch var to new equation var
        # JAX 0.7.0: Use t.val to get var from tracer, and TracingEqn for frame.add_eqn
        invars = [t.val for t in tracers]
        if create_initial_env:
            env = create_initial_env(params, invars)
        if outvars:
            out_tracers, returned_vars = tuple(
                zip(*(_tracer_and_outvar(jaxpr_trace, var, env) for var in outvars), strict=True)
            )
        else:
            out_tracers, returned_vars = (), ()

        eqn = jax.core.new_jaxpr_eqn(
            invars,
            returned_vars,
            primitive,
            params,
            jax.core.no_effects,
            source_info,
        )
        tracing_eqn = TracingEqn(
            list(tracers),
            returned_vars,
            primitive,
            params,
            eqn.effects,
            source_info,
            eqn.ctx,
        )
        jaxpr_trace.frame.add_eqn(tracing_eqn)
        return out_tracers

    jax.interpreters.partial_eval.custom_staging_rules[primitive] = custom_staging_rule
