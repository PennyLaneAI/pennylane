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
Runtime patches for JAX internals to fix compatibility issues.

This module provides patches for JAX internal functions to fix bugs that affect PennyLane's
capture mechanism. These patches can be applied selectively using a context manager instead
of being applied globally at import time.

This approach is inspired by Catalyst's JAX patches (see catalyst.jax_extras.patches),
which similarly monkey-patch JAX internal functions for compatibility.

JAX 0.7.2+ Patches
------------------

1. **iota custom_partial_eval_rule**: Fixed dynamic shape handling for array creation.

   The issue in JAX 0.7.2's iota handling:
   - The built-in partial evaluation doesn't properly handle dynamic shapes
   - This affects ALL array creation operations with traced dimensions:
     * jax.numpy.arange(traced_value)
     * jax.numpy.ones((traced_value,))
     * jax.numpy.zeros(traced_value)
     * Any operation using lax.iota with dynamic shapes

   The fix:
   - Register a custom_partial_eval_rule for iota_p
   - Use new_eqn_recipe to properly create equation recipes for dynamic tracing
   - Set tracer.recipe instead of manually adding equations to the frame
   - This enables array creation with traced dimensions to work correctly

Impact
------
These patches fix many dynamic shape tests that were previously failing:
- Array creation operations (jnp.arange, jnp.ones, jnp.zeros with traced dimensions)
- For loop operations with dynamic shapes
- While loop operations with dynamic shapes
- Custom staging rules with dynamic shapes

Usage
-----
Patches are applied automatically on import for JAX 0.7.0+. You can also use the context manager:

    >>> from pennylane.capture.jax_patches import apply_jax_patches
    >>> with apply_jax_patches():
    ...     # JAX patches are active here
    ...     result = some_function_needing_patches()
    >>> # Patches are reverted here
"""

# pylint: disable=import-outside-toplevel,too-many-arguments,redefined-outer-name
# pylint: disable=unused-import,no-else-return,unidiomatic-typecheck,use-dict-literal
# pylint: disable=protected-access

has_jax = True
try:
    import jax
except ImportError:  # pragma: no cover
    has_jax = False  # pragma: no cover


def _add_make_eqn_helper():
    """
    Add a make_eqn helper method to DynamicJaxprTrace.

    This helper properly creates TracingEqn objects, which is needed for JAX 0.7.0
    compatibility. This is based on Catalyst's approach to the same issue.
    """
    from jax._src import config as jax_config
    from jax._src import source_info_util
    from jax._src.core import JaxprEqnContext, Var
    from jax._src.interpreters import partial_eval as pe
    from jax._src.interpreters.partial_eval import (
        DynamicJaxprTracer,
        TracingEqn,
        compute_on,
        xla_metadata_lib,
    )

    def make_eqn(
        self,
        in_tracers: list,
        out_avals_or_tracers: list,
        primitive,
        params: dict,
        effects: set,
        source_info=None,
        ctx=None,
    ):
        """Create a tracing equation properly.

        Args:
            in_tracers: Input tracers
            out_avals_or_tracers: Output abstract values OR output tracers (with vars already created)
            primitive: The primitive operation
            params: Parameters for the primitive
            effects: Effects of the operation
            source_info: Source information for debugging
            ctx: JaxprEqnContext (created if not provided)

        Returns:
            (eqn, out_tracers): TracingEqn and output tracers
        """
        source_info = source_info or source_info_util.new_source_info()
        ctx = ctx or JaxprEqnContext(
            compute_on.current_compute_type(),
            jax_config.threefry_partitionable.value,
            xla_metadata_lib.current_xla_metadata(),
        )

        # Normalize out_avals to a list
        if not isinstance(out_avals_or_tracers, (list, tuple)):
            out_avals = [out_avals_or_tracers]
        else:
            out_avals = out_avals_or_tracers

        outvars = [self.frame.newvar(aval) for aval in out_avals]

        if jax_config.enable_checks.value:
            assert all(isinstance(x, DynamicJaxprTracer) for x in in_tracers)
            assert all(isinstance(v, Var) for v in outvars)

        eqn = TracingEqn(list(in_tracers), outvars, primitive, params, effects, source_info, ctx)

        # Create output tracers - manually create DynamicJaxprTracer objects
        # We pass the equation as the parent parameter (4th argument to __init__)
        out_tracers = [
            DynamicJaxprTracer(self, aval, v, source_info, eqn)
            for aval, v in zip(out_avals, outvars)
        ]

        return eqn, out_tracers

    # Add the helper method to DynamicJaxprTrace
    pe.DynamicJaxprTrace.make_eqn = make_eqn


def _patch_dyn_shape_staging_rule():
    """
    Patch _dyn_shape_staging_rule to fix dynamic shape handling for JAX 0.7.2+.

    JAX 0.7.2 still uses custom_staging_rules for built-in primitives like iota.
    The issue is that _dyn_shape_staging_rule creates JaxprEqn objects which
    don't work with trace.frame.add_eqn() (expects TracingEqn).

    This patch directly monkey-patches the _dyn_shape_staging_rule and _iota_staging_rule
    functions in JAX to use the correct API.
    """
    from jax._src import core, source_info_util
    from jax._src.interpreters import partial_eval as pe
    from jax._src.lax import lax

    # Store original functions
    _original_functions["_dyn_shape_staging_rule"] = lax._dyn_shape_staging_rule
    _original_functions["_iota_staging_rule"] = lax._iota_staging_rule

    def patched_dyn_shape_staging_rule(trace, source_info, prim, out_aval, *args, **params):
        """Patched version using recipe-based approach for JaxprTrace."""
        if hasattr(trace, "make_eqn"):
            # DynamicJaxprTrace path - use make_eqn helper (created by _add_make_eqn_helper)
            # This is the old deprecated path that some code might still use
            in_tracers = [core.get_referent(arg) for arg in args]
            eqn, out_tracers = trace.make_eqn(
                in_tracers, out_aval, prim, params, core.no_effects, source_info
            )
            trace.frame.add_eqn(eqn)
            return out_tracers[0]
        else:
            # JaxprTrace path - DON'T use DynamicJaxprTracer, use JaxprTracer
            # And DON'T call trace.frame.add_eqn() - just set recipe
            out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(out_aval), None)
            eqn = pe.new_eqn_recipe(
                trace, args, [out_tracer], prim, params, core.no_effects, source_info
            )
            out_tracer.recipe = eqn
            return out_tracer

    def patched_iota_staging_rule(
        trace, source_info, *dyn_shape, dtype, shape, dimension, sharding
    ):
        """Patched version of _iota_staging_rule that inlines _dyn_shape_staging_rule logic."""
        params = {"dtype": dtype, "shape": shape, "dimension": dimension, "sharding": sharding}
        if not dyn_shape:
            return trace.default_process_primitive(lax.iota_p, (), params, source_info=source_info)

        # Inline the _dyn_shape_staging_rule logic here to avoid closure issues
        aval = core.DShapedArray(lax._merge_dyn_shape(shape, dyn_shape), dtype, False)

        # Choose path based on trace type
        if hasattr(trace, "make_eqn"):
            # DynamicJaxprTrace path - use make_eqn helper (created by _add_make_eqn_helper)
            in_tracers = [core.get_referent(arg) for arg in dyn_shape]
            eqn, out_tracers = trace.make_eqn(
                in_tracers, aval, lax.iota_p, params, core.no_effects, source_info
            )
            trace.frame.add_eqn(eqn)
            return out_tracers[0]
        else:
            # JaxprTrace path - use JaxprTracer with recipe (DON'T call add_eqn)
            out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
            eqn = pe.new_eqn_recipe(
                trace, dyn_shape, [out_tracer], lax.iota_p, params, core.no_effects, source_info
            )
            out_tracer.recipe = eqn
            return out_tracer

    # Apply the patches
    lax._dyn_shape_staging_rule = patched_dyn_shape_staging_rule
    lax._iota_staging_rule = patched_iota_staging_rule
    # Update custom_staging_rules
    pe.custom_staging_rules[lax.iota_p] = patched_iota_staging_rule


def _patch_pjit_staging_rule():
    """
    Patch pjit_staging_rule to fix dynamic shape handling for JAX 0.7.2+.

    The issue is that pjit_staging_rule calls trace.frame.add_eqn() with a JaxprEqn,
    but when the trace is a JaxprTrace (not DynamicJaxprTrace), this causes an
    AssertionError because JaxprTrace expects TracingEqn objects.

    The fix is to use the recipe-based approach (new_eqn_recipe) for JaxprTrace.
    """
    import jax._src.pjit as pjit_module
    from jax._src import config, core, source_info_util
    from jax._src.interpreters import partial_eval as pe
    from jax._src.interpreters import pxla

    # Store original function
    _original_functions["pjit_staging_rule"] = pjit_module.pjit_staging_rule

    def patched_pjit_staging_rule(trace, source_info, *args, **params):
        """Patched version that uses recipe-based approach for JaxprTrace."""
        # Handle the non-dynamic-shapes path and inline path - use original
        if params["compiler_options_kvs"]:
            raise ValueError(
                "`compiler_options` can only be passed to top-level `jax.jit`. Got"
                f' compiler_options={dict(params["compiler_options_kvs"])} specified on'
                f' a nested jit with name: {params["name"]} and source info:'
                f" {source_info_util.summarize(source_info)}"
            )

        # Handle inline path
        if (
            params["inline"]
            and all(isinstance(i, pjit_module.UnspecifiedValue) for i in params["in_shardings"])
            and all(isinstance(o, pjit_module.UnspecifiedValue) for o in params["out_shardings"])
            and all(i is None for i in params["in_layouts"])
            and all(o is None for o in params["out_layouts"])
        ):
            jaxpr = params["jaxpr"]
            if config.dynamic_shapes.value:
                with core.set_current_trace(trace):
                    out = core.eval_jaxpr(
                        jaxpr.jaxpr, jaxpr.consts, *args, propagate_source_info=False
                    )
            else:
                out = pe.inline_jaxpr_into_trace(
                    trace, source_info, jaxpr.jaxpr, jaxpr.consts, *args
                )
            return [trace.to_jaxpr_tracer(x, source_info) for x in out]

        # Handle dynamic shapes path - this is where the bug is
        jaxpr = params["jaxpr"]
        if config.dynamic_shapes.value:
            jaxpr, in_fwd, out_shardings, out_layouts = pjit_module._pjit_forwarding(
                jaxpr, params["out_shardings"], params["out_layouts"]
            )
            params = dict(params, jaxpr=jaxpr, out_shardings=out_shardings, out_layouts=out_layouts)

            # Check if we're in JaxprTrace (recipe-based) or DynamicJaxprTrace (frame-based)
            if hasattr(trace, "make_eqn"):
                # DynamicJaxprTrace path - use make_eqn to create proper TracingEqn
                out_avals = pjit_module._out_type(jaxpr)

                # Convert args to tracers if needed
                in_tracers = [core.get_referent(arg) for arg in args]

                # Use make_eqn to create the equation and tracers properly
                eqn, out_tracers = trace.make_eqn(
                    in_tracers, out_avals, pjit_module.jit_p, params, jaxpr.effects, source_info
                )
                trace.frame.add_eqn(eqn)
            else:
                # JaxprTrace path - use recipe-based approach
                out_avals = pjit_module._out_type(jaxpr)
                out_tracers = [
                    pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None) for aval in out_avals
                ]
                eqn = pe.new_eqn_recipe(
                    trace, args, out_tracers, pjit_module.jit_p, params, jaxpr.effects, source_info
                )
                for out_tracer in out_tracers:
                    out_tracer.recipe = eqn

            # Handle forwarding
            out_tracers_ = iter(out_tracers)
            out_tracers = [args[f] if type(f) is int else next(out_tracers_) for f in in_fwd]
            assert next(out_tracers_, None) is None
            return out_tracers

        # Handle mutable consts path
        elif any(isinstance(c, core.MutableArray) for c in jaxpr.consts):
            jaxpr, consts = pxla._move_mutable_consts(jaxpr)
            consts = [trace.new_const(c, source_info) for c in consts]
            in_shardings = (*params["in_shardings"],) + (pjit_module.UNSPECIFIED,) * len(consts)
            in_layouts = (*params["in_layouts"],) + (None,) * len(consts)
            donated_invars = (*params["donated_invars"],) + (False,) * len(consts)
            new_params = dict(
                params,
                jaxpr=jaxpr,
                in_shardings=in_shardings,
                in_layouts=in_layouts,
                donated_invars=donated_invars,
            )
            out_tracers = trace.default_process_primitive(
                pjit_module.jit_p, (*args, *consts), new_params, source_info=source_info
            )
        else:
            # Default path
            out_tracers = trace.default_process_primitive(
                pjit_module.jit_p, args, params, source_info=source_info
            )

        return out_tracers

    # Apply the patch
    pjit_module.pjit_staging_rule = patched_pjit_staging_rule
    pe.custom_staging_rules[pjit_module.jit_p] = patched_pjit_staging_rule


def _patch_bind_with_trace():
    """
    Patch Primitive.bind_with_trace to handle typeof failures gracefully.

    In JAX 0.7.2, bind_with_trace tries to apply typeof to all args, but fails
    with Python lists and other non-JAX types. The original code has a try/except
    but then raises an unhelpful error. This patch catches the exception and falls
    back to trace.process_primitive directly.
    """
    from jax._src import core
    from jax._src.core import typeof

    # Store original function
    _original_functions["bind_with_trace"] = core.Primitive.bind_with_trace

    def patched_bind_with_trace(self, trace, args, params):
        """Patched version that handles typeof failures gracefully."""
        try:
            in_type = list(map(typeof, args))
        except Exception:  # pylint: disable=broad-exception-caught
            # If typeof fails (e.g., with Python lists), just process the primitive
            return trace.process_primitive(self, args, params)
        else:
            # Original logic when typeof succeeds
            if self.is_high(*in_type, **params) and trace.requires_low:
                with core.set_current_trace(trace):
                    return self.to_lojax(*args, **params)
            return trace.process_primitive(self, args, params)

    # Apply the patch
    core.Primitive.bind_with_trace = patched_bind_with_trace


# Store original functions for restoration
_original_functions = {}
_patches_applied = False


def _apply_patches():
    """Apply JAX patches. Should only be called when patches are needed."""
    global _patches_applied  # pylint: disable=global-statement

    if _patches_applied:
        return  # Patches already applied

    if not has_jax:
        return

    from packaging.version import Version

    jax_version = Version(jax.__version__)
    # Handle both release versions (0.7.0) and dev versions (0.7.0.devXXX)
    # Dev versions should be treated as >= the base version they're based on
    should_apply = (jax_version.major == 0 and jax_version.minor >= 7) or (
        jax_version >= Version("0.7.0")
    )

    if should_apply:
        try:
            from jax._src.interpreters import partial_eval as pe
            from jax._src.lax import lax

            # Apply patches (which will store originals in _original_functions)
            _add_make_eqn_helper()  # REQUIRED: Adds make_eqn to DynamicJaxprTrace
            _patch_dyn_shape_staging_rule()  # REQUIRED: Patches iota for dynamic shapes
            _patch_pjit_staging_rule()  # REQUIRED: Handles dynamic shapes in jit
            _patch_bind_with_trace()  # REQUIRED: Handles typeof failures for PennyLane types

            _patches_applied = True
        except Exception as e:  # pylint: disable=broad-except
            import warnings

            warnings.warn(
                f"Failed to apply JAX patches for version {jax.__version__}: {e}. "
                "Some dynamic shape features may not work correctly.",
                UserWarning,
            )


def _revert_patches():
    """Revert JAX patches to their original implementations."""
    global _patches_applied  # pylint: disable=global-statement

    if not _patches_applied or not _original_functions:
        return  # No patches to revert

    try:
        from jax._src.interpreters import partial_eval as pe
        from jax._src.lax import lax

        # Restore patched functions
        if "_dyn_shape_staging_rule" in _original_functions:
            lax._dyn_shape_staging_rule = _original_functions["_dyn_shape_staging_rule"]
        if "_iota_staging_rule" in _original_functions:
            lax._iota_staging_rule = _original_functions["_iota_staging_rule"]
            # Also restore in custom_staging_rules
            pe.custom_staging_rules[lax.iota_p] = _original_functions["_iota_staging_rule"]

        _patches_applied = False
    except Exception as e:  # pylint: disable=broad-except
        import warnings

        warnings.warn(
            f"Failed to revert JAX patches: {e}. " "Some JAX functionality may be affected.",
            UserWarning,
        )


class apply_jax_patches:
    """Context manager to temporarily apply JAX patches.

    This context manager enables JAX compatibility patches only within its scope,
    reverting them when exiting. This provides local control over when patches
    are active, avoiding global side effects.

    Example:
        >>> from pennylane.capture.jax_patches import apply_jax_patches
        >>> with apply_jax_patches():
        ...     # Code that needs JAX patches (e.g., dynamic shapes)
        ...     result = some_function_with_dynamic_shapes()
        >>> # Patches are reverted here

    Note:
        This context manager is reentrant-safe. Nested calls will not
        revert patches until the outermost context exits.
    """

    _nesting_level = 0

    def __enter__(self):
        """Enter the context and apply JAX patches."""
        if apply_jax_patches._nesting_level == 0:
            _apply_patches()
        apply_jax_patches._nesting_level += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and revert JAX patches."""
        apply_jax_patches._nesting_level -= 1
        if apply_jax_patches._nesting_level == 0:
            _revert_patches()
        return False


# Apply patches automatically on module import for JAX 0.7.0+
# Since these patches fix bugs in JAX and don't change behavior for correct code,
# it's safe to apply them globally. Users can still use the context manager
# if they need fine-grained control.
if has_jax:
    from packaging.version import Version

    jax_version = Version(jax.__version__)
    # Handle both release versions (0.7.0) and dev versions (0.7.0.devXXX)
    # Dev versions should be treated as >= the base version they're based on
    if jax_version.major == 0 and jax_version.minor >= 7:
        _apply_patches()
    elif jax_version >= Version("0.7.0"):
        _apply_patches()
