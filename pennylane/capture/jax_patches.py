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
which similarly monkey-patch JAX internal functions for compatibility. The key insight
is to add a `make_eqn` helper method to DynamicJaxprTrace that properly creates
TracingEqn objects, which JAX 0.7.0 requires but doesn't provide in all code paths.

JAX 0.7.0+ Patches
------------------

1. **_dyn_shape_staging_rule**: Fixed dynamic shape handling in lax/lax.py lines 267-275.

   The bug in the original JAX implementation:
   - Uses `pe.new_jaxpr_eqn` which creates a JaxprEqn, but `trace.frame.add_eqn`
     expects a TracingEqn. This causes an AssertionError.
   - This bug affects ALL array creation operations with traced dimensions:
     * jax.numpy.arange(traced_value)
     * jax.numpy.ones((traced_value,))
     * jax.numpy.zeros(traced_value)
     * Any operation using lax.broadcasted_iota with dynamic shapes

   The fix:
   - For StagingJaxprTrace (has counter): Use `pe.new_eqn_recipe` which properly
     creates equation recipes for dynamic tracing.
   - For DynamicJaxprTrace (no counter): Create TracingEqn directly with proper
     JaxprEqnContext, avoiding the AssertionError.
   - This enables array creation with traced dimensions to work correctly.

2. **pjit_staging_rule**: Fixed dynamic shape handling in pjit.py lines 1894-1898.

   The bug in the original JAX implementation:
   - Uses `core.new_jaxpr_eqn` which creates a JaxprEqn, but `trace.frame.add_eqn`
     expects a TracingEqn. This causes an AssertionError.
   - Accesses `arg.var` which doesn't exist for DynamicJaxprTracer objects.

   The fix:
   - Use `pe.new_eqn_recipe` which properly creates equation recipes for dynamic tracing.
   - Wrap outvars in DynamicJaxprTracer instances before creating the equation.
   - Special handling for DynamicJaxprTrace (eval_jaxpr path) to avoid counter errors.
   - This enables pjit operations with dynamic shapes to work correctly.

Impact
------
These patches fix many dynamic shape tests that were previously failing due to these JAX bugs:
- Array creation operations (jnp.arange, jnp.ones, jnp.zeros with traced dimensions)
- Cond operations with dynamic shapes
- For loop operations with dynamic shapes
- While loop operations with dynamic shapes
- Custom staging rules with dynamic shapes

Without these patches, any operation creating arrays with traced dimensions would fail
with AssertionError in trace.frame.add_eqn.

Usage
-----
Instead of applying patches globally at import time, use the context manager:

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
    Patch _dyn_shape_staging_rule to fix dynamic shape handling.

    The bug in JAX 0.7.0's lax/lax.py lines 267-275 is that it uses:
    - pe.new_jaxpr_eqn instead of proper TracingEqn creation

    This causes an AssertionError when add_eqn expects a TracingEqn but gets a JaxprEqn.
    This affects all array creation operations with traced dimensions like jnp.arange,
    jnp.ones, jnp.zeros, etc.

    The fix uses the make_eqn helper to properly create TracingEqn objects.
    """
    from jax._src import core
    from jax._src.interpreters import partial_eval as pe
    from jax._src.lax import lax

    def patched_dyn_shape_staging_rule(trace, source_info, prim, out_aval, *args, **params):
        """Patched version of _dyn_shape_staging_rule using make_eqn helper."""
        # Check if we have a StagingJaxprTrace (has counter) or DynamicJaxprTrace (no counter)
        if hasattr(trace, "counter"):
            # StagingJaxprTrace path - use new_eqn_recipe (original JAX approach)
            var = trace.frame.newvar(out_aval)
            out_tracer = pe.DynamicJaxprTracer(trace, out_aval, var, source_info)
            eqn = pe.new_eqn_recipe(
                trace, args, [out_tracer], prim, params, core.no_effects, source_info
            )
            out_tracer.recipe = eqn
            return out_tracer

        # DynamicJaxprTrace path - use make_eqn helper
        eqn, out_tracers = trace.make_eqn(
            args, out_aval, prim, params, core.no_effects, source_info
        )
        trace.frame.add_eqn(eqn)
        # Return single tracer (not list) since out_aval is a single value
        return out_tracers[0]

    # Apply the patch
    lax._dyn_shape_staging_rule = patched_dyn_shape_staging_rule

    # Also need to patch the iota_staging_rule which uses _dyn_shape_staging_rule
    def patched_iota_staging_rule(
        trace, source_info, *dyn_shape, dtype, shape, dimension, sharding
    ):
        """Patched version of _iota_staging_rule."""
        params = {"dtype": dtype, "shape": shape, "dimension": dimension, "sharding": sharding}
        if not dyn_shape:
            return trace.default_process_primitive(lax.iota_p, (), params, source_info=source_info)
        aval = core.DShapedArray(lax._merge_dyn_shape(shape, dyn_shape), dtype, False)
        return patched_dyn_shape_staging_rule(
            trace, source_info, lax.iota_p, aval, *dyn_shape, **params
        )

    lax._iota_staging_rule = patched_iota_staging_rule

    # Update the custom staging rules
    pe.custom_staging_rules[lax.iota_p] = patched_iota_staging_rule


def _patch_pjit_staging_rule():
    """
    Patch pjit_staging_rule to fix dynamic shape handling.

    The bug in JAX 0.7.0's pjit.py lines 1894-1898 is that it uses:
    - core.new_jaxpr_eqn instead of pe.new_eqn_recipe
    - arg.var instead of accessing the correct tracer value

    This causes an AssertionError when add_eqn expects a TracingEqn but gets a JaxprEqn.
    """
    from jax._src import config, core, pjit
    from jax._src.interpreters import partial_eval as pe

    # Store the original function
    original_staging_rule = pjit.pjit_staging_rule

    def patched_pjit_staging_rule(trace, source_info, *args, **params):
        """Patched version of pjit_staging_rule with dynamic shape fixes."""
        # Use the original implementation for most cases
        if not config.dynamic_shapes.value:
            return original_staging_rule(trace, source_info, *args, **params)

        # Check if we're in the inline path
        if (
            params["inline"]
            and all(isinstance(i, pjit.UnspecifiedValue) for i in params["in_shardings"])
            and all(isinstance(o, pjit.UnspecifiedValue) for o in params["out_shardings"])
            and all(i is None for i in params["in_layouts"])
            and all(o is None for o in params["out_layouts"])
        ):
            # Use original for inline path
            return original_staging_rule(trace, source_info, *args, **params)

        jaxpr = params["jaxpr"]

        # This is the dynamic shapes path that needs fixing
        jaxpr, in_fwd, out_shardings, out_layouts = pjit._pjit_forwarding(
            jaxpr, params["out_shardings"], params["out_layouts"]
        )
        params = {
            **params,
            "jaxpr": jaxpr,
            "out_shardings": out_shardings,
            "out_layouts": out_layouts,
        }

        # Fix 1: Use list instead of map to create outvars
        outvars = [trace.frame.newvar(aval) for aval in pjit._out_type(jaxpr)]

        # DynamicJaxprTrace (eval_jaxpr path) vs StagingJaxprTrace need different approaches
        # DynamicJaxprTrace doesn't have 'counter' attribute
        if not hasattr(trace, "counter"):
            # For DynamicJaxprTrace: Use make_eqn helper (from _add_make_eqn_helper)
            # Pass avals (not tracers) to make_eqn
            in_tracers = [core.get_referent(arg) for arg in args]
            out_avals = [v.aval for v in outvars]
            eqn, out_tracers = trace.make_eqn(
                in_tracers, out_avals, pjit.jit_p, params, jaxpr.effects, source_info
            )
            trace.frame.add_eqn(eqn)
        else:
            # For StagingJaxprTrace: Use new_eqn_recipe
            eqn = pe.new_eqn_recipe(
                trace,
                args,
                [pe.DynamicJaxprTracer(trace, v.aval, v, source_info) for v in outvars],
                pjit.jit_p,
                params,
                jaxpr.effects,
                source_info,
            )

            out_tracers = [pe.DynamicJaxprTracer(trace, v.aval, v, source_info) for v in outvars]
            for t in out_tracers:
                t.recipe = eqn

        # Handle forwarding
        out_tracers_ = iter(out_tracers)
        out_tracers = [args[f] if isinstance(f, int) else next(out_tracers_) for f in in_fwd]
        assert next(out_tracers_, None) is None

        return out_tracers

    # Apply the patch
    pjit.pjit_staging_rule = patched_pjit_staging_rule
    # Also update the custom staging rules dict
    pe.custom_staging_rules[pjit.jit_p] = patched_pjit_staging_rule


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
            from jax._src import pjit
            from jax._src.interpreters import partial_eval as pe
            from jax._src.lax import lax

            # Store original functions for later restoration
            _original_functions["_dyn_shape_staging_rule"] = lax._dyn_shape_staging_rule
            _original_functions["_iota_staging_rule"] = lax._iota_staging_rule
            _original_functions["pjit_staging_rule"] = pjit.pjit_staging_rule
            _original_functions["custom_staging_iota"] = pe.custom_staging_rules.get(lax.iota_p)
            _original_functions["custom_staging_jit"] = pe.custom_staging_rules.get(pjit.jit_p)

            # Apply patches
            _add_make_eqn_helper()
            _patch_dyn_shape_staging_rule()
            _patch_pjit_staging_rule()

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
        from jax._src import pjit
        from jax._src.interpreters import partial_eval as pe
        from jax._src.lax import lax

        # Restore original functions
        lax._dyn_shape_staging_rule = _original_functions["_dyn_shape_staging_rule"]
        lax._iota_staging_rule = _original_functions["_iota_staging_rule"]
        pjit.pjit_staging_rule = _original_functions["pjit_staging_rule"]

        # Restore custom staging rules
        if _original_functions["custom_staging_iota"] is not None:
            pe.custom_staging_rules[lax.iota_p] = _original_functions["custom_staging_iota"]
        elif lax.iota_p in pe.custom_staging_rules:
            del pe.custom_staging_rules[lax.iota_p]

        if _original_functions["custom_staging_jit"] is not None:
            pe.custom_staging_rules[pjit.jit_p] = _original_functions["custom_staging_jit"]
        elif pjit.jit_p in pe.custom_staging_rules:
            del pe.custom_staging_rules[pjit.jit_p]

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
