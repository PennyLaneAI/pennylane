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

This module patches JAX internal functions to fix bugs that affect PennyLane's
capture mechanism. These patches are applied at module import time and are
version-specific.

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
These patches fix 27+ dynamic shape tests that were previously failing due to these JAX bugs:
- Array creation operations (jnp.arange, jnp.ones, jnp.zeros with traced dimensions)
- Cond operations with dynamic shapes
- For loop operations with dynamic shapes
- While loop operations with dynamic shapes
- Custom staging rules with dynamic shapes

Without these patches, any operation creating arrays with traced dimensions would fail
with AssertionError in trace.frame.add_eqn.
"""

# pylint: disable=import-outside-toplevel,too-many-arguments,redefined-outer-name
# pylint: disable=unused-import,no-else-return,unidiomatic-typecheck,use-dict-literal

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


# Apply patches based on JAX version
if has_jax:

    # Check JAX version
    from packaging.version import Version

    jax_version = Version(jax.__version__)
    if jax_version >= Version("0.7.0"):
        try:
            _add_make_eqn_helper()
            _patch_dyn_shape_staging_rule()
            _patch_pjit_staging_rule()
        except Exception as e:  # pylint: disable=broad-except
            import warnings

            warnings.warn(
                f"Failed to apply JAX patches for version {jax.__version__}: {e}. "
                "Some dynamic shape features may not work correctly.",
                UserWarning,
            )
