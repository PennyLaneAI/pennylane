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

Similar to Catalyst's approach, we monkey-patch specific JAX internal functions
that have bugs affecting dynamic shape support.

JAX 0.7.0+ Patches
------------------

1. **pjit_staging_rule**: Fixed dynamic shape handling in pjit.py lines 1894-1898.

   The bugs in the original JAX implementation:
   - Uses `core.new_jaxpr_eqn` which creates a JaxprEqn, but `trace.frame.add_eqn`
     expects a TracingEqn. This causes an AssertionError.
   - Accesses `arg.var` which doesn't exist for DynamicJaxprTracer objects.

   The fix:
   - Use `pe.new_eqn_recipe` which properly creates equation recipes for dynamic tracing.
   - Wrap outvars in DynamicJaxprTracer instances before creating the equation.
   - This ensures the equation is properly added to the trace frame.

These patches enable dynamic shape support in PennyLane's capture mechanism,
allowing tests like test_dynamic_shots and several TestDynamicShapes tests to pass.

Note: Some dynamic shape operations (like creating arrays with traced dimensions)
still don't work with JAX 0.7.0 due to fundamental limitations in JAX's implementation.
"""

import jax

# Check JAX version
from packaging.version import Version

jax_version = Version(jax.__version__)


def _patch_pjit_staging_rule():
    """
    Patch pjit_staging_rule to fix dynamic shape handling.

    The bug in JAX 0.7.0's pjit.py lines 1894-1898 is that it uses:
    - core.new_jaxpr_eqn instead of pe.new_eqn_recipe
    - arg.var instead of accessing the correct tracer value

    This causes an AssertionError when add_eqn expects a TracingEqn but gets a JaxprEqn.
    """
    # pylint: disable=import-outside-toplevel, protected-access
    from jax import tree_util
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

        # Use original for DynamicJaxprTrace (eval_jaxpr path)
        # DynamicJaxprTrace doesn't have 'counter' attribute
        if not hasattr(trace, "counter"):
            return original_staging_rule(trace, source_info, *args, **params)

        jaxpr = params["jaxpr"]

        # This is the dynamic shapes path that needs fixing
        jaxpr, in_fwd, out_shardings, out_layouts = pjit._pjit_forwarding(
            jaxpr, params["out_shardings"], params["out_layouts"]
        )
        params = dict(params, jaxpr=jaxpr, out_shardings=out_shardings, out_layouts=out_layouts)

        # Fix 1: Use list instead of map to create outvars
        outvars = [trace.frame.newvar(aval) for aval in pjit._out_type(jaxpr)]

        # Fix 2: Use pe.new_eqn_recipe and DynamicJaxprTracer
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
        out_tracers = [args[f] if type(f) is int else next(out_tracers_) for f in in_fwd]
        assert next(out_tracers_, None) is None

        return out_tracers

    # Apply the patch
    pjit.pjit_staging_rule = patched_pjit_staging_rule
    # Also update the custom staging rules dict
    pe.custom_staging_rules[pjit.jit_p] = patched_pjit_staging_rule


# Apply patches based on JAX version
if jax_version >= Version("0.7.0"):
    try:
        _patch_pjit_staging_rule()
    except Exception as e:  # pylint: disable=broad-except
        import warnings

        warnings.warn(
            f"Failed to apply JAX patches for version {jax.__version__}: {e}. "
            "Some dynamic shape features may not work correctly.",
            UserWarning,
        )
