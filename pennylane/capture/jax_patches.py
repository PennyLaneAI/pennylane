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
r"""
Runtime patches for JAX 0.7.x dynamic-shape compatibility.

For a detailed explanation of these patches, see:
    pennylane/capture/JAX_PATCHES_EXPLAINED.md

Problem
-------
JAX 0.7.x has a bug where `_dyn_shape_staging_rule` and `pjit_staging_rule` create
`JaxprEqn` objects, but `trace.frame.add_eqn` asserts for `TracingEqn`. This breaks
ALL array creation with traced dimensions::

    jnp.arange(n)       # n is traced → AssertionError
    jnp.ones((n,))      # n is traced → AssertionError
    jnp.zeros(n)        # n is traced → AssertionError

Solution
--------
We inject a `make_eqn` helper into `DynamicJaxprTrace` that properly creates
`TracingEqn` objects, then patch the buggy staging rules to use this helper.

Patches Applied
---------------
1. ``DynamicJaxprTrace.make_eqn`` — Helper to create TracingEqn with proper context
2. ``lax._dyn_shape_staging_rule`` — Fixed to use make_eqn helper
3. ``pjit.pjit_staging_rule`` — Fixed to use make_eqn helper
4. ``pe.custom_staging_rules[jit_p]`` — Registry entry for patched pjit rule

Usage
-----
Apply patches via the Patcher context manager::

    from pennylane.capture.patching import Patcher
    from pennylane.capture.jax_patches import get_jax_patches

    with Patcher(*get_jax_patches()):
        jaxpr = jax.make_jaxpr(fn, abstracted_axes={0: 'n'})(x)

Inspiration
-----------
This approach is modeled after Catalyst's JAX patches (see catalyst.jax_extras.patches).

Note
----
JAX 0.7.x only has ``DynamicJaxprTrace`` — the ``StagingJaxprTrace`` from older
JAX versions no longer exists. All patches assume DynamicJaxprTrace.
"""

# pylint: disable=too-many-arguments
# pylint: disable=unused-import,no-else-return,unidiomatic-typecheck,use-dict-literal
# pylint: disable=protected-access,possibly-used-before-assignment

has_jax = True
try:
    import jax

    # only do the following if jax is 0.7.x
    jax_version = jax.__version__
    from packaging import version

    if version.parse(jax_version) >= version.parse("0.7.0") and version.parse(
        jax_version
    ) < version.parse("0.8.0"):
        from jax._src import config as jax_config
        from jax._src import core, pjit, source_info_util
        from jax._src.core import JaxprEqnContext, Var
        from jax._src.interpreters import partial_eval as pe
        from jax._src.interpreters.partial_eval import (
            DynamicJaxprTracer,
            TracingEqn,
            compute_on,
            xla_metadata_lib,
        )
        from jax._src.lax import lax
except ModuleNotFoundError:  # pragma: no cover
    has_jax = False  # pragma: no cover


def _add_make_eqn_helper():
    """
    Return a make_eqn helper method to DynamicJaxprTrace.

    This helper properly creates TracingEqn objects, which is needed for JAX 0.7.0
    compatibility. This is based on Catalyst's approach to the same issue.

    Returns:
        tuple: (DynamicJaxprTrace, "make_eqn", make_eqn).
    """

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

    return (pe.DynamicJaxprTrace, "make_eqn", make_eqn)


def _patch_dyn_shape_staging_rule():
    """
    Return _dyn_shape_staging_rule patch to fix dynamic shape handling.

    The bug in JAX 0.7.0's lax/lax.py lines 267-275 is that it uses:
    - pe.new_jaxpr_eqn instead of proper TracingEqn creation

    This causes an AssertionError when add_eqn expects a TracingEqn but gets a JaxprEqn.
    This affects all array creation operations with traced dimensions like jnp.arange,
    jnp.ones, jnp.zeros, etc.

    The fix uses the make_eqn helper to properly create TracingEqn objects.

    Returns:
        list: List of patch tuples.
    """

    def patched_dyn_shape_staging_rule(trace, source_info, prim, out_aval, *args, **params):
        """Patched version of _dyn_shape_staging_rule using make_eqn helper.

        Note: JAX 0.7.x only uses DynamicJaxprTrace (no StagingJaxprTrace exists).
        The make_eqn helper creates TracingEqn which add_eqn expects.
        """
        # Use make_eqn helper to create TracingEqn properly
        eqn, out_tracers = trace.make_eqn(
            args, out_aval, prim, params, core.no_effects, source_info
        )
        trace.frame.add_eqn(eqn)
        # Return single tracer (not list) since out_aval is a single value
        return out_tracers[0]

    # Return just the core patch - the wrappers will call the patched version
    return [
        (lax, "_dyn_shape_staging_rule", patched_dyn_shape_staging_rule),
    ]


def _patch_pjit_staging_rule():
    """
    Return pjit_staging_rule patch to fix dynamic shape handling.

    The bug in JAX 0.7.0's pjit.py lines 1894-1898 is that it uses:
    - core.new_jaxpr_eqn instead of pe.new_eqn_recipe
    - arg.var instead of accessing the correct tracer value

    This causes an AssertionError when add_eqn expects a TracingEqn but gets a JaxprEqn.

    Returns:
        list: List of patch tuples.
    """
    # Store the original function
    original_staging_rule = pjit.pjit_staging_rule

    def patched_pjit_staging_rule(trace, source_info, *args, **params):
        """Patched version of pjit_staging_rule with dynamic shape fixes."""
        # Use the original implementation for most cases
        if not jax_config.dynamic_shapes.value:
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

        # Use make_eqn helper to create TracingEqn properly
        # Note: JAX 0.7.x only uses DynamicJaxprTrace (no StagingJaxprTrace exists)
        in_tracers = [core.get_referent(arg) for arg in args]
        out_avals = [v.aval for v in outvars]
        eqn, out_tracers = trace.make_eqn(
            in_tracers, out_avals, pjit.jit_p, params, jaxpr.effects, source_info
        )
        trace.frame.add_eqn(eqn)

        # Handle forwarding
        out_tracers_ = iter(out_tracers)
        out_tracers = [args[f] if isinstance(f, int) else next(out_tracers_) for f in in_fwd]
        assert next(out_tracers_, None) is None

        return out_tracers

    return [
        (pjit, "pjit_staging_rule", patched_pjit_staging_rule),
        (pe.custom_staging_rules, "__dict_item__", pjit.jit_p, patched_pjit_staging_rule),
    ]


def get_jax_patches():
    """Get patch tuples for use with Patcher context manager.

    Returns a tuple of (obj, attr, new_value) tuples that can be passed to Patcher.
    These patches fix JAX 0.7.0+ compatibility issues for dynamic shapes and pjit.

    Returns:
        tuple: Patch tuples for Patcher, or empty tuple if patches not needed

    Example:
        >>> from pennylane.capture.patching import Patcher
        >>> from pennylane.capture.jax_patches import get_jax_patches
        >>> with Patcher(*get_jax_patches()):
        ...     # JAX operations with patches applied
        ...     jaxpr = jax.make_jaxpr(my_function)(args)
    """
    if not has_jax:
        return ()

    patches = []

    # Get all patches from the helper functions
    patches.append(_add_make_eqn_helper())
    patches.extend(_patch_dyn_shape_staging_rule())
    patches.extend(_patch_pjit_staging_rule())

    return tuple(patches)
