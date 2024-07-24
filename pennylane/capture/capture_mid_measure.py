# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This submodule defines the abstract classes and primitives for capturing mid-circuit measurements.
"""

from functools import lru_cache
from typing import Optional

import pennylane as qml

has_jax = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    has_jax = False


@lru_cache
def _get_abstract_mid_measure_legacy():
    if not has_jax:  # pragma: no cover
        raise ImportError("Jax is required for plxpr.")  # pragma: no cover

    class AbstractMidMeasure(jax.core.ShapedArray):
        """An abstract mid-circuit measurement value."""

        def __eq__(self, other):
            return isinstance(other, AbstractMidMeasure)

        def __hash__(self):
            return hash("AbstractMidMeasure")

    arithmetic_fns = [
        "eq",
        "ne",
        "invert",
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "rtruediv",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
    ]

    dtype_priority = {
        jnp.dtype("complex128" if jax.config.jax_enable_x64 else "complex64"): 3,
        jnp.dtype("float64" if jax.config.jax_enable_x64 else "float32"): 2,
        jnp.dtype("int64" if jax.config.jax_enable_x64 else "int32"): 1,
        jnp.dtype("bool"): 0,
    }

    def _create_arithmetic_prim(f_str):
        prim = jax.core.Primitive(f_str)

        dunder_str = f"__{f_str}__"

        @prim.def_impl
        def _(*args):
            return getattr(type(args[0]), dunder_str)(*args)

        @prim.def_abstract_eval
        def _(*args):
            dtype = max(args, key=lambda x: dtype_priority[x.dtype]).dtype
            return AbstractMidMeasure(args[0].shape, dtype)

        @staticmethod
        def fn(*args):
            return prim.bind(*args)

        return fn

    for f_str in arithmetic_fns:
        math_fn = _create_arithmetic_prim(f_str)
        setattr(AbstractMidMeasure, f"_{f_str}", math_fn)

    jax.core.raise_to_shaped_mappings[AbstractMidMeasure] = lambda aval, _: aval

    return AbstractMidMeasure


@lru_cache
def create_mid_measure_primitive_legacy() -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an mid-circuit measurement type.

    Called when using :func:`~pennylane.measure`.

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive corresponding to a mid-circuit
        measurement. ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive("mid_measure")

    @primitive.def_impl
    def _(wires, reset=False, postselect=None):
        # pylint: disable=protected-access
        return qml.measurements.mid_measure._measure_impl(wires, reset=reset, postselect=postselect)

    abstract_type = _get_abstract_mid_measure_legacy()
    dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32

    @primitive.def_abstract_eval
    def _(*_, **__):
        return abstract_type((), dtype)

    return primitive


# ##############################################################################################################################################################################
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# #####OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#OTHER_IMPL#####
# ##############################################################################################################################################################################


def _get_mid_measure_dtype():
    if not has_jax:  # pragma: no cover
        raise ImportError("Jax is required for plxpr.")  # pragma: no cover

    class MidMeasureDtype(jax.numpy.bool_):
        """An abstract mid-circuit measurement value."""

        def __eq__(self, other):
            return isinstance(other, MidMeasureDtype)

        def __hash__(self):
            return hash("MidMeasureDtype")

    return MidMeasureDtype


toggle = True


@lru_cache
def create_mid_measure_primitive() -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an mid-circuit measurement type.

    Called when using :func:`~pennylane.measure`.

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive corresponding to a mid-circuit
        measurement. ``None`` is returned if jax is not available.

    """

    if not toggle:
        return create_mid_measure_primitive_legacy()

    if not has_jax:
        return None

    primitive = jax.core.Primitive("mid_measure")

    @primitive.def_impl
    def _(wires, reset=False, postselect=None):
        # pylint: disable=protected-access

        return qml.measurements.mid_measure._measure_impl(wires, reset=reset, postselect=postselect)

    abstract_dtype = _get_mid_measure_dtype()

    @primitive.def_abstract_eval
    def _(*_, **__):
        aval = jax.core.ShapedArray((), jnp.bool_)
        return aval

    return primitive


def _create_abstract_mcm():

    class AbstractMCM:
        def __eq__(self, other):
            return isinstance(other, AbstractMCM)

        def __hash__(self):
            return hash("AbstractMCM")

    arithmetic_fns = [
        "eq",
        "ne",
        "invert",
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "rtruediv",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
    ]

    dtype_priority = {
        jnp.dtype("complex128" if jax.config.jax_enable_x64 else "complex64"): 3,
        jnp.dtype("float64" if jax.config.jax_enable_x64 else "float32"): 2,
        jnp.dtype("int64" if jax.config.jax_enable_x64 else "int32"): 1,
        jnp.dtype("bool"): 0,
    }

    def _create_arithmetic_prim(f_str):
        prim = jax.core.Primitive(f_str)

        dunder_str = f"__{f_str}__"

        @prim.def_impl
        def _(*args):
            return getattr(type(args[0]), dunder_str)(*args)

        @prim.def_abstract_eval
        def _(*args):
            dtype = max(args, key=lambda x: dtype_priority[x.dtype]).dtype
            return AbstractMCM(args[0].shape, dtype)

        @staticmethod
        def fn(*args):
            return prim.bind(*args)

        return fn

    for f_str in arithmetic_fns:
        math_fn = _create_arithmetic_prim(f_str)
        setattr(AbstractMCM, f"_{f_str}", math_fn)

    jax.core.raise_to_shaped_mappings[AbstractMCM] = lambda aval, _: aval

    return AbstractMCM


def create_mcm_value_primitive():
    if not has_jax:
        return None

    primitive = jax.core.Primitive("MCM")

    @primitive.def_impl
    def _(measurements, processing_fn=lambda v: v):
        return type.__call__(qml.measurements.MeasurementValue, measurements, processing_fn)

    abstract_type = _create_abstract_mcm()

    @primitive.def_abstract_eval
    def _(*_, **__):
        return abstract_type()

    return primitive
