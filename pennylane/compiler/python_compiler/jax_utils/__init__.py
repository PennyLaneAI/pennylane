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

"""Utilities for translating JAX to xDSL"""

from functools import wraps
from typing import Any, Callable, TypeAlias

import jax
import jaxlib


from jax.extend import mlir as jmlir  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Context as jContext  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Module as jModule  # pylint: disable=no-name-in-module
from jaxlib.mlir.dialects import stablehlo as jstablehlo  # pylint: disable=no-name-in-module

from xdsl.dialects import arith as xarith
from xdsl.dialects import builtin as xbuiltin
from xdsl.dialects import func as xfunc
from xdsl.dialects import scf as xscf
from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects import tensor as xtensor
from xdsl.dialects import transform as xtransform

from xdsl.parser import Parser as xParser
from xdsl.context import Context as xContext

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect

JaxJittedFunction: TypeAlias = jaxlib.xla_extension.PjitFunction


def _module_inline(func: JaxJittedFunction, *args, **kwargs) -> jModule:
    """Get the module from the jax.jitted function"""
    return func.lower(*args, **kwargs).compiler_ir()


def module(func: JaxJittedFunction) -> Callable[Any, jModule]:
    """
    Decorator for _module_inline
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> jModule:
        return _module_inline(func, *args, **kwargs)

    return wrapper


def _generic_inline(func: JaxJittedFunction, *args, **kwargs) -> str:  # pragma: no cover
    """
    Create the generic textual representation for the jax.jit'ed function
    """
    lowered = func.lower(*args, **kwargs)
    mod = lowered.compiler_ir()
    return mod.operation.get_asm(binary=False, print_generic_op_form=True, assume_verified=True)


def generic(func: JaxJittedFunction) -> Callable[Any, str]:  # pragma: no cover
    """
    Decorator for _generic_inline.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        return _generic_inline(func, *args, **kwargs)

    return wrapper


def parse_generic_to_xdsl_module(program: str) -> xbuiltin.ModuleOp:  # pragma: no cover
    """Parses generic MLIR program to xDSL module"""
    ctx = xContext(allow_unregistered=True)
    ctx.load_dialect(xarith.Arith)
    ctx.load_dialect(xbuiltin.Builtin)
    ctx.load_dialect(xfunc.Func)
    ctx.load_dialect(xscf.Scf)
    ctx.load_dialect(xstablehlo.StableHLO)
    ctx.load_dialect(xtensor.Tensor)
    ctx.load_dialect(xtransform.Transform)
    ctx.load_dialect(QuantumDialect)
    moduleOp: xbuiltin.ModuleOp = xParser(ctx, program).parse_module()
    return moduleOp


def parse_generic_to_jax_module(program: str) -> jModule:  # pragma: no cover
    """Parses an MLIR program in string representation to a jax Module"""
    with jContext() as ctx:
        ctx.allow_unregistered_dialects = True
        jstablehlo.register_dialect(ctx)  # pylint: disable=no-member
        return jModule.parse(program)


def jax_from_docstring(func: Callable) -> jModule:  # pragma: no cover
    """Parses an MLIR program in string representation located in the docstring."""

    @wraps(func)
    def wrapper(*_, **__):
        return parse_generic_to_jax_module(func.__doc__)

    return wrapper


def _xdsl_module_inline(
    func: JaxJittedFunction, *args, **kwargs
) -> xbuiltin.ModuleOp:  # pragma: no cover
    generic_repr = _generic_inline(func, *args, **kwargs)
    return parse_generic_to_xdsl_module(generic_repr)


def xdsl_from_docstring(func: Callable) -> xbuiltin.ModuleOp:  # pragma: no cover
    """Parses a docstring into an xdsl module"""

    @wraps(func)
    def wrapper(*_, **__):
        return parse_generic_to_xdsl_module(func.__doc__)

    return wrapper


def xdsl_module(func: JaxJittedFunction) -> Callable[Any, xbuiltin.ModuleOp]:  # pragma: no cover
    """
    Decorator for _xdsl_module_inline
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> xbuiltin.ModuleOp:
        return _xdsl_module_inline(func, *args, **kwargs)

    return wrapper
