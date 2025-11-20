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

"""Utilities for converting to xDSL module."""

from collections.abc import Callable, Sequence
from functools import wraps
from typing import TypeAlias

from catalyst import QJIT
from jax._src.lib import _jax
from jaxlib.mlir.dialects import stablehlo as jstablehlo  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Context as jContext  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Module as jModule  # pylint: disable=no-name-in-module
from xdsl.context import Context as xContext
from xdsl.dialects import builtin as xbuiltin
from xdsl.dialects import func as xfunc
from xdsl.ir import Dialect as xDialect
from xdsl.traits import SymbolTable as xSymbolTable

from .parser import QuantumParser

JaxJittedFunction: TypeAlias = _jax.PjitFunction  # pylint: disable=c-extension-no-member


def _mlir_module_inline(func: JaxJittedFunction, *args, **kwargs) -> jModule:
    """Get the MLIR module from a jax.jitted function"""
    return func.lower(*args, **kwargs).compiler_ir()


def mlir_module(func: JaxJittedFunction) -> Callable[..., jModule]:
    """Returns a wrapper that creates an MLIR module from a jax.jitted function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> jModule:
        return _mlir_module_inline(func, *args, **kwargs)

    return wrapper


def _generic_str_inline(func: JaxJittedFunction, *args, **kwargs) -> str:  # pragma: no cover
    """Create the generic textual representation for a jax.jitted function"""
    lowered = func.lower(*args, **kwargs)
    mod = lowered.compiler_ir()
    return mod.operation.get_asm(binary=False, print_generic_op_form=True, assume_verified=True)


def generic_str(func: JaxJittedFunction) -> Callable[..., str]:  # pragma: no cover
    """Returns a wrapper that creates the generic textual representation for a
    jax.jitted function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        return _generic_str_inline(func, *args, **kwargs)

    return wrapper


def parse_generic_to_xdsl_module(
    program: str, extra_dialects: Sequence[xDialect] | None = None
) -> xbuiltin.ModuleOp:  # pragma: no cover
    """Parses a generic MLIR program string to an xDSL module."""
    ctx = xContext(allow_unregistered=True)
    parser = QuantumParser(ctx, program, extra_dialects=extra_dialects)
    moduleOp: xbuiltin.ModuleOp = parser.parse_module()
    return moduleOp


def parse_generic_to_mlir_module(program: str) -> jModule:  # pragma: no cover
    """Parses a generic MLIR program string to an MLIR module."""
    with jContext() as ctx:
        ctx.allow_unregistered_dialects = True
        jstablehlo.register_dialect(ctx)  # pylint: disable=no-member
        return jModule.parse(program)


def mlir_from_docstring(func: Callable) -> jModule:  # pragma: no cover
    """Returns a wrapper that parses an MLIR program string located in the docstring
    into an MLIR module."""

    @wraps(func)
    def wrapper(*_, **__):
        return parse_generic_to_mlir_module(func.__doc__)

    return wrapper


def _xdsl_module_inline(
    func: JaxJittedFunction, *args, **kwargs
) -> xbuiltin.ModuleOp:  # pragma: no cover
    """Get the xDSL module from a jax.jitted function"""
    generic_repr = _generic_str_inline(func, *args, **kwargs)
    return parse_generic_to_xdsl_module(generic_repr)


def xdsl_from_docstring(func: Callable) -> xbuiltin.ModuleOp:  # pragma: no cover
    """Returns a wrapper that parses an MLIR program string located in the docstring
    into an xDSL module."""

    @wraps(func)
    def wrapper(*_, **__):
        return parse_generic_to_xdsl_module(func.__doc__)

    return wrapper


def xdsl_module(func: JaxJittedFunction) -> Callable[..., xbuiltin.ModuleOp]:  # pragma: no cover
    """Returns a wrapper that creates an xDSL module from a jax.jitted function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> xbuiltin.ModuleOp:
        return _xdsl_module_inline(func, *args, **kwargs)

    return wrapper


def inline_module(
    from_mod: xbuiltin.ModuleOp, to_mod: xbuiltin.ModuleOp, change_main_to: str = None
) -> None:
    """Inline the contents of one xDSL module into another xDSL module. The inlined body is appended
    to the end of ``to_mod``.

    If ``from_mod`` has a ``main`` function, its name is changed to ``change_main_to`` if specified.
    """
    if change_main_to:
        main = xSymbolTable.lookup_symbol(from_mod, "main")
        if main is not None:
            assert isinstance(main, xfunc.FuncOp)
            main.properties["sym_name"] = xbuiltin.StringAttr(change_main_to)

    for op in from_mod.body.ops:
        xSymbolTable.insert_or_update(to_mod, op.clone())


def inline_jit_to_module(func: JaxJittedFunction, mod: xbuiltin.ModuleOp) -> Callable[..., None]:
    """Inline a ``jax.jit``-ed Python function to an xDSL module. The inlined body is appended
    to the end of ``mod`` in-place. The name of the entry point function of ``func`` is the same
    as the name of ``func``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_mod = _xdsl_module_inline(func, *args, **kwargs)
        inline_module(func_mod, mod, change_main_to=func.__name__)

    return wrapper


def xdsl_from_qjit(func: QJIT) -> Callable[..., xbuiltin.ModuleOp]:
    """Decorator to convert QJIT-ed functions into xDSL modules."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func.jaxpr, *_ = func.capture(args, **kwargs)
        _mlir_module = func.generate_ir()
        _generic_str = _mlir_module.operation.get_asm(
            binary=False, print_generic_op_form=True, assume_verified=True
        )
        return parse_generic_to_xdsl_module(_generic_str)

    return wrapper
