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

from collections.abc import Callable, Sequence
from functools import wraps
from typing import TypeAlias

from catalyst import QJIT
from jax._src.lib import _jax
from jaxlib.mlir.dialects import stablehlo as jstablehlo  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Context as jContext  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Module as jModule  # pylint: disable=no-name-in-module
from xdsl.context import Context as xContext
from xdsl.dialects import arith as xarith
from xdsl.dialects import builtin as xbuiltin
from xdsl.dialects import func as xfunc
from xdsl.dialects import scf as xscf
from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects import tensor as xtensor
from xdsl.dialects import transform as xtransform
from xdsl.ir import Dialect as xDialect
from xdsl.parser import Parser as xParser
from xdsl.traits import SymbolTable as xSymbolTable

from .dialects import MBQC, Catalyst, Quantum

JaxJittedFunction: TypeAlias = _jax.PjitFunction  # pylint: disable=c-extension-no-member


class QuantumParser(xParser):  # pylint: disable=abstract-method,too-few-public-methods
    """A subclass of ``xdsl.parser.Parser`` that automatically loads relevant dialects
    into the input context.

    Args:
        ctx (xdsl.context.Context): Context to use for parsing.
        input (str): Input program string to parse.
        name (str): The name for the input. ``"<unknown>"`` by default.
        extra_dialects (Sequence[xdsl.ir.Dialect]): Any additional dialects
            that should be loaded into the context before parsing.
    """

    default_dialects: tuple[xDialect] = (
        xarith.Arith,
        xbuiltin.Builtin,
        xfunc.Func,
        xscf.Scf,
        xstablehlo.StableHLO,
        xtensor.Tensor,
        xtransform.Transform,
        Quantum,
        MBQC,
        Catalyst,
    )

    def __init__(
        self,
        ctx: xContext,
        input: str,
        name: str = "<unknown>",
        extra_dialects: Sequence[xDialect] | None = (),
    ) -> None:
        super().__init__(ctx, input, name)

        extra_dialects = extra_dialects or ()
        for dialect in self.default_dialects + tuple(extra_dialects):
            self.ctx.load_dialect(dialect)


def _module_inline(func: JaxJittedFunction, *args, **kwargs) -> jModule:
    """Get the module from the jax.jitted function"""
    return func.lower(*args, **kwargs).compiler_ir()


def module(func: JaxJittedFunction) -> Callable[..., jModule]:
    """Decorator for _module_inline"""

    @wraps(func)
    def wrapper(*args, **kwargs) -> jModule:
        return _module_inline(func, *args, **kwargs)

    return wrapper


def _generic_inline(func: JaxJittedFunction, *args, **kwargs) -> str:  # pragma: no cover
    """Create the generic textual representation for the jax.jit'ed function"""
    lowered = func.lower(*args, **kwargs)
    mod = lowered.compiler_ir()
    return mod.operation.get_asm(binary=False, print_generic_op_form=True, assume_verified=True)


def generic(func: JaxJittedFunction) -> Callable[..., str]:  # pragma: no cover
    """Decorator for _generic_inline."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        return _generic_inline(func, *args, **kwargs)

    return wrapper


def parse_generic_to_xdsl_module(
    program: str, extra_dialects: Sequence[xDialect] | None = None
) -> xbuiltin.ModuleOp:  # pragma: no cover
    """Parses generic MLIR program to xDSL module"""
    ctx = xContext(allow_unregistered=True)
    parser = QuantumParser(ctx, program, extra_dialects=extra_dialects)
    moduleOp: xbuiltin.ModuleOp = parser.parse_module()
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


def xdsl_module(func: JaxJittedFunction) -> Callable[..., xbuiltin.ModuleOp]:  # pragma: no cover
    """Decorator for _xdsl_module_inline"""

    @wraps(func)
    def wrapper(*args, **kwargs) -> xbuiltin.ModuleOp:
        return _xdsl_module_inline(func, *args, **kwargs)

    return wrapper


def inline_module(
    from_mod: xbuiltin.ModuleOp, to_mod: xbuiltin.ModuleOp, change_main_to: str = None
) -> None:
    """Inline the contents of one xDSL module into another xDSL module. The inlined body is appended
    to the end of ``to_mod``."""
    if change_main_to:
        main = xSymbolTable.lookup_symbol(from_mod, "main")
        if main is not None:
            assert isinstance(main, xfunc.FuncOp)
            main.properties["sym_name"] = xbuiltin.StringAttr(change_main_to)

    for op in from_mod.body.ops:
        xSymbolTable.insert_or_update(to_mod, op.clone())


def inline_jit_to_module(func: JaxJittedFunction, mod: xbuiltin.ModuleOp, *args, **kwargs) -> None:
    """Inline a ``jax.jit``-ed Python function to an xDSL module. The inlined body is appended
    to the end of ``mod``."""
    func_mod = _xdsl_module_inline(func, *args, **kwargs)
    inline_module(func_mod, mod, change_main_to=func.__name__)


def xdsl_from_qjit(func: QJIT) -> Callable[..., xbuiltin.ModuleOp]:
    """Decorator to convert QJIT-ed functions into xDSL modules."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func.jaxpr, *_ = func.capture(args, **kwargs)
        mlir_module = func.generate_ir()
        generic_str = mlir_module.operation.get_asm(
            binary=False, print_generic_op_form=True, assume_verified=True
        )
        return parse_generic_to_xdsl_module(generic_str)

    return wrapper
