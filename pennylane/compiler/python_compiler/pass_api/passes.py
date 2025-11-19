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
"""This file contains PennyLane's API for defining compiler passes."""

from collections.abc import Callable
from inspect import signature
from types import UnionType
from typing import ClassVar, Union, get_args, get_origin

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def _update_type_hints(hint: type[Operation] | type[Operation]) -> Callable:
    """Update the signature of a ``match_and_rewrite`` method to use the provided operation
    as the first argument's type hint."""
    if get_origin(hint) in (Union, UnionType):
        expected_types = get_args(hint)
    else:
        expected_types = (hint,)

    if not all(issubclass(e, Operation) for e in expected_types):
        raise TypeError(
            "Only Operation types or unions of Operation types can be used to "
            "register rewrite rules."
        )

    def _update_match_and_rewrite(method: Callable) -> Callable:
        params = tuple(signature(method).parameters)
        assert len(params) == 3 and params[0] == "self"

        # Update type hint of operation argument
        # TODO: Is it fine to mutate in-place or should we return a new function?
        op_arg_name = params[-2]
        method.__annotations__[op_arg_name] = hint

        return method

    return _update_match_and_rewrite


def _create_rewrite_pattern(
    hint: type[Operation] | type[Operation], rewrite_rule: Callable
) -> RewritePattern:
    """Given a rewrite rule defined as a function, create a RewritePattern which
    can be used with xDSL's pass API."""

    # pylint: disable=too-few-public-methods, arguments-differ
    class _RewritePattern(RewritePattern):
        """Unnamed rewrite pattern for transforming a matched operation."""

        _pass: PLModulePass

        def __init__(self, _pass):
            self._pass = _pass
            super().__init__()

        @op_type_rewrite_pattern
        @_update_type_hints(hint)
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
            rewrite_rule(self._pass, op, rewriter)

    return _RewritePattern


class PLModulePass(ModulePass):
    """A xdsl ModulePass subclass for defining passes."""

    name: ClassVar[str]
    recursive: bool
    greedy: bool
    _rewrite_patterns: ClassVar[dict[Operation, RewritePattern]] = {}

    def __init__(self, recursive: bool = True, greedy: bool = False):
        self.recursive = recursive
        self.greedy = greedy

    @classmethod
    def rewrite_rule(
        cls, hint: type[Operation] | type[Operation]
    ) -> Callable[[Operation, PatternRewriter], Callable]:
        """Decorator to register a rewrite rule.

        The rewrite rule must have the following signature:

        .. code-block:: python

            @PLModulePass.rewrite_rule(MyOperation)
            def rewrite_myop(self, op: MyOperation, rewriter: PatternRewriter) -> None:
                ...

        .. note::

            If a rewrite rule for the provided operation already exists, the old rule
            will get overwritten.

        Args:
            hint (type[Operation]): Operation class for which to register the rule

        Returns:
            Callable: a decorator to register the rewrite rule with the ModulePass
        """

        def decorator(rule: Callable[[Operation, PatternRewriter], None]) -> Callable:
            rewrite_pattern = _create_rewrite_pattern(hint, rule)
            cls._rewrite_patterns[hint] = rewrite_pattern
            return rule

        return decorator

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:  # pylint: disable=unused-argument
        """Apply the transformation to the input module.

        If ``greedy`` is ``True``, the rewrite rules will be applied greedily, i.e., for each
        operation in the worklist, we will apply only the first rewrite rule that matches the
        operation. Otherwise, they will be applied by creating a different worklist for each
        rewrite rule.

        If ``recursive`` is ``True``, the worklist algorithm will continue applying the rewrite
        rules until a steady-state is reached.

        .. note::

            The input module is mutated in-place.

        Args:
            ctx: Context containing operation and attribute registrations
            op: Module to which to apply the transform
        """
        if self.greedy:
            pattern = GreedyRewritePatternApplier(
                rewrite_patterns=[rp(self) for rp in self._rewrite_patterns.values()]
            )
            walker = PatternRewriteWalker(pattern=pattern, apply_recursively=self.recursive)
            walker.rewrite_module(op)

        else:
            for rp in self._rewrite_patterns.values():
                walker = PatternRewriteWalker(pattern=rp(self), apply_recursively=self.recursive)
                walker.rewrite_module(op)
