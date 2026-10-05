# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Pragmas are source code comments that influence how AutoGraph converts the statement they
annotate. They are written as ``# <name>: <body>``, where ``<name>`` must start with ``qp`` so
that ordinary comments like ``# TODO: ...`` are never mistaken for a pragma.

A pragma may be placed at the end of the statement it annotates, or on its own line(s) directly
above it:

.. code-block:: python

    for i in range(n):  # qphint: num-iters=10
        qp.X(0)

    # qphint: num-iters=10
    while i < n:
        i = i + 1

Comments do not appear in the abstract syntax tree, so they are recovered by tokenizing the
source code of the function being converted and matching comments to statements by line number.
"""

import ast
import io
import re
import tokenize
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from malt.lang import directives
from malt.pyct import anno

from pennylane.exceptions import AutoGraphError, AutoGraphWarning

PRAGMA_PATTERN = re.compile(r"^#\s*(?P<name>qp\w+)\s*:\s*(?P<body>.*)$")

_NON_CODE_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
)


@dataclass(frozen=True)
class Pragma:
    """A pragma recovered from a source code comment.

    Args:
        name (str): the pragma name, i.e. the text before the colon
        body (str): the text after the colon
        lineno (int): the line the comment itself was written on
    """

    name: str
    body: str
    lineno: int


@dataclass
class GeneratedEdit:
    """A keyword argument to add to the AutoGraph call generated for an annotated statement.

    Some statements, such as ``if``, offer no annotation that AutoGraph carries through to its
    own primitives. For those, the pragma is recorded here and applied to the generated call
    once the AutoGraph passes have run, matching the call back to the statement by the line it
    originated from.

    Args:
        lineno (int): the line the annotated statement originated from
        call (str): the name of the generated ``ag__`` call to add the keyword argument to
        keyword (str): the name of the keyword argument
        values (dict[str, ast.expr]): the dictionary to pass as the keyword argument
    """

    lineno: int
    call: str
    keyword: str
    values: dict[str, ast.expr]

    @property
    def key(self) -> tuple[int, str, str]:
        """Identifies the keyword argument this edit contributes to."""
        return (self.lineno, self.call, self.keyword)


PRAGMA_HANDLERS: dict[str, Callable[[ast.stmt, Pragma, str | None], GeneratedEdit | None]] = {}


def register_pragma(name: str) -> Callable:
    """Register the handler for a pragma comment.

    The handler receives the statement the pragma is attached to, the :class:`~.Pragma` itself,
    and which clause of the statement was annotated, which is ``None`` except for a pragma
    written on an ``else:`` line, where it is ``"orelse"``. A handler either annotates the
    statement in place and returns ``None``, or returns a :class:`~.GeneratedEdit` to be applied
    once the AutoGraph passes have run.

    Args:
        name (str): the pragma name, which must start with ``qp``

    Returns:
        Callable: a decorator registering the handler it is applied to
    """

    def decorator(handler):
        PRAGMA_HANDLERS[name] = handler
        return handler

    return decorator


def transform(node: ast.AST, ctx) -> tuple[ast.AST, list[GeneratedEdit]]:
    """Apply the pragmas found in the source code of the entity being converted.

    Args:
        node (ast.AST): the syntax tree of the entity being converted
        ctx (malt.pyct.transformer.Context): the transformation context

    Returns:
        tuple[ast.AST, list[GeneratedEdit]]: the annotated syntax tree, and the edits to apply
        to the code AutoGraph generates from it
    """
    pragmas = _scan_pragmas(ctx.info.source_code)
    if not pragmas:
        return node, []

    transformer = PragmaTransformer(pragmas)
    transformer.visit(node)
    transformer.warn_unattached()
    return node, transformer.edits


def apply_edits(node: ast.AST, edits: list[GeneratedEdit]) -> ast.AST:
    """Add the keyword arguments requested by pragmas to the calls AutoGraph generated.

    Args:
        node (ast.AST): the converted syntax tree
        edits (list[GeneratedEdit]): the edits collected while applying pragmas

    Returns:
        ast.AST: the converted syntax tree
    """
    if not edits:
        return node

    by_target: dict[tuple[int, str], list[GeneratedEdit]] = {}
    for edit in edits:
        by_target.setdefault((edit.lineno, edit.call), []).append(edit)

    applied = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        origin = anno.getanno(call, anno.Basic.ORIGIN, default=None)
        if origin is None:
            continue
        for edit in by_target.get((origin.loc.lineno, call.func.attr), ()):
            call.keywords.append(ast.keyword(arg=edit.keyword, value=_as_dict(edit.values)))
            applied.add(edit.key)

    for edit in edits:
        if edit.key not in applied:
            warnings.warn(
                f"A pragma annotating the statement on line {edit.lineno} could not be applied, "
                f"because AutoGraph did not generate a '{edit.call}' call for it.",
                AutoGraphWarning,
            )

    return node


class PragmaTransformer(ast.NodeVisitor):
    """Dispatches each pragma to its handler, applying it to the statement it annotates.

    Args:
        pragmas (dict[int, list[Pragma]]): pragmas keyed by the line of the statement they
            annotate, as produced by ``_scan_pragmas``
    """

    def __init__(self, pragmas: dict[int, list[Pragma]]):
        self._pragmas = pragmas
        self._edits: dict[tuple[int, str, str], GeneratedEdit] = {}

    @property
    def edits(self) -> list[GeneratedEdit]:
        """The edits to apply to the code AutoGraph generates."""
        return list(self._edits.values())

    def visit(self, node):
        if isinstance(node, ast.stmt):
            # A one-line compound statement shares its line with the statements in its body.
            # Parents are visited first, so removing the entry here gives the pragma to the
            # outermost statement on the line.
            for pragma in self._pragmas.pop(node.lineno, ()):
                self._dispatch(node, pragma, None)
            for lineno in _else_linenos(node):
                for pragma in self._pragmas.pop(lineno, ()):
                    self._dispatch(node, pragma, "orelse")
        self.generic_visit(node)

    def warn_unattached(self):
        """Warn about any pragma that could not be matched to a statement."""
        for pragmas in self._pragmas.values():
            for pragma in pragmas:
                _warn_unattached(pragma)

    def _dispatch(self, node: ast.stmt, pragma: Pragma, branch: str | None):
        handler = PRAGMA_HANDLERS.get(pragma.name)
        if handler is None:
            known = ", ".join(repr(name) for name in sorted(PRAGMA_HANDLERS))
            warnings.warn(
                f"Unknown AutoGraph pragma '{pragma.name}' on line {pragma.lineno} will be "
                f"ignored. The available pragmas are {known}.",
                AutoGraphWarning,
            )
            return

        edit = handler(node, pragma, branch)
        if edit is None:
            return

        existing = self._edits.get(edit.key)
        if existing is None:
            self._edits[edit.key] = edit
            return

        for key, value in edit.values.items():
            if key in existing.values:
                raise AutoGraphError(
                    f"The '{pragma.name}' pragma on line {pragma.lineno} sets '{key}' for a "
                    f"statement that already has it set."
                )
            existing.values[key] = value


def _else_linenos(node: ast.stmt) -> list[int]:
    """Find the lines of an ``else:`` clause that carry a pragma.

    There is no syntax tree node for an ``else:`` clause, so a pragma written on that line
    belongs to no statement. It is instead recovered from the gap between the end of the body
    and the start of the ``orelse`` block. An ``elif`` leaves no gap, since its own ``if``
    node starts on the ``elif`` line, so it is picked up as an ordinary statement instead.
    """
    orelse = getattr(node, "orelse", None)
    if not orelse or not node.body:
        return []
    return list(range(node.body[-1].end_lineno + 1, orelse[0].lineno))


def _as_dict(values: dict[str, ast.expr]) -> ast.Dict:
    """Build the dictionary literal to pass to a generated call."""
    return ast.Dict(
        keys=[ast.Constant(key) for key in values],
        values=list(values.values()),
    )


def _tokenize(source: str) -> list[tokenize.TokenInfo]:
    """Tokenize source code, discarding an incomplete tail.

    Source code resolution is brittle for lambdas and may produce a fragment that cannot be
    fully tokenized. Whatever was tokenized before the error is still usable.
    """
    tokens = []
    try:
        tokens.extend(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        pass
    return tokens


def _scan_pragmas(source: str) -> dict[int, list[Pragma]]:
    """Collect the pragmas in the given source, keyed by the line of the statement they annotate.

    A pragma that trails other code belongs to the statement on its own line. A pragma occupying
    a whole line belongs to the next statement, so it is held back until a line of actual code
    is reached.
    """
    pragmas: dict[int, list[Pragma]] = {}
    pending: list[Pragma] = []

    for token in _tokenize(source):
        if token.type == tokenize.COMMENT:
            match = PRAGMA_PATTERN.match(token.string.strip())
            if match is None:
                continue
            pragma = Pragma(match["name"], match["body"].strip(), token.start[0])
            if token.line[: token.start[1]].strip():
                pragmas.setdefault(token.start[0], []).append(pragma)
            else:
                pending.append(pragma)
        elif pending and token.type not in _NON_CODE_TOKENS:
            pragmas.setdefault(token.start[0], []).extend(pending)
            pending = []

    for pragma in pending:
        _warn_unattached(pragma)

    return pragmas


def _warn_unattached(pragma: Pragma):
    warnings.warn(
        f"The '{pragma.name}' pragma on line {pragma.lineno} is not attached to a statement "
        f"and will be ignored. Place it at the end of a statement, or on its own line directly "
        f"above one.",
        AutoGraphWarning,
    )


@register_pragma("qphint")
def _apply_qphint(node: ast.stmt, pragma: Pragma, branch: str | None) -> GeneratedEdit | None:
    """Apply ``# qphint: key=value`` to a loop or to one branch of an ``if`` statement.

    On a loop, the hints are recorded as AutoGraph loop options, which AutoGraph passes to the
    ``for_stmt`` and ``while_stmt`` primitives, where they become a :func:`~.hint` on the
    generated :func:`~.for_loop` or :func:`~.while_loop`. On an ``if``, they are deferred and
    added to the generated ``if_stmt`` call, where they become a :func:`~.hint` on the branch.
    """
    if isinstance(node, ast.If):
        return _hint_if_branch(node, pragma, branch)

    if not isinstance(node, (ast.For, ast.While)):
        raise AutoGraphError(
            f"The 'qphint' pragma on line {pragma.lineno} can only annotate a 'for', 'while' or "
            f"'if' statement, but it annotates a '{type(node).__name__}' statement."
        )

    if branch is not None:
        raise AutoGraphError(
            f"The 'qphint' pragma on line {pragma.lineno} annotates the 'else' clause of a loop, "
            f"which AutoGraph does not convert. Annotate the loop itself instead."
        )

    _reject_explicit_loop_options(node, pragma)

    annotation = dict(anno.getanno(node, anno.Basic.DIRECTIVES, {}))
    options = dict(annotation.get(directives.set_loop_options, {}))
    for key, value in _parse_hints(pragma).items():
        if key in options:
            raise AutoGraphError(
                f"The 'qphint' pragma on line {pragma.lineno} sets '{key}' for a loop that "
                f"already has it set."
            )
        options[key] = value

    annotation[directives.set_loop_options] = options
    anno.setanno(node, anno.Basic.DIRECTIVES, annotation)
    return None


def _hint_if_branch(node: ast.If, pragma: Pragma, branch: str | None) -> GeneratedEdit:
    """Record the hints for one branch of an ``if`` statement.

    AutoGraph offers no annotation for ``if`` statements the way it does for loops, so the
    hints are carried to the generated ``if_stmt`` call instead, identified by the line the
    statement originated from.
    """
    origin = anno.getanno(node, anno.Basic.ORIGIN, default=None)
    if origin is None:
        raise AutoGraphError(
            f"The 'qphint' pragma on line {pragma.lineno} annotates a statement with no source "
            f"location, so it cannot be applied."
        )

    keyword = "false_hints" if branch == "orelse" else "true_hints"
    return GeneratedEdit(origin.loc.lineno, "if_stmt", keyword, _parse_hints(pragma))


def _reject_explicit_loop_options(node: ast.stmt, pragma: Pragma):
    """Reject a loop that carries both a pragma and a ``set_loop_options`` call.

    AutoGraph's own directives converter runs after this pass and replaces the whole set of loop
    options, which would silently discard the pragma.
    """
    if not node.body:
        return
    first = node.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Call):
        return

    func = first.value.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name == "set_loop_options":
        raise AutoGraphError(
            f"The loop annotated by the 'qphint' pragma on line {pragma.lineno} also calls "
            f"'set_loop_options', which would discard the pragma. Please use only one of the two."
        )


def _parse_hints(pragma: Pragma) -> dict[str, ast.expr]:
    """Parse the body of a ``qphint`` pragma into hint keys and their value expressions."""
    hints: dict[str, ast.expr] = {}

    for piece in _split_pairs(pragma.body):
        key, separator, value = piece.partition("=")
        key = key.strip()
        if not separator or not key:
            raise AutoGraphError(
                f"Malformed 'qphint' pragma on line {pragma.lineno}: expected comma separated "
                f"'key=value' pairs, but got {piece!r}."
            )
        if key in hints:
            raise AutoGraphError(
                f"The 'qphint' pragma on line {pragma.lineno} sets '{key}' more than once."
            )
        hints[key] = _parse_value(key, value.strip(), pragma)

    if not hints:
        raise AutoGraphError(
            f"The 'qphint' pragma on line {pragma.lineno} is empty. It should be written as "
            f"'# qphint: key=value'."
        )

    return hints


def _parse_value(key: str, value: str, pragma: Pragma) -> ast.expr:
    """Validate a hint value and return it as an expression to embed in the converted code."""
    try:
        ast.literal_eval(value)
    except (MemoryError, SyntaxError, TypeError, ValueError) as error:
        raise AutoGraphError(
            f"The value of '{key}' in the 'qphint' pragma on line {pragma.lineno} must be a "
            f"Python literal, but got {value!r}."
        ) from error
    return ast.parse(value, mode="eval").body


def _split_pairs(body: str) -> list[str]:
    """Split a pragma body on the commas that separate its pairs.

    Commas inside brackets belong to a value rather than separating two pairs.
    """
    pieces = []
    depth = 0
    start = 0

    for token in _tokenize(body):
        if token.type != tokenize.OP:
            continue
        if token.string in "([{":
            depth += 1
        elif token.string in ")]}":
            depth -= 1
        elif token.string == "," and depth == 0:
            pieces.append(body[start : token.start[1]])
            start = token.end[1]

    pieces.append(body[start:])
    return [piece for piece in (piece.strip() for piece in pieces) if piece]
