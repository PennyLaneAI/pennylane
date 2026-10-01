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
Adds a tool for annotating things with compiler hints.
"""

import functools
from collections.abc import Callable, Set
from difflib import SequenceMatcher
from typing import Any


def process_hints(hints: dict[str, Any], supported: Set[str]) -> dict[str, Any]:
    """Return a copy of ``hints`` remapped onto canonical supported keys.

    Exact matches are kept. Close misspellings are rewritten to the matching
    supported key so typos like ``"num_iters"`` become ``"num-iters"``.
    Similarity uses a case-insensitive :class:`~difflib.SequenceMatcher` ratio
    with a cutoff of ``0.8``. Unrecognized keys are ignored.

    Args:
        hints (dict[str, Any]): user-provided compiler hints
        supported (Set[str]): allowed canonical hint keys

    Returns:
        dict[str, Any]: hints keyed only by names in ``supported``

    >>> process_hints({"num_iters": 10}, {"num-iters"})
    {'num-iters': 10}
    >>> process_hints({"numiters": 10}, {"num-iters"})
    {'num-iters': 10}

    """
    processed: dict[str, Any] = {}
    for key, value in hints.items():
        if key in supported:
            canonical = key
        else:
            canonical = None
            for target in sorted(supported):
                ratio = SequenceMatcher(a=key.casefold(), b=target.casefold()).ratio()
                if ratio >= 0.8:
                    canonical = target
                    break

        if canonical is None:
            continue
        if canonical in processed:
            raise ValueError(f"Multiple hint keys map to {canonical!r}. Got {tuple(hints)}.")
        processed[canonical] = value
    return processed


class HintedCallable:
    """A function annotated with compiler hints.

    Args:
        f (Callable): a generic function that should be annotated
        hints (dict[str, Any]): a dictionary of hints to be applied

    """

    def __init__(self, f: Callable, hints: dict[str, Any]):
        self._f = f
        self._hints = hints
        functools.update_wrapper(self, f)

    def __repr__(self):
        return f"<HintedCallable({self.f}, {self.hints})>"

    @property
    def f(self) -> Callable:
        """The callable to be annotated."""
        return self._f

    @property
    def hints(self) -> dict[str, Any]:
        """The compiler hints for the function."""
        return self._hints

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


@functools.singledispatch
def apply_hint(f, hints: dict[str, Any]):
    """A single dispatch function that applies a hint to a certain type of object.

    Args:
        f : the thing to apply the hints to
        hints (dict[str, Any]): a dictionary of hints to be applied

    This is a single dispatch function, and custom behaviour for more types of classes can
    be registered. For example, the ``ForLoopCallable`` produced by :func:`~for_loop` can
    have a custom way of applying the hint.

    By default, all callables are converted to a :class:`~HintedCallable` for deferred handling.

    >>> def f(x): return x
    >>> hinted_f = qp.hint({"identity": True})(f)
    >>> hinted_f
    <HintedCallable(<function f at 0x113e21260>, {'identity': True})>
    >>> hinted_f.hints
    {'identity': True}

    """
    raise NotImplementedError(
        f"No registered way to apply compiler hints to object of type {type(f)}"
    )


@apply_hint.register
def _apply_to_callable(f: Callable, hints: dict) -> HintedCallable:
    return HintedCallable(f, hints)


@apply_hint.register
def _stack_to_HintedCallable(f: HintedCallable, hints: dict) -> HintedCallable:
    return HintedCallable(f, f.hints | hints)


def hint(hints: dict[str, Any]) -> Callable:
    """Create a decorator for applying compiler hints.

    Args:
        hints (dict[str, Any]): a dictionary of compiler hints

    Returns:
        Callable: a decorator that can be applied.

    **Available Hints:**

    * :func:`~.for_loop` supports `"num-iters"` to indicate the number of loops
    : :func:`~.while_loop` supports `"num-iters"` to indicate the number of loops

    By hinting control flow like :func:`~.for_loop` and :func:`~.while_loop`, profiling
    with :func:`~.specs` can fully specify the number of iterations.

    .. code-block:: python

        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit', wires=1))
        def c(n):

            @qp.for_loop(n)
            def loop(i):
                qp.X(0)

            #  hinted loop
            qp.hint({"num-iters": 10})(loop)()

            # normal loop
            loop()

            return qp.expval(qp.Z(0))

    >>> print(qp.specs(c, level=0)(5).resources)
    Symbolic Variables: a
    Quantum operations:
    - Total: a + 10
      - PauliX: a + 10
    Measurement processes:
    - expval(PauliZ): 1
    Total wires: 1
    Circuit Depth: Not computed

    The concrete ``10`` corresponds to the hinted loop, contrasting the
    symbolic ``a`` from to the unhinted loop.

    This function can also  be used as a decorator:

    .. code-block:: python

        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit' wires=10))
        def c(n):

            @qp.hint({"num-iters": 10})
            @qp.while_loop(lambda i: i < 10)
            def loop(i):
                qp.X(i)
                return i + 1

            loop(0)
            return qp.expval(qp.Z(0))

    """

    def decorator(f):
        return apply_hint(f, hints)

    return decorator
