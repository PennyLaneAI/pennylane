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

"""A drop-in, extended replacement for ``qp.registers``

The stock ``qml.registers`` returns a plain ``dict[str, Wires]``. This subclass
preserves that exact behaviour/return value but adds wire-reindexing arithmetic:
``reg1 + reg2`` concatenates the two register sets, offsetting the second set's
integer wire labels so they don't collide with the first.

Register-name clashes (the same key in both operands) are resolved rather than
blindly offset; see ``__add__`` for the rules.
"""

from __future__ import annotations

import warnings

import pennylane as qml
from pennylane.wires import Wires


class registers(dict):
    """``dict[str, Wires]`` produced like ``qml.registers`` plus concatenation.

    Parameters
    ----------
    register_dict : dict | None
        Same as ``qml.registers``: name -> int (n wires) or nested dict. May also
        be an already-built name -> ``Wires`` mapping (used internally).
    verbose : bool
        If ``True`` (default), emit a ``UserWarning`` when a name clash is resolved
        by merging differing-but-same-size wire sets.
    """

    def __init__(self, register_dict=None, verbose: bool = True):
        self.verbose = verbose
        if register_dict is None:
            super().__init__()
        elif isinstance(register_dict, dict) and all(
            isinstance(v, Wires) for v in register_dict.values()
        ):
            # Already a mapping of name -> Wires (e.g. produced internally by __add__).
            super().__init__(register_dict)
        elif isinstance(register_dict, dict):
            # Spec dict (int / nested dict values): defer to PennyLane for the
            # canonical flattening / nesting semantics.
            super().__init__(qml.registers(register_dict))
        else:
            raise ValueError(f"Expected a dict, got {type(register_dict).__name__}.")

    # -- helpers --------------------------------------------------------------
    @property
    def n_wires(self) -> int:
        """Number of distinct wires spanned by this register set."""
        return len({l for w in self.values() for l in w.labels})

    def _offset(self) -> int:
        """Smallest non-negative offset that clears all existing int labels."""
        int_labels = [l for w in self.values() for l in w.labels if isinstance(l, int)]
        return (max(int_labels) + 1) if int_labels else 0

    @staticmethod
    def _labels(wires) -> set:
        return set(wires.labels)

    # -- clash resolution -----------------------------------------------------
    def _resolve_clash(self, name, existing, incoming, others: set):
        """Resolve a register-name clash on ``name``.

        ``existing`` = self[name], ``incoming`` = other[name] (both raw, un-offset).
        ``others`` is the set of wire labels used by every *other* register in the
        merged result. Returns the chosen ``Wires`` (or ``None`` to drop the key,
        which only happens when an identical entry already exists).
        """
        if len(existing) != len(incoming):
            raise ValueError(
                f"Register {name!r} clashes with different sizes "
                f"({len(existing)} vs {len(incoming)}); refusing to merge."
            )

        if self._labels(existing) == self._labels(incoming):
            return existing  # identical -> keep one, drop the duplicate silently

        # Same size, different wires: keep whichever does not collide with the rest.
        e_ok = self._labels(existing).isdisjoint(others)
        i_ok = self._labels(incoming).isdisjoint(others)

        if e_ok and not i_ok:
            chosen = existing
        elif i_ok and not e_ok:
            chosen = incoming
        elif e_ok and i_ok:
            chosen = existing  # both safe -> prefer the left/existing operand
        else:
            raise ValueError(
                f"Register {name!r} clashes ({list(existing.labels)} vs "
                f"{list(incoming.labels)}); neither is disjoint from the other "
                "registers, so the merge is ambiguous."
            )

        if self.verbose:
            warnings.warn(
                f"Register {name!r} appears in both operands with different but "
                f"same-size wires ({list(existing.labels)} vs {list(incoming.labels)}); "
                f"merged by keeping {list(chosen.labels)}.",
                UserWarning,
                stacklevel=3,
            )
        return chosen

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, other) -> registers:
        if not isinstance(other, dict):
            return NotImplemented
        offset = self._offset()
        merged = dict(self)
        for name, wires in other.items():
            if name not in merged:
                # Disjoint name: reindex so it can't collide with the left operand.
                merged[name] = Wires(
                    [l + offset if isinstance(l, int) else l for l in wires.labels]
                )
            else:
                # Name clash: compare RAW wires (no offset) per the resolution rules.
                others = {l for k, w in merged.items() if k != name for l in w.labels}
                chosen = self._resolve_clash(name, merged[name], wires, others)
                merged[name] = chosen
        return registers(merged, verbose=self.verbose)

    def __radd__(self, other):
        # Enables sum([...], start=registers()) and 0 + reg.
        if other == 0 or other is None:
            return registers(self, verbose=self.verbose)
        return NotImplemented

    def __repr__(self) -> str:
        return f"registers({super().__repr__()})"
