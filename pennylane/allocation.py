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
This module contains the commands for allocating and freeing wires dynamically.
"""
from functools import lru_cache
from typing import Sequence

try:
    import jax
except ImportError:
    pass

from pennylane.capture import enabled as capture_enabled
from pennylane.operation import Operator
from pennylane.wires import Wires


class DynamicWire:
    """A wire whose concrete value will be determined later during a compilation step or execution.

    Multiple dynamic wires can correspond to the same device wire as long as they are properly allocated and
    freed.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return f"<DynamicWire>"


@lru_cache()
def _get_allocate_prim():
    allocate_prim = jax.extend.core.Primitive("allocate")
    allocate_prim.multiple_results = True

    @allocate_prim.def_impl
    def _(*, num_wires, require_zeros=True, reset_to_original=False):
        raise ValueError("no concrete value for wire available.")

    @allocate_prim.def_abstract_eval
    def _(*, num_wires, require_zeros=True, reset_to_original=False):
        return [jax.core.ShapedArray((), dtype=int) for _ in range(num_wires)]

    return allocate_prim


@lru_cache
def _get_deallocate_prim():

    deallocate_prim = jax.extend.core.Primitive("deallocate")
    deallocate_prim.multiple_results = True

    @deallocate_prim.def_impl
    def _(*wires):
        return []

    @deallocate_prim.def_abstract_eval
    def _(*wires):
        return []

    return deallocate_prim


class Allocate(Operator):
    """An instruction to values for dynamic wires.

    Args:
        wires (list[DynamicWire]): a list of dynamic wire values.

    Keyword Args:
        require_zeros (bool): Whether or not the wire must start in a ``0`` state.
        reset_to_original (bool): Whether or not the qubit will be in the same state upon being freed.

    ..see-also:: :func:`~.allocate`, :func:`~.allocate_ctx`.

    """

    def __init__(self, wires, require_zeros=True, reset_to_original=False):
        super().__init__(wires=wires)
        self._hyperparameters = {
            "require_zeros": require_zeros,
            "reset_to_original": reset_to_original,
        }

    @property
    def require_zeros(self):
        return self.hyperparameters["require_zeros"]

    @property
    def reset_to_original(self):
        return self.hyperparameters["reset_to_original"]

    @classmethod
    def from_num_wires(cls, num_wires: int, require_zeros=True, reset_to_original=False):
        """Initialize an ``Allocate`` op from a number of wires instead of already constructed dynamic wires."""
        wires = tuple(DynamicWire() for _ in range(num_wires))
        return cls(wires=wires, require_zeros=require_zeros, reset_to_original=reset_to_original)


class Deallocate(Operator):

    def __init__(self, wires):
        super().__init__(wires=wires)


class DeallocateAll(Operator):
    pass


class Borrow(Operator):
    """An instruction to borrow wires."""

    def __init__(self, wires):
        super().__init__(wires=wires)


class Return(Operator):
    """An instruction to return borrowed wires."""

    def __init__(self, wires):
        super().__init__(wires=wires)


def allocate(num_wires, require_zeros=True, reset_to_original=False):
    if capture_enabled():
        return _get_allocate_prim().bind(
            num_wires=num_wires, require_zeros=require_zeros, reset_to_original=reset_to_original
        )
    op = Allocate.from_num_wires(
        num_wires, require_zeros=require_zeros, reset_to_original=reset_to_original
    )
    return op.wires


def deallocate(obj):
    if capture_enabled():
        if isinstance(obj, Sequence):
            return _get_deallocate_prim().bind(*obj)
        return _get_deallocate_prim().bind(obj)
    if isinstance(obj, DynamicWire):
        return Deallocate(obj)
    if isinstance(obj, (Wires, Sequence)):
        dynamic_wires = tuple(w for w in obj if isinstance(w, DynamicWire))
        return Deallocate(dynamic_wires)
    raise NotImplementedError


class allocate_ctx:
    """Temporarily allocate dynamic wires while making sure to automatically deallocate them at the end.

    Args:
        num_wires (int): the number of dynamic wires to allocate.

    Keyword Args:
        require_zeros (bool): whether or not the wires must start in the ``0`` state
        reset_to_original (bool): whether or not the wires return to the same state as they started.

    .. code-block:: python

        @qml.qnode(qml.device('default.qubit', wires=("a", "b")))
        def c():
            with qml.allocate_ctx(2, require_zeros=True, reset_to_original=False) as wires:
                qml.CNOT(wires)
            with qml.allocate_ctx(2, require_zeros=True, reset_to_original=False) as wires:
                qml.IsingXX(0.5, wires)
            return qml.probs()


    >>> print(qml.draw(c)())
    <DynamicWire>: ─╭Allocate─╭●─────────────╭Deallocate─┤  Probs
    <DynamicWire>: ─╰Allocate─╰X─────────────╰Deallocate─┤  Probs
    <DynamicWire>: ─╭Allocate─╭IsingXX(0.50)─╭Deallocate─┤  Probs
    <DynamicWire>: ─╰Allocate─╰IsingXX(0.50)─╰Deallocate─┤  Probs
    >>> print(qml.draw(c, level="device")())
    a: ─╭●──┤↗│  │0⟩─╭IsingXX(0.50)─┤ ╭Probs
    b: ─╰X──┤↗│  │0⟩─╰IsingXX(0.50)─┤ ╰Probs

    """

    def __init__(self, num_wires: int, require_zeros: bool = True, reset_to_original: bool = False):
        self.wires = allocate(
            num_wires, require_zeros=require_zeros, reset_to_original=reset_to_original
        )

    def __enter__(self):
        return self.wires

    def __exit__(self, *_, **__):
        deallocate(self.wires)


def borrow(wires):
    """Borrow wires into the pool of work wires."""
    return Borrow(wires)


def return_borrowed(wires):
    """Return wires borrowed into the pool of work wires."""
    return Return(wires)


class borrow_ctx:
    """Temporarily borrow wires while making sure to automatically return them at the end."""

    def __init__(self, wires):
        self.wires = borrow(wires).wires

    def __enter__(self):
        return self.wires

    def __exit__(self, *_, **__):
        return_borrowed(self.wires)
