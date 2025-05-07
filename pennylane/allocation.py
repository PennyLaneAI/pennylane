from enum import Enum
from functools import lru_cache
from typing import Sequence

try:
    import jax
except ImportError:
    pass

from pennylane.capture import enabled as capture_enabled
from pennylane.operation import Operator
from pennylane.transforms.core import transform
from pennylane.wires import Wires


class RegisterTypes(Enum):

    ZEROED = (True, True)
    BURNABLE = (True, False)
    BORROWABLE = (False, True)
    GARBAGE = (False, False)


ZEROED = RegisterTypes.ZEROED
BURNABLE = RegisterTypes.BURNABLE
BORROWABLE = RegisterTypes.BORROWABLE
GARBAGE = RegisterTypes.GARBAGE


class DynamicWire:

    def __init__(self):
        pass

    def __repr__(self):
        return f"<DynamicWire>"


@lru_cache()
def _get_allocate_prim():
    allocate_prim = jax.core.Primitive("allocate")
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

    deallocate_prim = jax.core.Primitive("deallocate")
    deallocate_prim.multiple_results = True

    @deallocate_prim.def_impl
    def _(*wires):
        return []

    @deallocate_prim.def_abstract_eval
    def _(*wires):
        return []

    return deallocate_prim


class Allocate(Operator):

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


class Deallocate(Operator):

    def __init__(self, wires):
        super().__init__(wires=wires)


class DeallocateAll(Operator):
    pass


def allocate(num_wires, require_zeros=True, reset_to_original=False):
    if capture_enabled():
        return _get_allocate_prim().bind(
            num_wires=num_wires, require_zeros=require_zeros, reset_to_original=reset_to_original
        )
    wires = tuple(DynamicWire() for _ in range(num_wires))
    Allocate(wires=wires, require_zeros=require_zeros, reset_to_original=reset_to_original)
    return wires


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

    def __init__(self, num_wires, require_zeros=True, reset_to_original=False):
        self.wires = allocate(
            num_wires, require_zeros=require_zeros, reset_to_original=reset_to_original
        )

    def __enter__(self):
        return self.wires

    def __exit__(self, *_, **__):
        deallocate(self.wires)


class WireManager:

    def __init__(self, zeroed=(), burnable=(), borrowable=(), garbage=()):
        self._registers = {
            ZEROED: list(zeroed),
            BURNABLE: list(burnable),
            BORROWABLE: list(borrowable),
            GARBAGE: list(garbage),
        }
        self._loaned = {}  # wire to register type

    def get_wire(self, require_zeros, reset_to_original):
        reg_type = RegisterTypes((require_zeros, reset_to_original))

        wire = self._registers[reg_type].pop()
        self._loaned[wire] = reg_type
        return wire

    def return_wire(self, wire):
        reg_type = self._loaned.pop(wire)
        self._registers[reg_type].append(wire)
        return

    def return_all(self):
        for w in self._loaned:
            self.return_wire(w)


def null_postprocessing(results):
    return results[0]


@transform
def resolve_dynamic_wires(tape, zeroed=(), burnable=(), borrowable=(), garbage=()):

    manager = WireManager(zeroed=zeroed, burnable=burnable, borrowable=borrowable, garbage=garbage)

    wire_map = {}

    new_ops = []
    for op in tape.operations:
        if isinstance(op, Allocate):
            for w in op.wires:
                wire_map[w] = manager.get_wire(**op.hyperparameters)
        elif isinstance(op, Deallocate):
            for w in op.wires:
                manager.return_wire(wire_map.pop(w))
        elif isinstance(op, DeallocateAll):
            manager.return_all()
        else:
            new_ops.append(op.map_wires(wire_map))
    return (tape.copy(ops=new_ops),), null_postprocessing
