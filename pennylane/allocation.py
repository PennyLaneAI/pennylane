from enum import Enum
from functools import lru_cache, partial
from itertools import chain
from typing import Sequence

try:
    import jax
except ImportError:
    pass

from pennylane.capture import enabled as capture_enabled
from pennylane.measurements import measure
from pennylane.operation import Operator
from pennylane.transforms.core import transform
from pennylane.wires import Wires


class RegisterTypes(Enum):

    ZEROED = (True, True)
    BURNABLE = (True, False)
    BORROWABLE = (False, True)
    GARBAGE = (False, False)
    REQUIRES_RESET = "requires_reset"


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

    def _get_burnable(self):
        if self._registers[BURNABLE]:
            wire = self._registers[BURNABLE].pop()
            self._loaned[wire] = GARBAGE
            return [], wire
        if self._registers[GARBAGE]:
            wire = self._registers[GARBAGE].pop()
            m = measure(wire, reset=True)
            self._loaned[wire] = GARBAGE
            return m.measurements, wire
        if self._registers[ZEROED]:
            wire = self._registers[ZEROED].pop()
            self._loaned[wire] = RegisterTypes.REQUIRES_RESET
            return [], wire
        raise ValueError("no available burnable, garbage, or zeroed wires.")

    def _get_zeroed(self):
        for reg in [ZEROED, BURNABLE]:
            if self._registers[reg]:
                wire = self._registers[reg].pop()
                self._loaned[wire] = reg
                return [], wire
        if self._registers[GARBAGE]:
            wire = self._registers[GARBAGE].pop()
            m = measure(wire, reset=True)
            self._loaned[wire] = BURNABLE
            return m.measurements, wire
        raise ValueError("no available burnable, garbage, or zeroed wires.")

    def _get_garbage(self):
        for reg in [GARBAGE, BURNABLE]:
            if self._registers[reg]:
                wire = self._registers[reg].pop()
                self._loaned[wire] = GARBAGE
                return [], wire
        if self._registers[ZEROED]:
            wire = self._registers[ZEROED].pop()
            self._loaned[wire] = RegisterTypes.REQUIRES_RESET
            return [], wire
        raise ValueError("no available burnable, garbage, or zeroed wires.")

    def _get_borrowable(self):
        for reg in [GARBAGE, BORROWABLE, BURNABLE, ZEROED]:
            if self._registers[reg]:
                wire = self._registers[reg].pop()
                self._loaned[wire] = reg
                return [], wire
        raise ValueError("no available burnable, garbage, borrowable, or zeroed wires.")

    def get_wire(self, require_zeros, reset_to_original):
        reg_type = RegisterTypes((require_zeros, reset_to_original))
        match reg_type:
            case RegisterTypes.BURNABLE:
                return self._get_burnable()
            case RegisterTypes.ZEROED:
                return self._get_zeroed()
            case RegisterTypes.GARBAGE:
                return self._get_garbage()
            case RegisterTypes.BORROWABLE:
                return self._get_borrowable()
            case _:
                raise ValueError("something went wrong")

    def return_wire(self, wire):
        reg_type = self._loaned.pop(wire)
        if reg_type == RegisterTypes.REQUIRES_RESET:
            self._registers[ZEROED].append(wire)
            return [measure(wire, reset=True)]
        self._registers[reg_type].append(wire)
        return []

    def return_all(self):
        return list(chain(self.return_wire(w) for w in self._loaned))


def null_postprocessing(results):
    return results[0]


@lru_cache
def _get_plxpr_resolve_dynamic_wires():  # pylint: disable=missing-docstring
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture.base_interpreter import PlxprInterpreter
    except ImportError:  # pragma: no cover
        return None

    class ResolveDynamicWires(PlxprInterpreter):

        def __init__(self, manager) -> None:
            """Initialize the interpreter."""
            super().__init__()
            self.manager = manager

    @ResolveDynamicWires.register_primitive(_get_allocate_prim())
    def _(self, num_wires, **kwargs):
        return [self.manager.get_wire(**kwargs) for _ in range(num_wires)]

    @ResolveDynamicWires.register_primitive(_get_deallocate_prim())
    def _(self, *wires):
        _ = [self.manager.return_wire(w) for w in wires]
        return []

    def resolve_dynamic_wires_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args):
        """Function for applying the ``map_wires`` transform on plxpr."""

        manager = WireManager(**tkwargs)

        interpreter = ResolveDynamicWires(manager)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return resolve_dynamic_wires_plxpr_to_plxpr


resolve_dynamic_wires_plxpr_to_plxpr = _get_plxpr_resolve_dynamic_wires()


@partial(transform, plxpr_transform=resolve_dynamic_wires_plxpr_to_plxpr)
def resolve_dynamic_wires(tape, zeroed=(), burnable=(), borrowable=(), garbage=()):

    manager = WireManager(zeroed=zeroed, burnable=burnable, borrowable=borrowable, garbage=garbage)

    wire_map = {}

    new_ops = []
    for op in tape.operations:
        if isinstance(op, Allocate):
            for w in op.wires:
                ops, wire = manager.get_wire(**op.hyperparameters)
                new_ops += ops
                wire_map[w] = wire
        elif isinstance(op, Deallocate):
            for w in op.wires:
                new_ops += manager.return_wire(wire_map.pop(w))
        elif isinstance(op, DeallocateAll):
            new_ops += manager.return_all()
        else:
            new_ops.append(op.map_wires(wire_map))

    mps = [mp.map_wires(wire_map) for mp in tape.measurements]
    return (tape.copy(ops=new_ops, measurements=mps),), null_postprocessing


@transform
def device_resolve_dynamic_wires(tape, device_wires):
    if not device_wires:
        raise NotImplementedError("need wires for now")
    burnable = set(device_wires) - set(tape.wires)
    return resolve_dynamic_wires(tape, burnable=burnable)
