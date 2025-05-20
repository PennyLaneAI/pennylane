from collections import deque
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
    Allocate(wires, require_zeros=require_zeros, reset_to_original=reset_to_original)
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

    def __init__(self, zeroed=(), dirty=(), min_integer=None, style="stack"):
        self.style = style
        reg_class = deque if style == "queue" else list
        self._zeroed = reg_class(zeroed)
        self._dirty = reg_class(dirty)
        self._loaned = {}  # wire to final register type
        self.min_integer = min_integer

    def get_wire(self, require_zeros, reset_to_original):
        if not self._zeroed and not self._dirty:
            if self.min_integer is None:
                raise ValueError("no wires left to allocate.")
            self.min_integer += 1
            self._zeroed.append(self.min_integer)
        if require_zeros:
            if self._zeroed:
                w = self._zeroed.popleft() if self.style == "queue" else self._zeroed.pop()
                self._loaned[w] = "zeroed" if reset_to_original else "dirty"
                return w, []
            w = self._dirty.popleft() if self.style == "queue" else self._dirty.pop()
            self._loaned[w] = "dirty"
            m = measure(w, reset=True)
            return w, m.measurements

        if self._dirty:
            w = self._dirty.popleft() if self.style == "queue" else self._dirty.pop()
            self._loaned[w] = "dirty"
            return w, []
        w = self._zeroed.popleft() if self.style == "queue" else self._zeroed.pop()
        self._loaned[w] = "zeroed" if reset_to_original else "dirty"
        return w, []

    def return_wire(self, wire):
        reg_type = self._loaned.pop(wire)
        if reg_type == "dirty":
            self._dirty.append(wire)
        else:
            self._zeroed.append(wire)

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
        return [self.manager.get_wire(**kwargs)[1] for _ in range(num_wires)]

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
def resolve_dynamic_wires(tape, zeroed=(), dirty=(), min_integer=None, style="stack"):

    manager = WireManager(zeroed=zeroed, dirty=dirty, min_integer=min_integer, style=style)

    wire_map = {}

    new_ops = []
    for op in tape.operations:
        if isinstance(op, Allocate):
            for w in op.wires:
                wire, ops = manager.get_wire(**op.hyperparameters)
                new_ops += ops
                wire_map[w] = wire
        elif isinstance(op, Deallocate):
            for w in op.wires:
                manager.return_wire(wire_map.pop(w))
        elif isinstance(op, DeallocateAll):
            new_ops += manager.return_all()
        else:
            new_ops.append(op.map_wires(wire_map))

    mps = [mp.map_wires(wire_map) for mp in tape.measurements]
    return (tape.copy(ops=new_ops, measurements=mps),), null_postprocessing


@transform
def device_resolve_dynamic_wires(tape, device_wires):
    if not device_wires:
        max_wire_int = max([0, *(w for w in tape.wires if isinstance(w, int))]) + 1
        return resolve_dynamic_wires(tape, min_integer=max_wire_int)
    zeroed = set(device_wires) - set(tape.wires)
    return resolve_dynamic_wires(tape, zeroed=zeroed)
