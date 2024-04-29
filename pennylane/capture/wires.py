import jax

from pennylane.wires import Wires


wires_p = jax.core.Primitive("wires")


@wires_p.def_impl
def _(*wires):
    int_wires = tuple(int(w) for w in wires)
    return Wires(int_wires)


class AbstractWires(jax.core.ShapedArray):
    def str_short(self, short_dtypes=False):
        return f"Wires[{self.n_wires}]"

    def __repr__(self):
        return f"Wires[{self.n_wires}]"

    def __str__(self):
        return f"Wires[{self.n_wires}]"

    def __init__(self, n_wires):
        self.n_wires = n_wires
        super().__init__((n_wires,), jax.numpy.int32)


jax.core.raise_to_shaped_mappings[AbstractWires] = lambda aval, weak_type: AbstractWires(
    aval.shape[0]
)


@wires_p.def_abstract_eval
def _(*wires):
    return AbstractWires(len(wires))


def wires(wires):
    iterable_wires_types = (list, tuple, Wires, range, set)
    wires = wires if isinstance(wires, iterable_wires_types) else (wires,)
    return wires_p.bind(*wires)
