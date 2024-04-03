

import jax

from pennylane.queuing import QueuingManager

def create_primitive(cls):
    primitive = jax.core.Primitive(cls.__name__)
    primitive.multiple_results = True

    @primitive.def_abstract_eval
    def _(*args, wires, **kwargs):
        return ()

    @primitive.def_impl
    def _(*args, wires, **kwargs):
        cls(*args, wires=wires, **kwargs)
        return ()
    return primitive