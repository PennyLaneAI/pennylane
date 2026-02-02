from pennylane.core.tape import *


def __getattr__(key):
    if key == "plxpr_to_tape":
        from pennylane.core.tape.plxpr_conversion import plxpr_to_tape

        return plxpr_to_tape
    raise AttributeError(f"module 'pennylane.tape' has no attribute '{key}'")  # pragma: no cover
