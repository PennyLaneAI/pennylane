from functools import wraps
from pennylane.tape.tapes import QuantumTape


def adjoint(fn):
    """ "Create a new method that applies the adjoint tape of `fn`.

    Example:
            `adjoint(fn)(args)` will apply all of the operations executed during fn(args),
            but in reverse and with each operation adjointed.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with QuantumTape(embed=False) as tape:
            fn(*args, **kwargs)
        for op in reversed(tape.queue):
            op.adjoint(do_queue=True)

    return wrapper
