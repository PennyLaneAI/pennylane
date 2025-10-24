from functools import partial

from .core import transform


@partial(transform, pass_name="disentangle-CNOT")
def disentangle_cnot(tape):
    raise NotImplementedError("only has an MLIR implementation.")


@partial(transform, pass_name="disentangle-SWAP")
def disentangle_swap(tape):
    raise NotImplementedError("only has an MLIR implementation")


@partial(transform, pass_name="ions-decomposition")
def ions_decomposition(tape):
    raise NotImplementedError("only has an MLIR implementation")


@partial(transform, pass_name="to-ppr")
def to_ppr(tape):
    raise NotImplementedError("only has an MLIR implementation")


@partial(transform, pass_name="commute_ppr")
def commute_ppr(tape, max_pauli_size=0):
    raise NotImplementedError("only has an MLIR implementation")
