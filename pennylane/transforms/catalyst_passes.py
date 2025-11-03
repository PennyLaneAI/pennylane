"""
Defines some catalyst passes.
"""

from .core import transform


def _make_no_tape_implementation(name):
    # pylint: disable=unused-variable
    def no_tape_implementation(tape, *args, **kwargs):
        raise NotImplementedError(f"{name} only has an MLIR implementation.")

    return no_tape_implementation


disentangle_cnot = transform(
    _make_no_tape_implementation("disentangle_cnot"), pass_name="disentangle-CNOT"
)
disentangle_swap = transform(
    _make_no_tape_implementation("disentangle_swap"), pass_name="disentangle-SWAP"
)
ions_decomposition = transform(
    _make_no_tape_implementation("ions_decomposition"), pass_name="ions-decomposition"
)
to_ppr = transform(_make_no_tape_implementation("to_ppr"), pass_name="to-ppr")
commute_ppr = transform(_make_no_tape_implementation("commute_ppr"), pass_name="commute_ppr")
