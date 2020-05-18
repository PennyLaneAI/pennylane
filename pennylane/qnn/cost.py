"""
This submodule contains frequently used cost functions.
"""
# pylint: disable=too-many-arguments, too-few-public-methods
import pennylane as qml


class MSECost:
    r"""Allows users to combine an ansatz circuit with some target observables and then
    derive a cost function from their expectation values."""

    def __init__(
        self,
        ansatz,
        observables,
        device,
        measure="expval",
        interface="autograd",
        diff_method="best",
        **kwargs
    ):
        self.qnodes = qml.map(
            ansatz,
            observables,
            device,
            measure=measure,
            interface=interface,
            diff_method=diff_method,
            **kwargs
        )

    def __call__(self, *args, target=None, **kwargs):
        return (self.qnodes(*args, **kwargs) - target) ** 2
