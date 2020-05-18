import pennylane as qml


class MSECost:
    r"""Allows users to combine an ansatz circuit with some target observables and then
    derive a cost function from their expectation values."""

    def __init__(
            self, ansatz, observables, device, interface='autograd', diff_method='best', **kwargs
    ):
        # The default value for the 'measure' parameter is 'expval', but here it is made explicit for clarity
        self.qnodes = qml.map(
            ansatz, observables, device, measure='expval', interface=interface, diff_method=diff_method, **kwargs
        )

    def __call__(self, *args, target=None, **kwargs):
        return (self.qnodes(*args, **kwargs) - target) ** 2
