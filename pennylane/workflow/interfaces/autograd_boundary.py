import autograd
from autograd.numpy.numpy_boxes import ArrayBox

from pennylane.math import to_numpy

from ..executor import Executor


@autograd.extend.primitive
def autograd_registered_function(
    parameters, tapes=None, next_executor=None, derivative_executor=None
):
    return next_executor(tapes)


def wrapped_vjp(ans, parameters, tapes, next_executor, derivative_executor):
    def vjp(dy):
        vjps = derivative_executor.compute_vjp(tapes, dy, reduction_method="append")
        return [to_numpy(v, max_depth=2) if isinstance(v, ArrayBox) else v for v in vjps]

    return vjp


autograd.extend.defvjp(autograd_registered_function, wrapped_vjp, argnums=[0])


class AutogradLayer(Executor):
    def __init__(self, next_executor, derivative_executor, grad_on_execution=False):
        self._next_executor = next_executor
        self._derivative_executor = derivative_executor
        self._grad_on_execution = grad_on_execution

    def __call__(self, circuits):
        parameters = autograd.builtins.tuple(
            [autograd.builtins.list(t.get_parameters()) for t in circuits]
        )
        return autograd_registered_function(
            parameters, circuits, self._next_executor, self._derivative_executor
        )

    @property
    def next_layer(self):
        return self._next_executor

    @property
    def configuration(self):
        return (self._next_executor, self._derivative_executor)
