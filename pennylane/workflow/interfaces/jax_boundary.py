import jax

from ..executor import Executor
from ..gradient_layers import DerivativeExecutor


# Role of grad on execution for jax
# jvp function is only called when the gradients are actually requested, so
# we don't need to worry about computing the derivatives when they won't be used.
# if the jvp method i


class JaxLayer(Executor):
    def __init__(
        self,
        next_executor: Executor,
        derivative_executor: DerivativeExecutor,
        grad_on_execution=False,
    ):
        self._next_executor = jax.custom_jvp(next_executor)
        self._derivative_executor = derivative_executor

        def execute_and_jvp_with_append(primals, tangents):
            tangent_variables = tuple(list(t.get_parameters()) for t in tangents[0])
            return derivative_executor.execute_and_compute_jvp(
                primals[0], tangent_variables, reduction_method="append"
            )

        self._next_executor.defjvp(execute_and_jvp_with_append)

    def __call__(self, circuits):
        return self._next_executor(circuits)

    @property
    def next_layer(self):
        return self._next_executor.fun

    @property
    def configuration(self):
        return (self._derivative_executor, self._next_executor.fun)
