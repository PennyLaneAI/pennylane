import jax

from ..executor import Executor
from ..gradient_layers import DerivativeExecutor


class JaxLayer(Executor):
    def __init__(self, next_executor: Executor, derivative_executor: DerivativeExecutor):
        self._next_executor = jax.custom_jvp(next_executor)

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
        return (self._next_executor.fun, self._next_executor.jvp.__self__)
