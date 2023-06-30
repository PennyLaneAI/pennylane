import tensorflow as tf

from pennylane import math

from ..executor import Executor
from ..gradient_layers import DerivativeExecutor


class TFLayer(Executor):
    def __init__(
        self,
        next_executor: Executor,
        derivative_executor: DerivativeExecutor,
        grad_on_execution=False,
    ):
        self._next_executor = next_executor
        self._derivative_executor = derivative_executor
        self._grad_on_execution = grad_on_execution

    def __call__(self, circuits):
        parameters = []
        for tape in circuits:
            parameters.extend(tape.get_parameters())

        @tf.custom_gradient
        def tf_registered_function(*inner_params):
            if self._grad_on_execution:
                results, jacs = self._derivative_executor.execute_and_compute_vjp()
            else:
                results = self._next_executor(circuits)

            def grad_fn(*dy, **tfkwargs):
                vjps = self._derivative_executor.compute_vjp(
                    circuits, dy, reduction_method="extend"
                )

                # filter out untrainable parameters if they happen to appear in the vjp
                vjps = [vjp for vjp in vjps if 0 not in math.shape(vjp)]

                variables = tfkwargs.get("variables")
                return (vjps, variables) if variables is not None else vjps

            return results, grad_fn

        return tf_registered_function(*parameters)

    @property
    def next_layer(self):
        return self._next_executor.fun

    @property
    def configuration(self):
        return (self._next_executor, self._derivative_executor)
