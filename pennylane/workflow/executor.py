import abc

from concurrent.futures import ProcessPoolExecutor

from typing import Optional, Callable, Tuple

from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch
from pennylane.transforms.core import TransformProgram

Batch = Tuple[QuantumScript]

PostProcessingFn = Callable[[ResultBatch], ResultBatch]


class Executor(abc.ABC):
    """A callable class capable of turning quantum tapes into results."""

    def __repr__(self):
        config_str = repr(self.configuration).replace("\n", "\n\t")
        return f"{type(self).__name__}(\n\t{config_str}\n)"

    @abc.abstractmethod
    def __call__(self, circuits: Batch) -> ResultBatch:
        pass

    @property
    def next_layer(self) -> Optional["Executor"]:
        """
        Points to the next executor if the instance is nested.
        """
        return None


class DeviceExecutor(Executor):
    """Device execution with a bound configuration.

    Args:
        execution_config (ExecutionConfig): the configuration for the device execution
        Device (Device): the device to perform the execution on.


    """

    def __repr__(self):
        return f"DeviceExecutor({self._device})"

    def __init__(self, execution_config, device):
        self._execution_config = execution_config
        self._device = device

    def __call__(self, circuits: Batch) -> ResultBatch:
        return self._device.execute(circuits, self._execution_config)

    @property
    def configuration(self):
        return (self._execution_config, self._device)


class TransformProgramLayer(Executor):
    """Applies a transform program's pre and post processing around the next stage in the pipeline.

    Args:
        next_executor (Executor):
        transform_program (TransformProgram)


    """

    def __init__(self, next_executor: Executor, transform_program: TransformProgram):
        self._next_executor = next_executor
        self._transform_program = transform_program

    def __repr__(self):
        prog_str = repr(self._transform_program).replace("\n", "\n\t")
        executor_str = repr(self._next_executor).replace("\n", "\n\t")
        return f"{type(self).__name__}(\n\t{prog_str},\n\t{executor_str}\n)"

    @property
    def configuration(self):
        return (self._transform_program, self._next_executor)

    @property
    def next_layer(self):
        return self._next_executor

    def __call__(self, circuits: Batch) -> ResultBatch:
        new_circuits, post_processing_fn = self._transform_program(circuits)
        if len(new_circuits) == 0:  # for example with caching
            return post_processing_fn(tuple())
        new_circuits = tuple(new_circuits)  # transforms might return lists accidentally
        results = self._next_executor(new_circuits)
        return post_processing_fn(results)


class MultiProcessingLayer(Executor):
    """Uses ``concurrent.futures.ProcessPoolExecutor`` to dispatch execution over multiple processes.

    Args:
        next_executor (Executor): Where to perform the executions
        max_workers (int): how many workers to use.


    """

    def __init__(self, next_executor, max_workers: int = 1):
        self._next_executor = next_executor
        self._max_workers = max_workers

    def __call__(self, circuits) -> ResultBatch:
        results = None
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            future = executor.map(self._next_executor, [(c,) for c in circuits])
            results = tuple(f[0] for f in future)
        return results

    @property
    def next_layer(self):
        return self._next_executor
