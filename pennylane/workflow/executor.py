import abc

from concurrent.futures import ProcessPoolExecutor

from typing import Optional, Callable, Tuple

from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch
from pennylane.transforms.core import TransformProgram

Batch = Tuple[QuantumScript]

PostProcessingFn = Callable[[ResultBatch], ResultBatch]


class Executor(abc.ABC):
    def __hash__(self):
        name = type(self).__name__
        return hash((name, self.configuration))

    def __repr__(self):
        config_str = repr(self.configuration).replace("\n", "\n\t")
        return f"{type(self).__name__}(\n\t{config_str}\n)"

    @abc.abstractmethod
    def __call__(self, circuits: Batch) -> ResultBatch:
        pass

    @property
    def next_layer(self):
        return None

    @property
    def configuration(self):
        """All the information needed to fully reproduce the executor.

        Should be able to reproduce the object by ``type(obj)(*obj.configuration)`.
        """
        return tuple()


class DeviceExecutor(Executor):
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
    def __init__(self, next_executor: Executor, transform_program: TransformProgram):
        self._next_executor = next_executor
        self._transform_program = transform_program

    def __repr__(self):
        prog_str = repr(self._transform_program).replace("\n", "\n\t")
        executor_str = repr(self._next_executor).replace("\n", "\n\t")
        return f"{type(self).__name__}(\n\t{prog_str},\n\t{executor_str}\n)"

    @property
    def configuration(self):
        return (self._next_executor, self._transform_program)

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
    def __init__(self, next_executor, num_processes: int = 1):
        self._next_executor = next_executor
        self._num_processes = num_processes

    def __call__(self, circuits) -> ResultBatch:
        results = None
        with ProcessPoolExecutor(max_workers=self._num_processes) as executor:
            future = executor.map(self._next_executor, [(c,) for c in circuits])
            results = tuple(f[0] for f in future)
        return results

    @property
    def next_layer(self):
        return self._next_executor
