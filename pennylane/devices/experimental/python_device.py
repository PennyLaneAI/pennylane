# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file defines a prototype python device conforming to the new device interface.
"""
from typing import Sequence, Tuple, Union, Callable
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pennylane.tape import QuantumScript
from pennylane.workflow import ExecutionConfig

from .device_interface import AbstractDevice
from .simulator import PlainNumpySimulator, adjoint_diff_gradient
from .python_preprocessor import simple_preprocessor


def _multiprocessing_single_execution(
    task_i: int, single_qs: QuantumScript, return_dict: "multiprocessing.managers.DictProxy"
):
    """Perform a single execution. Wrapped for use with the ``multiprocessing`` module.

    Args:
        task_i (int): an integer corresponding to the place of the single qauntum script in a batch.
        single_qs (QuantumScript): A single quantum script to execute
        return_dict (multiprocessing.managers.DictProxy): A dictionary to store the result in.
    """
    simulator = PlainNumpySimulator()
    return_dict[task_i] = simulator.execute(single_qs)


def _single_execution(single_qs: QuantumScript):
    """Perform a single execution on an instantiated simulator.

    Args:
        single_qs (QuantumScript): A single quantum script to execute
    """
    simulator = PlainNumpySimulator()
    return simulator.execute(single_qs)


class PythonDevice(AbstractDevice):
    """Device containing a Python simulator favouring composition-like interface.

    Keyword Args:
        use_multiprocessing=False (Bool): whether or not to perform the execution of batches with ``multiprocessing``.

    """

    def __init__(self, use_multiprocessing=False, use_mpi=False, use_ray=False):
        self.use_multiprocessing = use_multiprocessing
        self.use_mpi = use_mpi
        self.use_ray = use_ray
        super().__init__()

    def execute(
        self, qscripts: Union[QuantumScript, Sequence[QuantumScript]], execution_config=None
    ):
        if isinstance(qscripts, QuantumScript):
            simulator = PlainNumpySimulator()
            results = simulator.execute(qscripts)

            if self.tracker.active:
                self.tracker.update(executions=1, results=results)
                self.tracker.record()

            return results

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(qscripts))
            self.tracker.record()

        if self.use_ray:
            return self._ray_execution(qscripts)
        if self.use_mpi:
            return self._mpi_execution(qscripts)
        if not self.use_multiprocessing:
            return tuple(self.execute(qs) for qs in qscripts)
        return self._multiprocess_execution(qscripts)

    @staticmethod
    def _multiprocess_execution(qscript: Sequence[QuantumScript]) -> Tuple:
        """Perform the execution of a quantum script batch with multiple processes."""
        results = None
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            if executor is not None:
                results = executor.map(_single_execution, qscript)
        return tuple(results)

    @staticmethod
    def _mpi_execution(qscript: Sequence[QuantumScript]) -> Tuple:
        """Perform the execution of a quantum script batch with multiple processes."""
        from mpi4py.futures import MPIPoolExecutor

        results = None
        with MPIPoolExecutor() as executor:
            if executor is not None:
                results = executor.map(_single_execution, qscript)
        return tuple(results)

    @staticmethod
    def _ray_execution(qscript: Sequence[QuantumScript]) -> Tuple:
        """Perform the execution of a quantum script batch with multiple processes usin Ray"""
        import ray

        @ray.remote
        def fn(qscript: QuantumScript):
            return _single_execution(qscript)

        futures = [fn.remote(s) for s in qscript]
        return tuple(ray.get(futures))

    def preprocess(
        self, qscript: Union[QuantumScript, Sequence[QuantumScript]], execution_config=None
    ) -> Tuple[Sequence[QuantumScript], Callable]:
        def identity_post_processing(res):
            """Identity post-processing function created by PythonDevice preprocessing."""
            return res

        return simple_preprocessor(qscript), identity_post_processing

    def gradient(self, qscript: QuantumScript, execution_config=None):
        execution_config = ExecutionConfig()
        if execution_config.order != 1 and execution_config.shots is None:
            raise NotImplementedError

        if self.tracker.active:
            self.tracker.update(gradients=1)
            self.tracker.record()

        if isinstance(qscript, QuantumScript):
            simulator = PlainNumpySimulator()
            return adjoint_diff_gradient(qscript, simulator)

        return tuple(self.gradient(qs, execution_config) for qs in qscript)

    @classmethod
    def supports_gradient_with_configuration(cls, execution_config) -> bool:
        """Determine whether or not a gradient is available with a given execution configuration.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.

        """
        return execution_config.order == 1 and execution_config.shots is None
