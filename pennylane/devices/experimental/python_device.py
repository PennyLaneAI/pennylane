from typing import List, Tuple, Union, Callable
import multiprocessing

import pennylane as qml
from pennylane import adjoint, Tracker
from pennylane.gradients import param_shift
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumScript

from .device_interface import AbstractDevice
from .simulators import PlainNumpySimulator, JaxSimulator, adjoint_diff_gradient
from .python_preprocessor import simple_preprocessor


def _multiprocessing_single_execution(task_i, single_qs, return_dict):
    interface = qml.math.get_interface(*single_qs.get_parameters(trainable_only=False))
    simulator = JaxSimulator() if interface == "jax" else PlainNumpySimulator()
    return_dict[task_i] = simulator.execute(single_qs)


class PythonDevice(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    def __init__(self, use_mutliprocessing=False):
        self.use_multiprocessing = use_mutliprocessing

    def _get_simulator(self, qscript: QuantumScript):
        interface = qml.math.get_interface(*qscript.get_parameters(trainable_only=False))
        return JaxSimulator() if interface == "jax" else PlainNumpySimulator()

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]], execution_config=None):
        if isinstance(qscript, QuantumScript):
            simulator = self._get_simulator(qscript)
            results = simulator.execute(qscript)

            if self.tracker.active:
                self.tracker.update(executions=1, results=results)
                self.tracker.record()

            return results

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(qscript))
            self.tracker.record()

        if not self.use_multiprocessing:
            return tuple(self.execute(qs) for qs in qscript)
        return self._multiprocess_execution(qscript)

    def _multiprocess_execution(self, qscript):

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i, single_qs in enumerate(qscript):
            p = multiprocessing.Process(
                target=_multiprocessing_single_execution, args=(i, single_qs, return_dict)
            )
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        return tuple(return_dict[i] for i in range(len(qscript)))

    def preprocess(
        self, qscript: Union[QuantumScript, List[QuantumScript]], execution_config=None
    ) -> Tuple[List[QuantumScript], Callable]:
        def identity_post_processing(res):
            """Identity post-processing function created by PythonDevice preprocessing."""
            return res

        return_script = simple_preprocessor(qscript)
        if isinstance(return_script, QuantumScript):
            return_script = [return_script]

        return return_script, identity_post_processing

    def gradient(self, qscript: QuantumScript, order: int = 1, execution_config=None):
        if order != 1:
            raise NotImplementedError

        if self.tracker.active:
            self.tracker.update(gradients=1)
            self.tracker.record()

        if isinstance(qscript, QuantumScript):
            simulator = self._get_simulator(qscript)
            return adjoint_diff_gradient(qscript, simulator)

        return tuple(self.gradient(qs, execution_config) for qs in qscript)
