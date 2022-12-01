from typing import List, Tuple, Union

import pennylane as qml
from pennylane import adjoint, Tracker
from pennylane.gradients import param_shift

from pennylane.operation import operation_derivative

from ..device_interface import *
from .python_simulator import PlainNumpySimulator
from .jax_simulator import JaxSimulator
from .preprocessor import simple_preprocessor


class TestDevicePythonSim(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    tracker = None

    def __init__(self, dev_config: Union[DeviceConfig, None] = None, *args, **kwargs):
        super().__init__(dev_config, *args, **kwargs)
        self.tracker = Tracker()

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]], execution_config=None):
        if isinstance(qscript, QuantumScript):
            interface = qml.math.get_interface(*qscript.get_parameters(trainable_only=False))
            simulator = JaxSimulator() if interface == "jax" else PlainNumpySimulator()
            results = simulator.execute(qscript)

            if self.tracker.active:
                self.tracker.update(executions=1, results=results)
                self.tracker.record()

            return results

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(qscript))
            self.tracker.record()

        return [self.execute(qs) for qs in qscript]

    def capabilities(self) -> DeviceConfig:
        return self.dev_config if hasattr(self, "dev_config") else {}

    def preprocess(
        self, qscript: Union[QuantumScript, List[QuantumScript]], execution_config=None
    ) -> Tuple[List[QuantumScript], Callable]:
        return_script = simple_preprocessor(qscript)
        if isinstance(return_script, QuantumScript):
            return_script = [return_script]
        return return_script, lambda res: res

    def execute_and_gradients(self, qscripts, *args, **kwargs):
        """Defined for temporary compatability."""
        res = self.execute(qscripts)
        grads = self.gradient(qscripts[0])
        grads = tuple(np.array(x) for x in grads)

        return res, [grads]

    def gradient(self, qscript: QuantumScript, order: int = 1):
        if order != 1:
            raise NotImplementedError

        if self.tracker.active:
            self.tracker.update(gradients=1)
            self.tracker.record()

        sim = PlainNumpySimulator()
        state = sim.create_zeroes_state(qscript.num_wires)
        for op in qscript.operations:
            state = sim.apply_operation(state, op)
        bra = sim.apply_operation(state, qscript.measurements[0].obs)
        ket = state

        grads = []
        for op in reversed(qscript.operations):
            adj_op = adjoint(op)
            ket = sim.apply_operation(ket, adj_op)

            if op.num_params != 0:
                dU = operation_derivative(op)
                ket_temp = sim.apply_matrix(ket, dU, op.wires)
                dM = 2 * np.real(np.vdot(bra, ket_temp))
                grads.append(dM)

            bra = sim.apply_operation(bra, adj_op)

        grads = grads[::-1]
        return grads
