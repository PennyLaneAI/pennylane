from typing import List, Tuple, Union

from ..device_interface import *
from .python_simulator import *
from .preprocessor import simple_preprocessor
from pennylane.gradients import param_shift
from pennylane import adjoint
from pennylane.operation import operation_derivative


class TestDevicePythonSim(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    short_name = "test_py_dev"
    name = "TestDevicePythonSim PennyLane plugin"
    pennylane_requires = 0.1
    version = 0.1
    author = "Xanadu Inc."

    def __init__(self, dev_config: Union[DeviceConfig, None] = None, *args, **kwargs):
        super().__init__(dev_config, *args, **kwargs)
        self._private_sim = PlainNumpySimulator()

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        if isinstance(qscript, QuantumScript):
            print(qscript.circuit)
            res = self._private_sim.execute(qscript)
            res = res[0] if len(res) == 1 else res
            return np.array(res)
        return [self.execute(qs) for qs in qscript]

    def capabilities(self) -> DeviceConfig:
        return self.dev_config if hasattr(self, "dev_config") else {}

    def preprocess(
        self, qscript: Union[QuantumScript, List[QuantumScript]]
    ) -> Tuple[List[QuantumScript], Callable]:
        return simple_preprocessor(qscript)

    def execute_and_gradients(self, qscripts, *args, **kwargs):
        """Defined for temporary compatability."""
        res = self.execute(qscripts)
        grads = self.gradient(qscripts[0])
        grads = tuple(np.array(x) for x in grads)

        return res, [grads]

    def gradient(self, qscript: QuantumScript, order: int = 1):
        if order != 1:
            raise NotImplementedError

        state = self._private_sim.create_zeroes_state(2)
        for op in qscript.operations:
            state = self._private_sim.apply_operation(state, op)
        bra = self._private_sim.apply_operation(state, qscript.measurements[0].obs)
        ket = state

        grads = []
        for op in reversed(qscript.operations):
            adj_op = adjoint(op)
            ket = self._private_sim.apply_operation(ket, adj_op)

            if op.num_params != 0:
                dU = operation_derivative(op)
                ket_temp = self._private_sim.apply_matrix(ket, dU, op.wires)
                dM = 2 * np.real(np.vdot(bra, ket_temp))
                grads.append(dM)

            bra = self._private_sim.apply_operation(bra, adj_op)

        grads = grads[::-1]
        return grads
