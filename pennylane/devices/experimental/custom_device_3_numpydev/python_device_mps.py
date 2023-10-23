from typing import List, Tuple, Union

import pennylane as qml
from pennylane import adjoint, Tracker
from pennylane.gradients import param_shift

from pennylane.operation import operation_derivative

from ..device_interface import *
from .python_mps import NumpyMPSSimulator
from .preprocessor import simple_preprocessor


class TestDeviceMPSSim(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    tracker = None

    def __init__(self, dev_config: Union[DeviceConfig, None] = None, *args, **kwargs):
        super().__init__(dev_config, *args, **kwargs)
        self.tracker = Tracker()

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]], execution_config=None, chi_max=100, eps=1e-15):
        if isinstance(qscript, QuantumScript):
            interface = qml.math.get_interface(*qscript.get_parameters(trainable_only=False))
            simulator = NumpyMPSSimulator()
            results = simulator.execute(qscript, chi_max=chi_max, eps=eps)

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

    def gradient(self, qscript: QuantumScript, order: int = 1, chi_max=20, eps=1e-10):
        if order != 1:
            raise NotImplementedError

        if self.tracker.active:
            self.tracker.update(gradients=1)
            self.tracker.record()

        sim = NumpyMPSSimulator()
        state = sim.init_MPS(qscript.num_wires)
        for op in qscript.operations:
            state = sim.apply_operation(state, op, chi_max=chi_max, eps=eps)
        bra = sim.apply_operation(state, qscript.measurements[0].obs, chi_max=chi_max, eps=eps)
        ket = state

        grads = []
        for op in reversed(qscript.operations):
            adj_op = adjoint(op)
            ket = sim.apply_operation(ket, adj_op, chi_max=chi_max, eps=eps)

            if op.num_params != 0:
                # The factor 2 is a hacky solution for the generators being
                # non-unitary due to their 0.5 factor
                dU = 2 * operation_derivative(op)
                dU = qml.QubitUnitary(dU, wires=op.wires)
                ket_temp = sim.apply_operation(ket, dU, chi_max=chi_max, eps=eps)
                dM = np.real(overlap(bra, ket_temp, is_adjoint=True)) # missing factor 2 accordingly
                grads.append(dM)

            bra = sim.apply_operation(bra, adj_op, chi_max=chi_max, eps=eps)

        grads = grads[::-1]
        return grads

def overlap(state1, state2, is_adjoint=False):
    """
    Compute overlap <bra|ket> of two MPS

    state1: first mps
    state2: second mps
    is_adjoint: whether or not the second state is already adjoint
    """
    # TODO make this more clever, 
    # i.e. vectorize first step and 
    # in the second contract from both ends.

    B1 = state1.Bs
    B2 = state2.Bs

    assert len(B1) == len(B2)

    if not is_adjoint:
        B2 = [_.conj() for _ in B2]

    # contract all physical indices
    res = []
    for i in range(len(B1)):
        res.append(np.tensordot(B1[i], B2[i], ([1], [1]))) # vL [d] vR - vL* [d*] vR*

    if len(B1) == 1:
        return qml.math.squeeze(res)

    # contract from left to right
    result = np.tensordot(res[0], res[1], ([1, 3], [0, 2])) # vL [vR] vL* {vR*} - [vL] vR {vL*} vR*
    for i in range(2, len(res)):
        result = np.tensordot(result, res[i], ([2, 3], [0, 2])) # vL vL* [vR] {vR*} - [vL] vR {vL*} vR*

    return qml.math.squeeze(result)