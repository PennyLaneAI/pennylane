from typing import Tuple
import torch

@torch.jit.script
class Op:

    def __init__(self, gate_name: str, wires: Tuple[int] = (0,)):
        self.gate_name = gate_name
        self.wires = wires

    def serialize(self):
        return (self.gate_name, self.wires)

    @classmethod
    def deserialize(cls, serialized):
        return Op(serialized[0], wires=serialized[1])


@torch.jit.script
class Circuit:

    def __init__(self, ops: Tuple[Op]):
        self.ops = ops

    def serialize(self):
        return map(lambda op: op.serialize(), self.ops)




@torch.jit.script
class TorchScriptDevice:

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = torch.zeros(int(2 ** num_qubits), dtype=torch.cfloat)
        self.state[0] = 1
        self.state = self.state.reshape((2,) * num_qubits)

    def run_circuit(self, circuit: Circuit):
        for op in circuit.ops:
            self.apply(op)

    def apply(self, op: Op):
        if op.gate_name == "RZ":
            self.state = self._apply_phase(self.state, op.wires[0], 1j)

    def _apply_phase(self, state, axis: int, phase_shift: complex):
        reordered_state = torch.swapaxes(state, 0, axis)
        sub_reordered_state_0 = reordered_state[0]
        sub_reorderd_state_1 = reordered_state[1]
        sub_reorderd_state_1 = torch.tensor(phase_shift, dtype=torch.cfloat) * sub_reorderd_state_1
        value = torch.stack([sub_reordered_state_0, sub_reorderd_state_1], dim=0)
        torch.swapaxes(reordered_state, 0, axis)
        return value

@torch.jit.script
def apply_circuit_to_device(circuit: Circuit, num_qubits: int):
    device = TorchScriptDevice(num_qubits)
    device.run_circuit(circuit)
    return device.state


# print(apply_circuit_to_device.code)
apply_circuit_to_device.save("ok.pt")
print(apply_circuit_to_device(Circuit(ops=(Op(gate_name="RZ", wires=(0,)),)), 4))