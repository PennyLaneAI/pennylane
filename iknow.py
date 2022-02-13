from typing import Tuple, List
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

@torch.jit.script
class Op:

    def __init__(self, gate_name: str, wires: Tuple[int] = (0,)):
        self.gate_name = gate_name
        self.wires = wires

    def serialize(self):
        return self.gate_name, self.wires

    @classmethod
    def deserialize(cls, serialized: Tuple[str, Tuple[int]]):
        return Op(serialized[0], wires=serialized[1])


@torch.jit.script
class Circuit:

    def __init__(self):
        self.ops: List[Op] = []

    def add_op(self, op: Op):
        self.ops.append(op)


def serialize_circuit(self):
    return [op.serialize for op in self.ops]




@torch.jit.script
class TorchScriptDevice:

    def __init__(self, num_qubits: torch.Tensor):
        self.num_qubits = num_qubits
        self.state = torch.zeros((2 ** num_qubits).type(dtype=torch.int), dtype=torch.cfloat)
        self.state[0] = 1
        self.state = self.state.reshape((2,) * num_qubits)

    def run_circuit(self, circuit: Circuit):
        for op in circuit.ops:
            self.apply(op)

    def apply(self, op: Op):
        if op.gate_name == "RZ":
            self.state = self._apply_phase(self.state, op.wires[0], torch.tensor(1j, dtype=torch.cfloat))

    def _apply_phase(self, state, axis: int, phase_shift: torch.Tensor):
        reordered_state = torch.swapaxes(state, 0, axis)
        sub_reordered_state_0 = reordered_state[0]
        sub_reorderd_state_1 = reordered_state[1]
        sub_reorderd_state_1 = phase_shift * sub_reorderd_state_1
        value = torch.stack([sub_reordered_state_0, sub_reorderd_state_1], dim=0)
        torch.swapaxes(reordered_state, 0, axis)
        return value

@torch.jit.script
def apply_circuit_to_device(num_qubits):
    circuit = Circuit()
    circuit.add_op(Op(gate_name="RZ", wires=(0,)))
    device = TorchScriptDevice(num_qubits)
    device.run_circuit(circuit)
    return device.state

class MyForward(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, num_qubits):
        circuit = Circuit()
        circuit.add_op(Op(gate_name="RZ", wires=(0,)))
        device = TorchScriptDevice(num_qubits)
        device.run_circuit(circuit)
        return device.state

writer.add_graph(MyForward(), [torch.tensor(4)])
writer.close()

# print(apply_circuit_to_device.code)
apply_circuit_to_device.save("ok.pt")
print(apply_circuit_to_device(torch.tensor(4, dtype=torch.int)))
torch.onnx.export(MyForward(), (torch.tensor(4),), "out.onnx", input_names=["num_qubits"], output_names=["state"])