import torch

@torch.jit.script
class TorchScriptDevice:

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = torch.zeros([2] * num_qubits)
        self.state[[0] * num_qubits] = 1

    def run_circuit(self, circuit):
        for op in circuit.ops:
            self.apply(op)

    def apply(self, op):
        pass

device = TorchScriptDevice()
print(device)
