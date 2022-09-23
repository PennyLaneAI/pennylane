from .device_interface import *

class TestDeviceDefault(AbstractDevice):
    "Class with bare device methods"

    def gradient(self, qscript: qml.tape.QuantumScript, order=1):
        return f"Hello from gradient order {order} with arg: {qscript}"
    
    def vjp(self, qscript: QuantumScript):
        return f"Hello from VJP with arg: {qscript}"

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        return f"Hello from execute with arg: {qscript}"

