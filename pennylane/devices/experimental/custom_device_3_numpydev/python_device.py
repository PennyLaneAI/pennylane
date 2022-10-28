from ..device_interface import *
from .python_simulator import *
from .preprocessor import simple_preprocessor

class TestDevicePythonSim(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    def __init__(self, dev_config: Union[DeviceConfig, None] = None, *args, **kwargs):
        super().__init__(dev_config, *args, **kwargs)
        self._private_sim = PlainNumpySimulator()

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        return self._private_sim.execute(qscript)

    def capabilities(self) -> DeviceConfig:
        return self.dev_config

    def preprocess(self, qscript : Union[QuantumScript, List[QuantumScript]]):
        return simple_preprocessor(qscript)
