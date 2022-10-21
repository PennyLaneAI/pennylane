from ..device_interface import *
from .python_device_driver import *

class TestDevicePythonSim(AbstractDevice):
    "Device containing a Python simulator favouring composition-like interface"

    def __init__(self, dev_config: Union[DeviceConfig, None] = None, *args, **kwargs):
        super().__init__(dev_config, *args, **kwargs)
        self._private_sim = PlainNumpySimulator()
    
    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        return self._private_sim.execute(qscript)