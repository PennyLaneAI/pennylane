from ..device_interface import *


class TestDeviceBare(AbstractDevice):
    "Naive derived class with no provided methods."
    pass


@TestDeviceBare.register_execute()
def my_exec_function(self, qscript: QuantumScript):
    return f"Hello from execute with arg: {qscript}"


@TestDeviceBare.register_gradient(order=1)
def my_gradient_function(self, qscript: QuantumScript):
    return f"Hello from gradient with arg: {qscript}"
