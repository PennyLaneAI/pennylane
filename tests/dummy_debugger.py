# pylint: disable=missing-module-docstring
import pennylane as qml


# pylint: disable=too-few-public-methods
class Debugger:
    """A dummy debugger class"""

    def __init__(self):
        # Create a dummy object to act as the device
        # and add a dummy shots attribute to it
        self.device = type("", (), {})()
        self.device.shots = qml.measurements.Shots(None)

        self.active = True
        self.snapshots = {}
