import copy
import itertools
import numpy as np

import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import Operation

class PrepSelPrep(Operation):
    def __init__(self, lcu, control, id=None):
        coeffs, ops = lcu.terms()
        control = qml.wires.Wires(control)
        self.hyperparameters["lcu"] = lcu
        self.hyperparameters["coeffs"] = coeffs
        self.hyperparameters["ops"] = tuple(ops)
        self.hyperparameters["control"] = control

        if any(
            control_wire in qml.wires.Wires.all_wires([op.wires for op in ops])
            for control_wire in control
        ):
            raise ValueError("Control wires should be different from operation wires.")


        target_wires = qml.wires.Wires.all_wires([op.wires for op in ops])
        self.hyperparameters["target_wires"] = target_wires


        all_wires = target_wires + control
        super().__init__(*self.data, wires=all_wires, id=id)

    def _flatten(self):
        return (self.lcu), (self.control)

    @classmethod
    def _unflatten(cls, data, metadata) -> "PrepSelPrep":
        print(data)
        return cls(data, metadata)

    def __repr__(self):
        return f"PrepSelPrep(coeffs={tuple(self.coeffs)}, ops={self.ops}, control={self.control})"

    def map_wires(self, wire_map: dict) -> "Select":
        new_ops = [o.map_wires(wire_map) for o in self.hyperparameters["ops"]]
        new_control = [wire_map.get(wire, wire) for wire in self.hyperparameters["control"]]
        new_lcu = qml.dot(self.hyperparameters["coeffs"], new_ops)
        return PrepSelPrep(new_lcu, new_control)

    def decomposition(self):
        return self.compute_decomposition(self.hyperparameters['coeffs'], self.ops, self.control)

    @staticmethod
    def compute_decomposition(coeffs, ops, control):
        interface = qml.math.get_interface(coeffs)
        normalized_coeffs = (np.sqrt(coeffs) / qml.math.norm(np.sqrt(coeffs)))

        with qml.QueuingManager.stop_recording():
            prep_ops = qml.StatePrep.compute_decomposition(normalized_coeffs, control)
            select_ops = qml.Select.compute_decomposition(ops, control)
            adjoint_prep_ops = list(map(qml.adjoint, prep_ops))

        ops = prep_ops + select_ops + adjoint_prep_ops
        return ops

    def __copy__(self):
        """Copy this op"""
        cls = self.__class__
        copied_op = cls.__new__(cls)

        new_data = copy.copy(self.data)

        for attr, value in vars(self).items():
            if attr != "data":
                setattr(copied_op, attr, value)

        copied_op.data = new_data

        return copied_op

    @property
    def data(self):
        """Create data property"""
        return tuple(d for op in self.ops for d in op.data)

    @data.setter
    def data(self, new_data):
        """Set the data property"""
        for op in self.ops:
            op_num_params = op.num_params
            if op_num_params > 0:
                op.data = new_data[:op_num_params]
                new_data = new_data[op_num_params:]

    @property
    def coeffs(self):
        """The coefficients of the LCU."""
        return self.hyperparameters["coeffs"]

    @property
    def ops(self):
        """The operations of the LCU."""
        return self.hyperparameters["ops"]

    @property
    def lcu(self):
        """The LCU to be block-encoded."""
        return self.hyperparameters["lcu"]

    @property
    def control(self):
        """The control wires."""
        return self.hyperparameters["control"]

    @property
    def target_wires(self):
        """The wires of the input operators."""
        return self.hyperparameters["target_wires"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["control"] + self.hyperparameters["target_wires"]
