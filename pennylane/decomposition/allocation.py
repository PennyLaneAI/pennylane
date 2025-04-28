from copy import copy

from pennylane.operation import Operator
from pennylane.transforms.core import transform


class Allocate(Operator):

    def __init__(self, wires, require_zeros=True, reset_to_original=True):
        super().__init__(wires=wires)
        self._hyperparameters = {
            "require_zeros": require_zeros,
            "reset_to_original": reset_to_original,
        }

    @property
    def require_zeros(self):
        return self.hyperparameters["require_zeros"]

    @property
    def reset_to_original(self):
        return self.hyperparameters["reset_to_original"]


class Deallocate(Operator):

    def __init__(self, wires):
        super().__init__(wires=wires)


class DynamicWire:

    def __init__(self):
        pass

    def __repr__(self):
        return f"<DynamicWire>"


class allocate:

    def __init__(self, num_wires, require_zeros=True, reset_to_original=True):
        self.require_zeros = require_zeros
        self.reset_to_original = reset_to_original
        self.wires = [DynamicWire() for i in range(num_wires)]

    def __enter__(self):
        Allocate(
            self.wires, require_zeros=self.require_zeros, reset_to_original=self.reset_to_original
        )
        return self.wires

    def __exit__(self, *_, **__):
        Deallocate(self.wires)


class WireManager:

    def __init__(self, zeroed):
        self._zeroed = zeroed
        self.loaned = {}

    def get_wire(self, require_zeros, reset_to_original):
        if require_zeros and reset_to_original:
            wire = self._zeroed.pop()
            self.loaned[wire] = "zeroed"
            return wire
        else:
            raise NotImplementedError

    def return_wire(self, wire):
        wire_type = self.loaned.pop(wire)
        if wire_type == "zeroed":
            self._zeroed.append(wire)
        else:
            raise NotImplementedError
        return


@transform
def allocate_wires(tape, zeroed):

    manager = WireManager(zeroed)

    wire_map = {}

    new_ops = []
    for op in tape.operations:
        if isinstance(op, Allocate):
            for w in op.wires:
                wire_map[w] = manager.get_wire(**op.hyperparameters)
        elif isinstance(op, Deallocate):
            for w in op.wires:
                manager.return_wire(wire_map.pop(w))
        else:
            new_ops.append(op.map_wires(wire_map))
    return (tape.copy(ops=new_ops),), lambda res: res[0]
