from pennylane.operation import Operation, AnyWires

class MeasurementValue:
    def __init__(self, wire) -> None:
        self.wire = wire
        self.runtime_value = None

class Measurement(Operation):
    num_wires = 1

    def __init__(self, wire):
        self.wire = wire
        self.m_val = MeasurementValue(wire)
        super().__init__(wires=wire)

def Measure(wire):
    m_op = Measurement(wire)
    return m_op.m_val

class If(Operation):
    num_wires = AnyWires

    def __init__(self, m_val, then_op, else_op=None):
        self.m_val = m_val
        self.then_op = then_op
        self.else_op = else_op
        wires = list({m_val.wire} | set(then_op.wires) | set(getattr(else_op, "wires", [])))
        super().__init__(wires=wires)
