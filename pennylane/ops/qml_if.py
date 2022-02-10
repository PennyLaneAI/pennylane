from pennylane.operation import AnyWires, Operation


class If(Operation):
    """Condition operations on measurement results using this """
    num_wires = AnyWires

    def __init__(self, m_val, then_op, else_op=None):
        self.m_val = m_val
        self.then_op = then_op
        self.else_op = else_op
        wires = list({m_val.wire} | set(then_op.wires) | set(getattr(else_op, "wires", [])))
        super().__init__(wires=wires)