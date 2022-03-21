from pennylane.tape.tape import QuantumTape


class FunctionTape(QuantumTape):

    def __init__(self, param_names, *args, **kwargs):
        self.param_names = param_names
        self.values = None
        self.vars = {param_names[i]:(lambda: self.values[i]) for i in range(len(param_names))}
        super().__init__(*args, **kwargs)

    def __call__(self, *args):
        self.values = args

class IfTape(QuantumTape):

    def __init__(self, expr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr

class WhileTape(QuantumTape):
    def __init__(self, expr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr