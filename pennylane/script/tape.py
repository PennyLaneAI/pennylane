from pennylane.tape.tape import QuantumTape


class FunctionTape(QuantumTape):

    def __init__(self, names, *args, **kwargs):
        self.names = names
        self.values = None
        self.vars = {names[i]: (lambda: self.values[i]()) for i in range(len(names))}
        super().__init__(*args, **kwargs)

    def __call__(self, *args):
        self.values = args

class Expr:

    def __init__(self, lambda_expr):
        self.lambda_expr = lambda_expr

    def __call__(self):
        return self.lambda_expr()


class IfTape(QuantumTape):

    def __init__(self, expr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr


class WhileTape(QuantumTape):
    def __init__(self, expr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr