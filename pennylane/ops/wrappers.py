import pennylane as qml


class OpWrapper(qml.operation.Operator):
    num_wires = qml.operation.AnyWires

    def __init__(self, base=None, do_queue=True, id=None):
        self.base = base
        self.hyperparameters['base'] = base
        self.hyperparameters.update(base.hyperparameters)
        super().__init__(*base.parameters, wires=base.wires, do_queue=do_queue, id=id)
        self._name = f"{self.__class__.__name__}({self.base.name})"

    @property
    def num_wires(self):
        return self.base.num_wires

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def wires(self):
        return self.base.wires


class Sum(qml.operation.Operator):

    def __init__(self, left, right, do_queue=True, id=None):
        self.left = left
        self.right = right
        self.hyperparameters['left'] = left
        self.hyperparameters['right'] = right

        combined_wires = qml.wires.Wires.shared_wires([left.wires, right.wires])
        super().__init__(*left.parameters, *right.parameters, wires=combined_wires, do_queue=do_queue, id=id)
        self._name = f"{self.right.name} + {self.left.name}"

    @property
    def num_wires(self):
        return len(self.wires)

    @classmethod
    def compute_terms(cls, *params, **hyperparams):
        return [1., 1.], [hyperparams["left"], hyperparams["right"]]


class ScalarMul(qml.operation.Operator):

    def __init__(self, op, scalar, do_queue=True, id=None):
        self.hyperparameters['scalar'] = scalar
        self.hyperparameters['op'] = op

        super().__init__(*op.parameters, scalar, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"{scalar}  {op.name}"

    @property
    def num_wires(self):
        return len(self.wires)

    @classmethod
    def compute_terms(cls, *params, **hyperparams):
        return [hyperparams["scalar"]], [hyperparams["op"]]
