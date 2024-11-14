import pennylane as qml
import pennylane.labs.resource_estimation as re

class ResourceAdjoint(qml.ops.Adjoint, re.ResourceSymbolicOperator):
    """Resource class for Adjoint"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        pass

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        pass

class ResourceControlled(qml.ops.Controlled, re.ResourceOperator):
    """Resource class for Controlled"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        pass

    def resource_params(self):
        pass

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        pass

class ResourcePow(qml.ops.Pow, re.ResourceOperator):
    """Resource class for Pow"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        pass

    def resource_params(self):
        pass

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        pass
