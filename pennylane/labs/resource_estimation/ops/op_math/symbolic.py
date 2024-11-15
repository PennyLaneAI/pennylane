import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane.ops.op_math.adjoint import AdjointOperation

#pylint: disable=too-many-ancestors

class ResourceAdjoint(AdjointOperation, re.ResourceSymbolicOperator):
    """Resource class for Adjoint"""

    @staticmethod
    def _resource_decomp(*args, **kwargs, base_class, base_params):
        try:
            return base_class.adjoint_resource_rep()

class ResourceControlled(qml.ops.Controlled, re.ResourceSymbolicOperator):
    """Resource class for Controlled"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        pass

    def resource_params(self):
        pass

class ResourcePow(qml.ops.Pow, re.ResourceSymbolicOperator):
    """Resource class for Pow"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        pass

    def resource_params(self):
        pass
