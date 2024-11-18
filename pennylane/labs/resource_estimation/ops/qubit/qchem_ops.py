import pennylane as qml
from pennylane.labs.resource_estimation import ResourceOperator

class ResourceSingleExcitation(qml.SingleExcitation, ResourceOperator):
    """Resource Operator for Single Excitation"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceSingleExcitationMinus(qml.SingleExcitationMinus, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceSingleExcitationPlus(qml.SingleExcitationPlus, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceDoubleExcitation(qml.DoubleExcitation, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceDoubleExcitationMinus(qml.DoubleExcitationMinus, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceDoubleExcitationPlus(qml.DoubleExcitationPlus, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceOrbitalRotation(qml.OrbitalRotation, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceFermionicSWAP(qml.FermionicSWAP, ResourceOperator):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return
