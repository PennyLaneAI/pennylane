import pennylane as qml
import pennylane.labs.resource_estimation.resource_constructor as rc

class ResourceSingleExcitation(qml.SingleExcitation, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceSingleExcitationMinus(qml.SingleExcitationMinus, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceSingleExcitationPlus(qml.SingleExcitationPlus, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceDoubleExcitation(qml.DoubleExcitation, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceDoubleExcitationMinus(qml.DoubleExcitationMinus, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceDoubleExcitationPlus(qml.DoubleExcitationPlus, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceOrbitalRotation(qml.OrbitalRotation, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceFermionicSWAP(qml.FermionicSWAP, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return
