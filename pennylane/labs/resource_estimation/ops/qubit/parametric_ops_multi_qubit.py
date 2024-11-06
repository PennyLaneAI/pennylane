import pennylane as qml
import pennylane.labs.resource_estimation.resource_constructor as rc

class ResourceMultiRZ(qml.MultiRZ, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourcePauliRot(qml.PauliRot, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceIsingXX(qml.IsingXX, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceIsingYY(qml.IsingYY, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceIsingZZ(qml.IsingZZ, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class ResourceIsingXY(qml.IsingXY, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return

class PSWAP(qml.PSWAP, rc.ResourceConstructor):
    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, **kwargs):
        return
