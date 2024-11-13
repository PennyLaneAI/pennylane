import pennylane as qml
import pennylane.labs.resource_estimation as re

#pylint: disable=arguments-differ

class ResourceMultiRZ(qml.MultiRZ, re.ResourceOperator):
    """Resource class for MultiRZ"""

    @staticmethod
    def _resource_decomp(num_wires, **kwargs):
        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        rz = re.CompressedResourceOp(re.ResourceRZ, {})

        gate_types = {}
        gate_types[cnot] = 2*(num_wires-1)
        gate_types[rz] = 1

        return gate_types

    def resource_params(self):
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires, **kwargs):
        return re.CompressedResourceOp(cls, {"num_wires": num_wires})

class ResourcePauliRot(qml.PauliRot, re.ResourceOperator):
    """Resource class for PauliRot"""

    @staticmethod
    def _resource_decomp(active_wires, pauli_word, **kwargs):
        if set(pauli_word) == {"I"}:
            gp = re.ResourceGlobalPhase.resource_rep(**kwargs)
            return {gp: 1}

        h = re.ResourceHadamard.resource_rep(**kwargs)
        rx = re.ResourceRX.resource_rep(**kwargs)
        multi_rz = re.ResourceMultiRZ.resource_rep(active_wires, **kwargs)

        h_count = 0
        rx_count = 0

        for gate in pauli_word:
            if gate == "X":
                h_count += 2
            if gate == "Y":
                rx_count += 2

        gate_types = {}
        gate_types[h] = h_count
        gate_types[rx] = rx_count
        gate_types[multi_rz] = 1

        return gate_types

    def resource_params(self):
        pauli_word = self.hyperparameters["pauli_word"]
        return {
            "active_wires": len(pauli_word.replace("I", "")),
            "pauli_word": pauli_word,
        }

    @classmethod
    def resource_rep(cls, active_wires, pauli_word, **kwargs):
        return re.CompressedResourceOp(cls, {"active_wires": active_wires, "pauli_word": pauli_word})

class ResourceIsingXX(qml.IsingXX, re.ResourceOperator):
    """Resource class for IsingXX"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        return

class ResourceIsingYY(qml.IsingYY, re.ResourceOperator):
    """Resource class for IsingYY"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        return

class ResourceIsingZZ(qml.IsingZZ, re.ResourceOperator):
    """Resource class for IsingZZ"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        return

class ResourceIsingXY(qml.IsingXY, re.ResourceOperator):
    """Resource class for IsingXY"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        return

class ResourcePSWAP(qml.PSWAP, re.ResourceOperator):
    """Resource class for PSWAP"""

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        return

    def resource_params(self):
        return

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        return
