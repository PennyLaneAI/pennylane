from typing import List, Tuple

import numpy as np
from pennylane import (
    BasisState,
    Hadamard,
    Projector,
    QubitStateVector,
    Rot,
)
from pennylane.grouping import is_pauli_word
from pennylane.operation import Observable, Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap

# Remove after the next release of PL
# Add from pennylane import matrix
import pennylane as qml

try:
    from .lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
    )
    from .lightning_qubit_ops.adjoint_diff import (
        NamedObsC64,
        NamedObsC128,
        HermitianObsC64,
        HermitianObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
        HamiltonianC64,
        HamiltonianC128,
        OpsStructC64,
        OpsStructC128,
        create_ops_list_C128 as create_ops_list
    )
except ImportError:
    pass


def _obs_has_kernel(ob: Observable) -> bool:
    """Returns True if the input observable has a supported kernel in the C++ backend.

    Args:
        ob (Observable): the input observable

    Returns:
        bool: indicating whether ``obs`` has a dedicated kernel in the backend
    """
    if is_pauli_word(ob):
        return True
    if isinstance(ob, (Hadamard, Projector)):
        return True
    if isinstance(ob, Tensor):
        return all(_obs_has_kernel(o) for o in ob.obs)
    return False


def _serialize_named_hermitian_ob(o, wires_map: dict, use_csingle: bool):
    """Serializes an observable (Named or Hermitian)"""
    assert not isinstance(o, Tensor)

    if use_csingle:
        ctype = np.complex64
        named_obs = NamedObsC64
        hermitian_obs = HermitianObsC64
    else:
        ctype = np.complex128
        named_obs = NamedObsC128
        hermitian_obs = HermitianObsC128

    wires_list = o.wires.tolist()
    wires = [wires_map[w] for w in wires_list]
    if _obs_has_kernel(o):
        return named_obs(o.name, wires)
    return hermitian_obs(qml.matrix(o).ravel().astype(ctype), wires)


def _serialize_tensor_ob(ob, wires_map: dict, use_csingle: bool):
    """Serialize a tensor observable"""
    assert isinstance(ob, Tensor)

    if use_csingle:
        tensor_obs = TensorProdObsC64
    else:
        tensor_obs = TensorProdObsC128

    return tensor_obs([_serialize_ob(o, wires_map, use_csingle) for o in ob.obs])


def _serialize_hamiltonian(ob, wires_map: dict, use_csingle: bool):
    if use_csingle:
        rtype = np.float32
        hamiltonian_obs = HamiltonianC64
    else:
        rtype = np.float64
        hamiltonian_obs = HamiltonianC128

    coeffs = np.array(unwrap(ob.coeffs)).astype(rtype)
    terms = [_serialize_ob(t, wires_map, use_csingle) for t in ob.ops]
    return hamiltonian_obs(coeffs, terms)


def _serialize_ob(ob, wires_map, use_csingle):
    if isinstance(ob, Tensor):
        return _serialize_tensor_ob(ob, wires_map, use_csingle)
    elif ob.name == "Hamiltonian":
        return _serialize_hamiltonian(ob, wires_map, use_csingle)
    else:
        return _serialize_named_hermitian_ob(ob, wires_map, use_csingle)


def _serialize_observables(tape: QuantumTape, wires_map: dict, use_csingle: bool = False) -> List:
    """Serializes the observables of an input tape.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires
        use_csingle (bool): whether to use np.complex64 instead of np.complex128

    Returns:
        list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with the C++ backend
    """

    return [_serialize_ob(ob, wires_map, use_csingle) for ob in tape.observables]


def _serialize_ops(
    tape: QuantumTape, wires_map: dict
) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
    """Serializes the operations of an input tape.

    The state preparation operations are not included.

    Args:
        tape (QuantumTape): the input quantum tape
        wires_map (dict): a dictionary mapping input wires to the device's backend wires

    Returns:
        Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
        of operation names, a list of operation parameters, a list of observable wires, a list of
        inverses, and a list of matrices for the operations that do not have a dedicated kernel.
    """
    names = []
    params = []
    wires = []
    inverses = []
    mats = []

    uses_stateprep = False

    for o in tape.operations:
        if isinstance(o, (BasisState, QubitStateVector)):
            uses_stateprep = True
            continue
        elif isinstance(o, Rot):
            op_list = o.expand().operations
        else:
            op_list = [o]

        for single_op in op_list:
            is_inverse = single_op.inverse

            name = single_op.name if not is_inverse else single_op.name[:-4]
            names.append(name)

            if not hasattr(StateVectorC128, name):
                params.append([])
                mats.append(qml.matrix(single_op))

                if is_inverse:
                    is_inverse = False
            else:
                params.append(single_op.parameters)
                mats.append([])

            wires_list = single_op.wires.tolist()
            wires.append([wires_map[w] for w in wires_list])
            inverses.append(is_inverse)

    return (names, params, wires, inverses, mats), uses_stateprep
