import numpy as np

import pennylane as qml
from pennylane.ops.op_math.decompositions.single_qubit_unitary import _get_xzx_angles

from .operations import RotXZX


mbqc_gate_set = {qml.CNOT, qml.H, qml.S, qml.RZ, RotXZX, qml.X, qml.Y, qml.Z, qml.I, qml.GlobalPhase}


@qml.register_resources({RotXZX: 1, qml.GlobalPhase: 1})
def _rot_to_xzx(phi, theta, omega, wires, **__):
    mat = qml.Rot.compute_matrix(phi, theta, omega)
    phi, theta, lam, gamma = _get_xzx_angles(mat)

    RotXZX(lam % (2 * np.pi), theta % (2 * np.pi), phi % (2 * np.pi), wires)
    qml.GlobalPhase(-gamma)


@qml.transform
def convert_to_mbqc_gateset(tape):
    tapes, fn = qml.transforms.decompose(tape,
                                         gate_set=mbqc_gate_set,
                                         alt_decomps={qml.Rot: [_rot_to_xzx]}
                                         )
    return tapes, fn
