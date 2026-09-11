r"""
PPM / PBC cost of one step of the THC-qubitized quantum walk (https://arxiv.org/abs/2011.03494).

Needs ``application_algos`` on the path, a Catalyst with the ``to-ppr`` / ``ppr-to-ppm`` passes and
the ``qubitization_thc`` branch of PennyLane (tested on catalyst 0.16.0-dev94, pennylane
0.46.0-dev97). The THC factors are random stand-ins of the right shape: the cost depends only on
the register sizes (M, N, aleph, beth), not on the numerical values.
"""

import time
from collections import Counter

import numpy as np
import pennylane as qp

from application_algos.compilation import CLIFFORD, default_qjit_pipelines
from pennylane.labs.templates import qubitization_thc, qubitization_thc_wires
from pennylane.ops.op_math.controlled_ops import _toffoli_elbow_resources

qp.decomposition.enable_graph()


# +---------------------------------+
# |---- Algorithm hyper params -----|
# +---------------------------------+

M = 12  # THC rank
N = 8  # number of spin orbitals
aleph = 6  # alias-sampling keep-probability bits (PREPARE)
beth = 6  # bits per Givens angle (SELECT)
epsilon = 1e-3  # rotation synthesis precision
precision = int(qp.math.ceil_log2(2 * np.pi / epsilon))


# +---------------------------------+
# |---- Load THC factors -----------|
# +---------------------------------+

rng = np.random.default_rng(0)
zeta = rng.standard_normal((M, M))
zeta = (zeta + zeta.T) / 2  # the two-body THC tensor is symmetric
chi = rng.standard_normal((M, N // 2))
t_ell = rng.standard_normal(N // 2)
t_eigenvectors = np.linalg.qr(rng.standard_normal((N // 2, N // 2)))[0]


# +---------------------------------+
# |---- Set up registers -----------|
# +---------------------------------+

register_sizes = dict(qubitization_thc_wires(M, N, aleph, beth)) | {
    "angle_wires": precision,
    "phase_grad_wires": precision,
    "rot_work_wires": precision - 1,  # SemiAdder scratch of the RZ rule
    "elbow_work_wire": 1,  # the zeroed wire the Toffoli elbow rule needs
}
registers = qp.registers(register_sizes)
num_wires = sum(register_sizes.values())


# +---------------------------------+
# |---- Custom decomp rules --------|
# +---------------------------------+


# PennyLane's own elbow rule for the Toffoli (``_toffoli_elbow`` in
# ``pennylane/ops/op_math/controlled_ops.py``) does ``allocate(1, ZERO, restored=True)`` instead of
# borrowing a work wire, which is where the ``Allocate`` ops come from. Same rule, but on a wire we
# own: only one is live at a time, so one wire is enough for the whole circuit.
def make_toffoli_elbow_decomp(work_wire):
    @qp.register_resources(_toffoli_elbow_resources)
    def _toffoli_elbow(wires):
        qp.change_op_basis(
            qp.Elbow([wires[0], wires[1], work_wire]),
            qp.CNOT([work_wire, wires[2]]),
        )

    return _toffoli_elbow


# Phase gradient rotations, see https://pennylane.ai/compilation/phase-gradient
fixed_decomps = {
    qp.Toffoli: make_toffoli_elbow_decomp(registers["elbow_work_wire"][0]),
    qp.RZ: qp.transforms.decompositions.make_rz_to_phase_gradient_decomp(
        angle_wires=registers["angle_wires"],
        phase_grad_wires=registers["phase_grad_wires"],
        work_wires=registers["rot_work_wires"],
    ),
    qp.RY: qp.list_decomps(qp.RY)["_ry_to_rz_cliff"],
}


# +---------------------------------+
# |---- Compilation setup ----------|
# +---------------------------------+

# Elbows are targets, so the AND / un-AND structure of the algorithm is what gets compiled.
gate_set = CLIFFORD | {"TemporaryAND": 500, "Adjoint(TemporaryAND)": 0}

# Catalyst's ``to-ppr`` only takes H, S, T, X, Y, Z, S/T-dagger, I, CNOT, CZ and rotations, so the
# elbows are unrolled again on the way to MLIR and do not appear in the table.
mlir_gate_set = CLIFFORD | {"T": 500, "Adjoint(T)": 500}


def walk_operator():
    """One step of the qubitized walk: PREPARE, SELECT, PREPARE^dagger and the reflection."""
    with qp.queuing.AnnotatedQueue() as q:
        qubitization_thc(
            zeta,
            t_ell,
            chi,
            t_eigenvectors,
            aleph,
            beth,
            registers["system_wires"],
            registers["index_wires"],
            registers["prep_garbage_wires"],
            registers["gradient_wires"],
            registers["work_wires"],
        )
    return qp.tape.QuantumScript.from_queue(q)


def decompose(ops, target, num_work_wires):
    tape = qp.tape.QuantumScript(ops)
    (out,), _ = qp.transforms.decompose(
        tape, gate_set=target, num_work_wires=num_work_wires, fixed_decomps=fixed_decomps
    )
    return out.operations


if __name__ == "__main__":

    start_wallclock = time.time()
    start_proc = time.process_time()
    print(f"{M=} {N=} {aleph=} {beth=} {epsilon=} {precision=} {num_wires=}")

    elbow_ops = decompose(walk_operator().operations, gate_set, num_work_wires=None)
    print(Counter(op.name for op in elbow_ops))

    mlir_ops = decompose(elbow_ops, mlir_gate_set, num_work_wires=0)
    print(Counter(op.name for op in mlir_ops))

    @qp.qjit(capture=False, target="mlir", pipelines=default_qjit_pipelines)
    ## PBC DIALECT
    @qp.transforms.ppr_to_ppm
    @qp.transforms.to_ppr
    ## QUANTUM DIALECT
    @qp.transforms.combine_global_phases
    @qp.qnode(qp.device("null.qubit", wires=num_wires))
    def circuit(*_):
        for op in mlir_ops:
            qp.apply(op)
        return qp.expval(qp.Z(registers["system_wires"][0]))

    print(qp.specs(circuit, level="all-mlir")())

    print(f"Compilation took {time.process_time() - start_proc} CPU seconds ")
    print(f"                 {time.time() - start_wallclock} seconds.")
