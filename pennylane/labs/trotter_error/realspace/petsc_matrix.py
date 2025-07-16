from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

from petsc4py import PETSc
from slepc4py import SLEPc

if TYPE_CHECKING:
    from pennylane.labs.trotter_error.realspace import RealspaceMatrix, RealspaceOperator, RealspaceSum

def petsc_matrix(rs_mat: RealspaceMatrix, gridpoints: int) -> PETSc.Mat:
    block_matrix = []
    for row in range(rs_mat.states):
        row_blocks = [_realspace_sum_matrix(rs_mat.block(row, col), gridpoints) for col in range(rs_mat.states)]
        block_matrix.append(row_blocks)

    ret = PETSc.Mat().createNest(
        block_matrix,
        comm=PETSc.COMM_WORLD
    )
    ret.assemble()

    return ret


def spectral_norm(rs_mat: RealspaceMatrix, gridpoints: int):
    mat = petsc_matrix(rs_mat, gridpoints)

    solver = SLEPc.EPS()
    solver.create()
    solver.setOperators(mat)
    solver.setProblemType(SLEPc.EPS.ProblemType.HEP)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    solver.setFromOptions()
    solver.solve()

    nconv = solver.getConverged()
    vr, _ = mat.getVecs()
    vi, _ = mat.getVecs()

    return [solver.getEigenpair(i, vr, vi) for i in range(nconv)]

def _realspace_op_matrix(rs_op: RealspaceOperator, gridpoints: int) -> PETSc.Mat:
    ret = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    ret.setSizes((gridpoints**rs_op.modes, gridpoints**rs_op.modes))
    ret.setType(PETSc.Mat.Type.AIJ)
    ret.assemble()

    nonzero_coeffs = rs_op.coeffs.nonzero()
    for index, coeff in nonzero_coeffs.items():
        mat = _tensor_with_identity(rs_op.modes, gridpoints, index, rs_op.ops)
        ret.axpy(coeff, mat)

    ret.assemble()

    return ret

def _realspace_sum_matrix(rs_sum: RealspaceSum, gridpoints: int) -> PETSc.Mat:
    ret = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    ret.setSizes((gridpoints**rs_sum.modes, gridpoints**rs_sum.modes))
    ret.setType(PETSc.Mat.Type.AIJ)
    ret.assemble()

    ret = sum([_realspace_op_matrix(rs_op, gridpoints) for rs_op in rs_sum.ops], ret)
    ret.assemble()

    return ret

def _creation_matrix(gridpoints: int) -> PETSc.Mat:
    mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    mat.setSizes((gridpoints, gridpoints))
    mat.setType(PETSc.Mat.Type.AIJ)

    for i in range(gridpoints-1):
        mat.setValue(i+1, i, math.sqrt(i+1), PETSc.InsertMode.INSERT)

    mat.assemble()

    return mat

def _annihilation_matrix(gridpoints: int) -> PETSc.Mat:
    mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    mat.setSizes((gridpoints, gridpoints))
    mat.setType(PETSc.Mat.Type.AIJ)

    for i in range(gridpoints-1):
        mat.setValue(i, i+1, math.sqrt(i+1), PETSc.InsertMode.INSERT)

    mat.assemble()

    return mat

def _position_matrix(gridpoints: int) -> PETSc.Mat:
    return (_creation_matrix(gridpoints) + _annihilation_matrix(gridpoints)) / math.sqrt(2)

def _momentum_matrix(gridpoints: int) -> PETSc.Mat:
    return 1j*(_creation_matrix(gridpoints) - _annihilation_matrix(gridpoints)) / math.sqrt(2)

def _identity_matrix(dim: int) -> PETSc.Mat:
    mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    mat.setSizes((dim, dim))
    mat.setType(PETSc.Mat.Type.AIJ)

    for i in range(dim):
        mat.setValue(i, i, 1, PETSc.InsertMode.INSERT)

    mat.assemble()

    return mat

def _tensor_with_identity(modes: int, gridpoints: int, index: Tuple[int], ops: Tuple[str]) -> PETSc.Mat:
    mode_ops = [_identity_matrix(gridpoints)] * modes

    for mode in range(modes):
        for op_index, mode_at_index in enumerate(index):
            if mode == mode_at_index:
                if ops[op_index] == "P":
                    mode_ops[mode] = mode_ops[mode].matMult(_momentum_matrix(gridpoints))
                elif ops[op_index] == "Q":
                    mode_ops[mode] = mode_ops[mode].matMult(_position_matrix(gridpoints))
                else:
                    raise ValueError(f"Ops must be P or Q. Got {ops[op_index]}.")

    for mat in mode_ops:
        mat.assemble()

    ret = mode_ops[0]
    for mat in mode_ops[1:]:
        ret = ret.kron(mat)

    ret.assemble()

    return ret
