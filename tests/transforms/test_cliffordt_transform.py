# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Clifford+T transform."""

import math
from functools import reduce

import pytest

import pennylane as qml
from pennylane.transforms.decompositions.clifford_t_transform import (
    _CLIFFORD_T_GATES,
    _CachedCallable,
    _map_wires,
    _merge_param_gates,
    _one_qubit_decompose,
    _rot_decompose,
    _two_qubit_decompose,
    check_clifford_t,
    clifford_t_decomposition,
)
from pennylane.transforms.optimization.optimization_utils import _fuse_global_phases

_SKIP_GATES = (qml.Barrier, qml.Snapshot, qml.WireCut)
_CLIFFORD_PHASE_GATES = _CLIFFORD_T_GATES + _SKIP_GATES

INVSQ2 = 1 / math.sqrt(2)
PI = math.pi


# pylint: disable=too-few-public-methods
class CustomOneQubitOperation(qml.operation.Operation):
    num_wires = 1

    @staticmethod
    def compute_matrix():
        return qml.math.conj(qml.math.transpose(qml.S.compute_matrix()))


# pylint: disable=too-few-public-methods
class CustomTwoQubitOperation(qml.operation.Operation):
    num_wires = 2

    @staticmethod
    def compute_matrix():
        return qml.math.conj(qml.math.transpose(qml.CNOT.compute_matrix()))


def circuit_1():
    """Circuit 1 with quantum chemistry gates"""
    qml.RZ(1.0, wires=[0])
    qml.PhaseShift(1.0, wires=[1])
    qml.SingleExcitation(2.0, wires=[1, 2])
    qml.PauliX(0)
    return qml.expval(qml.PauliZ(1))


def circuit_2():
    """Circuit 2 without chemistry gates"""
    qml.CRX(1, wires=[0, 1])
    qml.ISWAP(wires=[0, 1])
    qml.CSWAP(wires=[0, 1, 2])
    return qml.expval(qml.PauliZ(0))


def circuit_3():
    """Circuit 3 with Clifford gates"""
    qml.GlobalPhase(PI)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[1])
    qml.ISWAP(wires=[0, 1])
    qml.Hadamard(wires=[0])
    qml.WireCut(wires=[1])
    qml.RZ(PI, wires=[0])
    return qml.expval(qml.PauliZ(0))


def circuit_4():
    """Circuit 4 with a Template"""
    qml.RandomLayers(weights=qml.math.array([[0.1, -2.1, 1.4]]), wires=range(2))
    return qml.expval(qml.PauliZ(0))


def circuit_5():
    """Circuit 5 with Qubit Unitaries"""
    qml.DiagonalQubitUnitary(
        [qml.math.exp(1j * 0.1), qml.math.exp(1j * PI), INVSQ2 * (1 + 1j), INVSQ2 * (1 - 1j)],
        wires=[0, 1],
    )
    return qml.expval(qml.PauliZ(0))


def circuit_6():
    """Circuit 6 with adjoint S and T"""
    qml.adjoint(qml.S(wires=0))
    qml.PhaseShift(5 * math.pi / 2, wires=1)
    qml.adjoint(qml.T(wires=2))
    qml.PhaseShift(-3 * math.pi / 4, wires=3)
    return qml.expval(qml.PauliZ(0))


def circuit_7():
    """Circuit 7 with RX, RY, and RZ"""
    qml.RX(PI / 3, wires=0)
    qml.RZ(PI / 5, wires=0)
    qml.RY(PI / 8, wires=0)
    return qml.expval(qml.PauliZ(0))


def circuit_8():
    """Circuit 8 with only RZ and CNOT"""
    qml.RZ(PI / 8, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(PI / 2, wires=1)
    return qml.expval(qml.PauliZ(1))


def circuit_9(num_repeat, rand_angles):
    """Circuit 9 with a repeated operations"""
    for angle in rand_angles:
        for idx in range(num_repeat):
            qml.RZ(angle, idx)
        for idx in range(num_repeat):
            qml.CNOT([idx, (idx + 1) % num_repeat])
    return qml.expval(qml.Z(0))


class TestCliffordCompile:
    """Unit tests for clifford compilation function."""

    @pytest.mark.parametrize(
        "op, res",
        [
            (qml.DoubleExcitation(2.0, wires=[0, 1, 2, 3]), False),
            (qml.PauliX(wires=[1]), True),
            (qml.RX(3 * PI, wires=[1]), True),
            (qml.adjoint(qml.RX(3 * PI, wires=[1])), True),
            (qml.PhaseShift(2 * PI, wires=["a"]), True),
            (qml.ECR(wires=["e", "f"]), True),
            (qml.CH(wires=["a", "b"]), False),
            (qml.WireCut(0), False),
        ],
    )
    def test_clifford_checker(self, op, res):
        """Test Clifford checker operation for gate"""
        assert check_clifford_t(op) == res
        assert check_clifford_t(op, use_decomposition=True) == res

    @pytest.mark.parametrize(
        "circuit",
        [circuit_1, circuit_2, circuit_3, circuit_4, circuit_5],
    )
    def test_decomposition(self, circuit):
        """Test decomposition for the Clifford transform."""

        old_tape = qml.tape.make_qscript(circuit)()

        [new_tape], tape_fn = clifford_t_decomposition(old_tape, max_depth=3)

        assert all(
            isinstance(op, _CLIFFORD_PHASE_GATES)
            or isinstance(getattr(op, "base", None), _CLIFFORD_PHASE_GATES)
            for op in new_tape.operations
        )

        dev = qml.device("default.qubit")
        transform_program = dev.preprocess_transforms()
        res1, res2 = qml.execute(
            [old_tape, new_tape], device=dev, transform_program=transform_program
        )
        qml.math.isclose(res1, tape_fn([res2]), atol=1e-2)

    @pytest.mark.parametrize("circuit", [circuit_1, circuit_2, circuit_3])
    def test_decomposition_with_rs(self, circuit):
        """Test decomposition for the Clifford transform with Ross-Selinger method."""

        old_tape = qml.tape.make_qscript(circuit)()

        [new_tape], tape_fn = clifford_t_decomposition(old_tape, method="gridsynth")

        assert all(
            isinstance(op, _CLIFFORD_PHASE_GATES)
            or isinstance(getattr(op, "base", None), _CLIFFORD_PHASE_GATES)
            for op in new_tape.operations
        )

        dev = qml.device("default.qubit")
        transform_program = dev.preprocess_transforms()
        res1, res2 = qml.execute(
            [old_tape, new_tape], device=dev, transform_program=transform_program
        )
        qml.math.isclose(res1, tape_fn([res2]), atol=1e-2)

    @pytest.mark.catalyst
    @pytest.mark.jax
    @pytest.mark.external
    @pytest.mark.parametrize("circuit", [circuit_7, circuit_8])
    def test_decomposition_with_rs_qjit(self, circuit):
        """Test decomposition for the Clifford transform with Ross-Selinger method with QJIT enabled."""

        pytest.importorskip("jax")
        pytest.importorskip("catalyst")

        dev = qml.device("lightning.qubit", wires=4)
        qnode_cir = qml.qnode(dev)(circuit)
        decomp_cir = clifford_t_decomposition(qnode_cir, method="gridsynth")
        qjit_cir = qml.qjit(decomp_cir)

        res1, res2 = decomp_cir(), qjit_cir()
        assert qml.math.isclose(res1, res2, atol=1e-2)

    @pytest.mark.catalyst
    @pytest.mark.jax
    @pytest.mark.external
    def test_decomposition_with_rs_qjit_dynamic_param(self):
        """Test clifford T decomposition with qjit and dynamic parameters."""

        pytest.importorskip("jax")
        pytest.importorskip("catalyst")

        def circuit(angle, qb):
            qml.H(qb)
            qml.CNOT([qb, qb + 1])
            qml.RX(angle * 0.37, qb)
            qml.RZ(angle * 0.27, qb + 1)
            qml.RY(angle * 0.73, qb)
            qml.CNOT([qb + 1, qb])
            qml.H(qb)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        dev = qml.device("lightning.qubit", wires=2)
        decomposed_cir = qml.QNode(clifford_t_decomposition(circuit, method="gridsynth"), dev)
        qjit_cir = qml.qjit(decomposed_cir)

        angle, qb = PI, 0
        default_res, qjit_res = decomposed_cir(angle, qb), qjit_cir(angle, qb)

        assert qml.math.allclose(default_res, qjit_res, atol=1e-2)

    def test_qnode_decomposition(self):
        """Test decomposition for the Clifford transform applied to a QNode."""

        dev = qml.device("default.qubit")

        def qfunc():
            qml.PhaseShift(1.0, wires=[0])
            qml.PhaseShift(2.0, wires=[1])
            qml.ISWAP(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        original_qnode = qml.QNode(qfunc, dev)
        transformed_qnode = qml.QNode(
            clifford_t_decomposition(qfunc, max_depth=3, basis_length=10), dev
        )

        res1, res2 = original_qnode(), transformed_qnode()
        assert qml.math.isclose(res1, res2, atol=1e-2)

        tape = qml.workflow.construct_tape(transformed_qnode)()

        assert all(
            isinstance(op, _CLIFFORD_PHASE_GATES)
            or isinstance(getattr(op, "base", None), _CLIFFORD_PHASE_GATES)
            for op in tape.operations
        )

    def test_phase_shift_decomposition(self):
        """Test decomposition for the Clifford transform applied to the circuits with phase shifts."""
        old_tape = qml.tape.make_qscript(circuit_6)()

        [new_tape], _ = clifford_t_decomposition(old_tape, max_depth=3)

        compiled_ops = new_tape.operations

        assert qml.equal(compiled_ops[0], qml.adjoint(qml.S(0)))
        assert qml.equal(compiled_ops[1], qml.S(1))
        assert qml.equal(compiled_ops[2], qml.adjoint(qml.T(2)))
        assert qml.equal(compiled_ops[3], qml.T(3))

    @pytest.mark.parametrize("epsilon", [2e-2, 5e-2, 7e-2])
    @pytest.mark.parametrize("circuit", [circuit_3, circuit_4, circuit_5])
    def test_total_error(self, epsilon, circuit):
        """Ensure that given a certain epsilon, the total operator error is below the threshold."""
        dev = qml.device("default.qubit")

        qnode_basic = qml.QNode(circuit, dev)
        qnode_transformed = clifford_t_decomposition(
            qnode_basic, epsilon=epsilon, max_depth=10, basis_set=("T", "T*", "H")
        )
        mat_exact = qml.matrix(qnode_basic, wire_order=[0, 1])()
        mat_approx = qml.matrix(qnode_transformed, wire_order=[0, 1])()
        phase = qml.math.divide(
            mat_exact,
            mat_approx,
            out=qml.math.zeros_like(mat_exact, dtype=complex),
            where=mat_exact != 0,
        )[qml.math.nonzero(qml.math.round(mat_exact, 10))][0]
        mat_exact /= phase
        diff = mat_exact - mat_approx
        error = qml.math.sqrt(qml.math.real(qml.math.trace(qml.math.conj(diff).T @ diff)) / 2)
        assert error < epsilon

    @pytest.mark.parametrize(
        "op",
        [CustomOneQubitOperation(wires=0)],
    )
    def test_zxz_rotation_decomposition(self, op):
        """Test single-qubit gates are decomposed correctly using ZXZ rotations"""

        def circuit():
            qml.apply(op)
            return qml.probs(wires=0)

        old_tape = qml.tape.make_qscript(circuit)()

        [new_tape], tape_fn = clifford_t_decomposition(old_tape, max_depth=3)

        assert all(
            isinstance(op, _CLIFFORD_PHASE_GATES)
            or isinstance(getattr(op, "base", None), _CLIFFORD_PHASE_GATES)
            for op in new_tape.operations
        )

        dev = qml.device("default.qubit")
        transform_program = dev.preprocess_transforms()
        res1, res2 = qml.execute(
            [old_tape, new_tape], device=dev, transform_program=transform_program
        )
        qml.math.isclose(res1, tape_fn([res2]), atol=1e-2)

    @pytest.mark.parametrize(
        "op",
        [CustomTwoQubitOperation(wires=[0, 1])],
    )
    def test_su4_rotation_decomposition(self, op):
        """Test two-qubit gates are decomposed correctly using SU(4) rotations"""

        def circuit():
            qml.apply(op)
            return qml.probs(wires=0)

        old_tape = qml.tape.make_qscript(circuit)()

        [new_tape], tape_fn = clifford_t_decomposition(old_tape)

        assert all(
            isinstance(op, _CLIFFORD_PHASE_GATES)
            or isinstance(getattr(op, "base", None), _CLIFFORD_PHASE_GATES)
            for op in new_tape.operations
        )

        dev = qml.device("default.qubit")
        transform_program = dev.preprocess_transforms()
        res1, res2 = qml.execute(
            [old_tape, new_tape], device=dev, transform_program=transform_program
        )
        qml.math.isclose(res1, tape_fn([res2]), atol=1e-2)

    @pytest.mark.parametrize(
        "op", [qml.RX(1.0, wires="a"), qml.U3(1, 2, 3, wires=[1]), qml.PhaseShift(1.0, wires=[2])]
    )
    def test_one_qubit_decomposition(self, op):
        """Test decomposition for the Clifford transform."""

        decomp_ops, global_ops = _one_qubit_decompose(op)
        decomp_ops = _fuse_global_phases(decomp_ops + [global_ops])

        valid_gates = _CLIFFORD_PHASE_GATES + (qml.RZ,)
        assert all(
            isinstance(op, valid_gates) or isinstance(getattr(op, "base", None), valid_gates)
            for op in decomp_ops
        )

        global_op = decomp_ops.pop()
        assert isinstance(global_op, qml.GlobalPhase)

        matrix_op = reduce(
            lambda x, y: x @ y, [qml.matrix(op) for op in decomp_ops][::-1]
        ) * qml.matrix(global_op)

        # check for matrice equivalence up to global phase
        phase = qml.math.divide(
            matrix_op,
            qml.matrix(op),
            out=qml.math.zeros_like(matrix_op, dtype=complex),
            where=matrix_op != 0,
        )[qml.math.nonzero(qml.math.round(matrix_op, 10))]
        assert qml.math.allclose(
            phase / phase[0], qml.math.ones(qml.math.shape(phase)[0]), atol=1e-5
        )

    @pytest.mark.parametrize(
        "op",
        [
            qml.PSWAP(1.0, wires=["a", "b"]),
            qml.SingleExcitation(1, wires=[1, 2]),
            qml.IsingXX(1.0, wires=[2, 3]),
        ],
    )
    def test_two_qubit_decomposition(self, op):
        """Test decomposition for the Clifford transform."""

        decomp_ops = _fuse_global_phases(_two_qubit_decompose(op))

        valid_gates = _CLIFFORD_PHASE_GATES + (qml.RZ,)
        assert all(
            isinstance(op, valid_gates) or isinstance(getattr(op, "base", None), valid_gates)
            for op in decomp_ops
        )

        global_op = decomp_ops.pop()
        wire_map = {wire: idx for idx, wire in enumerate(op.wires)}
        mapped_op = [qml.map_wires(op, wire_map=wire_map) for op in decomp_ops][::-1]
        matrix_op = reduce(
            lambda x, y: x @ y, [qml.matrix(op, wire_order=[0, 1]) for op in mapped_op]
        ) * qml.matrix(global_op, wire_order=[0, 1])

        # check for matrice equivalence up to global phase
        phase = qml.math.divide(
            matrix_op,
            qml.matrix(op),
            out=qml.math.zeros_like(matrix_op, dtype=complex),
            where=matrix_op != 0,
        )[qml.math.nonzero(qml.math.round(matrix_op, 10))]
        assert qml.math.allclose(phase / phase[0], qml.math.ones(qml.math.shape(phase)[0]))

    @pytest.mark.parametrize(
        "op",
        [
            qml.adjoint(qml.RX(1.0, wires=["b"])),
            qml.Rot(1, 2, 3, wires=[2]),
            qml.PhaseShift(1.0, wires=[0]),
            qml.PhaseShift(3 * PI, wires=[0]),
        ],
    )
    def test_rot_decomposition(self, op):
        """Test decomposition for the Clifford transform."""

        decomp_ops = _fuse_global_phases(_rot_decompose(op))

        valid_gates = _CLIFFORD_PHASE_GATES + (qml.RZ,)
        assert all(
            isinstance(op, valid_gates) or isinstance(getattr(op, "base", None), valid_gates)
            for op in decomp_ops
        )

        decomp_ops, global_ops = decomp_ops[:-1], decomp_ops[-1]
        matrix_op = reduce(
            lambda x, y: x @ y, [qml.matrix(op) for op in decomp_ops][::-1]
        ) * qml.matrix(global_ops)

        # check for matrice equivalence up to global phase
        phase = qml.math.divide(
            matrix_op,
            qml.matrix(op),
            out=qml.math.zeros_like(matrix_op, dtype=complex),
            where=matrix_op != 0,
        )[qml.math.nonzero(qml.math.round(matrix_op, 10))]
        assert qml.math.allclose(phase / phase[0], qml.math.ones(qml.math.shape(phase)[0]))

    def test_merge_param_gates(self):
        """Test _merge_param_gates helper function"""
        operations = [
            qml.RX(0.1, wires=0),
            qml.RX(0.2, wires=0),
            qml.RY(0.3, wires=1),
            qml.RY(0.4, wires=1),
            qml.RX(0.5, wires=0),
        ]

        merge_ops = {"RX", "RY"}

        merged_ops, number_ops = _merge_param_gates(operations, merge_ops=merge_ops)

        assert len(merged_ops) == 2
        assert number_ops == 2

        assert isinstance(merged_ops[0], qml.RX)
        assert merged_ops[0].parameters == [0.8]  # 0.1 + 0.2 + 0.5 for wire 0
        assert isinstance(merged_ops[1], qml.RY)
        assert merged_ops[1].parameters == [0.7]  # 0.3 + 0.4 for wire 1

        merge_ops.discard("RY")
        merged_ops, number_ops = _merge_param_gates(operations, merge_ops=merge_ops)

        assert len(merged_ops) == 3
        assert number_ops == 1

        assert isinstance(merged_ops[0], qml.RX)
        assert merged_ops[0].parameters == [0.8]  # 0.1 + 0.2 + 0.5 for wire 0
        assert isinstance(merged_ops[1], qml.RY)
        assert merged_ops[1].parameters == [0.3]  # 0.3 for wire 1
        assert isinstance(merged_ops[1], qml.RY)
        assert merged_ops[2].parameters == [0.4]  # 0.4 for wire 1

    def test_raise_with_cliffordt_decomposition(self):
        """Test that exception is correctly raise when decomposing gates without any decomposition"""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            pass

        tape = qml.tape.QuantumScript([CustomOp(wires=[0, 1, 2])])

        with pytest.raises(ValueError, match="Cannot unroll"):
            clifford_t_decomposition(tape)

    @pytest.mark.parametrize("op", [qml.U1(1.0, wires=["b"])])
    def test_raise_with_rot_decomposition(self, op):
        """Test that exception is correctly raise when decomposing parametrized gates for which we already don't have a recipe"""

        with pytest.raises(
            ValueError,
            match="qml.RX, qml.RY, qml.RZ, qml.Rot and qml.PhaseShift",
        ):
            _rot_decompose(op)

    def test_zero_global_phase(self):
        """Test that global phase operation is added only when it is non-zero"""

        with qml.tape.QuantumTape() as tape:
            qml.CNOT([0, 1])

        [tape], _ = qml.clifford_t_decomposition(tape)

        assert not sum(isinstance(op, qml.GlobalPhase) for op in tape.operations)

    def test_raise_with_decomposition_method(self):
        """Test that exception is correctly raise when using incorrect decomposing method"""

        def qfunc():
            qml.RX(1.0, wires=[0])
            return qml.expval(qml.PauliZ(0))

        decomposed_qfunc = clifford_t_decomposition(qfunc, method="fast")

        with pytest.raises(
            NotImplementedError,
            match=r"Currently we only support Solovay-Kitaev \('sk'\) and Ross-Selinger \('gridsynth'\) decompositions",
        ):
            decomposed_qfunc()

    # pylint: disable= import-outside-toplevel
    @pytest.mark.all_interfaces
    def test_clifford_decompose_interfaces(self):
        """Test that unwrap converts lists to lists and interface variables to numpy."""

        dev = qml.device("default.qubit")

        def circuit(x):
            qml.RZ(x[0], wires=[0])
            qml.PhaseShift(x[1], wires=[1])
            qml.SingleExcitation(x[2], wires=[1, 2])
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(1))

        original_qnode = qml.QNode(circuit, dev)
        transfmd_qnode = qml.QNode(
            clifford_t_decomposition(circuit, max_depth=3, basis_length=10), dev
        )

        import jax
        import torch

        funres = []
        igrads = []
        coeffs = [1.0, 2.0, 3.0]
        for qcirc in [original_qnode, transfmd_qnode]:
            # Autograd Interface
            A = qml.numpy.array(coeffs)
            fres_numpy = qcirc(A)
            grad_numpy = qml.grad(qcirc, argnum=0)(A)

            # Jax Interface
            A = jax.numpy.array(coeffs)
            fres_jax = qcirc(A)
            grad_jax = jax.grad(qcirc, argnums=0)(A)

            # PyTorch Interface
            A = torch.tensor(coeffs, requires_grad=True)
            result = qcirc(A)
            result.backward()
            grad_torch = A.grad
            fres_torch = result

            funres.append([fres_numpy, fres_jax, fres_torch])
            igrads.append([grad_numpy, grad_jax, grad_torch])

        # Compare results
        assert all(qml.math.allclose(res1, res2, atol=1e-2) for res1, res2 in zip(*funres))
        assert all(qml.math.allclose(res1, res2, atol=1e-2) for res1, res2 in zip(*igrads))

    @pytest.mark.jax
    def test_abstract_wires(self):
        """Tests that rotations do not merge across operators with abstract wires."""

        import jax

        @jax.jit
        def f(w):
            tape = qml.tape.QuantumScript(
                [
                    qml.RX(0.5, wires=0),
                    qml.CNOT([w, 1]),
                    qml.RX(0.5, wires=0),
                ]
            )
            [tape], _ = clifford_t_decomposition(tape)
            return len(tape.operations)

        @jax.jit
        def f2(w):
            tape = qml.tape.QuantumScript(
                [
                    qml.CNOT([w, 1]),
                    qml.RX(0.5, wires=0),
                    qml.RX(0.5, wires=0),
                ]
            )
            [tape], _ = clifford_t_decomposition(tape)
            return len(tape.operations)

        assert f(0) > f2(0)


class TestCliffordCached:
    """Unit tests for clifford caching function."""

    # pylint: disable=protected-access, import-outside-toplevel, reimported
    def test_clifford_cached(self):
        """Test that the cached version of the circuit is equivalent to the original one."""

        import pennylane.transforms.decompositions.clifford_t_transform as clt2

        clt2._CLIFFORD_T_CACHE = None

        num_angles = 1
        rand_angles = qml.math.random.rand(num_angles)
        rand_angles = qml.math.concatenate((rand_angles, -rand_angles))

        num_repeat = 2
        old_tape = qml.tape.make_qscript(circuit_9)(num_repeat, rand_angles)
        _ = clifford_t_decomposition(old_tape, epsilon=10)

        assert isinstance(clt2._CLIFFORD_T_CACHE, _CachedCallable)
        cache_info = clt2._CLIFFORD_T_CACHE.query.cache_info()
        assert cache_info.misses == 2 * num_angles
        assert cache_info.hits == 2 * num_angles * (num_repeat - 1)

        num_repeat = 2
        old_tape = qml.tape.make_qscript(circuit_9)(num_repeat, rand_angles)
        _ = clifford_t_decomposition(old_tape, epsilon=10)

        assert isinstance(clt2._CLIFFORD_T_CACHE, _CachedCallable)
        cache_info = clt2._CLIFFORD_T_CACHE.query.cache_info()
        assert cache_info.misses == 2 * num_angles
        assert cache_info.hits == 2 * num_angles * (2 * num_repeat - 1)

        num_repeat = 2
        old_tape = qml.tape.make_qscript(circuit_9)(num_repeat, rand_angles)
        _ = clifford_t_decomposition(old_tape, cache_size=100)

        assert isinstance(clt2._CLIFFORD_T_CACHE, _CachedCallable)
        cache_info = clt2._CLIFFORD_T_CACHE.query.cache_info()
        assert cache_info.misses == 2 * num_angles
        assert cache_info.hits == 2 * num_angles * (num_repeat - 1)
        assert cache_info.maxsize == 100

    def test_wire_mapping(self):
        """Test that wire mapping is being cached correctly."""
        _map_wires.cache_clear()  # Clear the cache before testing

        for wire in range(5):
            assert _map_wires(qml.X(0), wire) == qml.X(wire)
        assert _map_wires.cache_info().hits == 0
        assert _map_wires.cache_info().misses == 5

        for wire in range(10):
            assert _map_wires(qml.X(0), wire) == qml.X(wire)
        assert _map_wires.cache_info().hits == 5
        assert _map_wires.cache_info().misses == 10

    # pylint: disable=protected-access, import-outside-toplevel, reimported
    def test_cached_with_rtol(self):
        """Test that caches are correctly identified as compatible or
        incompatible with a relative threshold for epsilon."""

        import pennylane.transforms.decompositions.clifford_t_transform as clt2

        clt2._CLIFFORD_T_CACHE = None

        cache1 = _CachedCallable(method="gridsynth", epsilon=1e-5, cache_size=100)

        assert cache1.compatible(
            method="gridsynth", epsilon=1e-3, cache_size=100, cache_eps_rtol=99, is_qjit=False
        )

        assert not cache1.compatible(
            method="gridsynth", epsilon=9e-6, cache_size=100, cache_eps_rtol=99, is_qjit=False
        )

        assert not cache1.compatible(
            method="gridsynth", epsilon=1e-4, cache_size=100, cache_eps_rtol=1e-1, is_qjit=False
        )


class TestCatalyst:
    """Unit tests for catalyst integration."""

    # pylint: disable=import-outside-toplevel
    @pytest.mark.external
    @pytest.mark.catalyst
    def test_catalyst_integration(self):
        """Test that the catalyst integration is working correctly."""

        import catalyst

        @qml.qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        @qml.clifford_t_decomposition
        def circuit():
            qml.RX(math.pi, [0])
            qml.RX(2 * math.pi, [1])
            return (catalyst.measure(0), catalyst.measure(1))

        results = circuit()
        assert results[0] and not results[1]
