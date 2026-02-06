# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the quantum phase estimation subroutine.
"""
import numpy as np
import pytest
from scipy.stats import unitary_group

import pennylane as qp
from pennylane.exceptions import QuantumFunctionError


@pytest.mark.jax
def test_standard_validity():
    """Test standard validity criteria using assert_valid."""
    op = qp.QuantumPhaseEstimation(np.eye(4), target_wires=(0, 1), estimation_wires=[2, 5])
    assert op.target_wires == qp.wires.Wires([0, 1])
    qp.ops.functions.assert_valid(op)


class TestError:
    """Test that the QPE error is computed correctly."""

    @pytest.mark.parametrize(
        # the reference error is computed manually for a QPE operation with 2 estimation wires
        ("operator_error", "expected_error"),
        [(0.01, 0.03), (0.02, 0.06), (0.03, 0.09)],
    )
    def test_error_operator(self, operator_error, expected_error):
        """Test that QPE error is correct for a given custom operator."""

        class CustomOP(qp.resource.ErrorOperation):
            # pylint: disable=too-few-public-methods
            def error(self):
                return qp.resource.SpectralNormError(operator_error)

        operator = CustomOP(wires=[0])
        qpe_error = qp.QuantumPhaseEstimation(operator, estimation_wires=range(1, 3)).error().error

        assert np.allclose(qpe_error, expected_error)

    def test_error_zero(self):
        """Test that QPE error is zero for an operator with no error method."""
        unitary = qp.RX(0.1, wires=0)
        qpe_error = qp.QuantumPhaseEstimation(unitary, estimation_wires=range(1, 3)).error().error

        assert qpe_error == 0.0

    def test_error_unitary(self):
        """Test that QPE error is correct for a given unitary error."""

        u_exact = qp.RY(0.50, wires=0)
        u_apprx = qp.RY(0.51, wires=0)

        class CustomOP(qp.resource.ErrorOperation):
            # pylint: disable=too-few-public-methods
            def error(self):
                error_value = qp.resource.SpectralNormError.get_error(u_exact, u_apprx)
                return qp.resource.SpectralNormError(error_value)

        m_exact = qp.matrix(qp.QuantumPhaseEstimation(u_exact, estimation_wires=range(1, 3)))
        m_apprx = qp.matrix(qp.QuantumPhaseEstimation(u_apprx, estimation_wires=range(1, 3)))

        matrix_error = qp.math.max(qp.math.svd(m_exact - m_apprx, compute_uv=False))

        operator = CustomOP(wires=[0])
        qpe_error = qp.QuantumPhaseEstimation(operator, estimation_wires=range(1, 3)).error().error

        assert np.allclose(qpe_error, matrix_error, atol=1e-4)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        # the reference error is computed manually for a QPE operation with 2 estimation wires
        ("operator_error", "expected_error"),
        [(0.01, 0.03), (0.02, 0.06)],
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_error_interfaces(self, operator_error, interface, expected_error):
        """Test that the error method works with all interfaces."""

        class CustomOP(qp.resource.ErrorOperation):
            # pylint: disable=too-few-public-methods
            def error(self):
                spectral_norm_error = qp.resource.SpectralNormError(
                    qp.math.array(operator_error, like=interface)
                )
                return spectral_norm_error

        operator = CustomOP(wires=[0])
        qpe_error = qp.QuantumPhaseEstimation(operator, estimation_wires=range(1, 3)).error().error

        assert qp.math.get_interface(qpe_error) == interface
        assert np.allclose(qpe_error, expected_error)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_expected_qscript(self):
        """Tests if QuantumPhaseEstimation populates the quantum script as expected for a fixed example"""

        m = qp.RX(0.3, wires=0).matrix()

        op = qp.QuantumPhaseEstimation(m, target_wires=[0], estimation_wires=[1, 2])
        qscript = qp.tape.QuantumScript(op.decomposition())

        unitary = qp.QubitUnitary(m, wires=[0])
        with qp.queuing.AnnotatedQueue() as q:
            qp.Hadamard(1)
            qp.Hadamard(2)
            qp.ctrl(qp.pow(unitary, 2), control=[1])
            qp.ctrl(qp.pow(unitary, 1), control=[2])
            qp.adjoint(qp.QFT(wires=[1, 2]))
        qscript2 = qp.tape.QuantumScript.from_queue(q)
        assert len(qscript) == len(qscript2)
        # qp.equal doesn't work for Adjoint or Pow op yet, so we stop before we get to it.
        for op1, op2 in zip(qscript[:2], qscript2[:2]):
            qp.assert_equal(op1, op2)

        qp.assert_equal(qscript[2].base.base, qscript2[2].base.base)
        assert qscript[2].base.z, qscript2[2].base.z
        assert qscript[2].control_wires == qscript2[2].control_wires

        qp.assert_equal(qscript[3].base.base, qscript2[3].base.base)
        assert qscript[3].base.z == qscript2[3].base.z
        assert qscript[3].control_wires == qscript2[3].control_wires

        assert isinstance(qscript[-1], qp.ops.op_math.Adjoint)  # pylint: disable=no-member
        qp.assert_equal(qscript[-1].base, qp.QFT(wires=(1, 2)))

        assert np.allclose(qscript[1].matrix(), qscript[1].matrix())
        assert np.allclose(qscript[3].matrix(), qscript[3].matrix())

    @pytest.mark.parametrize("phase", [2, 3, 6, np.pi])
    def test_phase_estimated(self, phase):
        """Tests that the QPE circuit can correctly estimate the phase of a simple RX rotation."""
        # pylint: disable=cell-var-from-loop
        estimates = []
        wire_range = range(2, 10)

        for wires in wire_range:
            dev = qp.device("default.qubit", wires=wires)
            m = qp.RX(phase, wires=0).matrix()
            target_wires = [0]
            estimation_wires = range(1, wires)

            with qp.queuing.AnnotatedQueue() as q:
                # We want to prepare an eigenstate of RX, in this case |+>
                qp.Hadamard(wires=target_wires)

                qp.QuantumPhaseEstimation(
                    m, target_wires=target_wires, estimation_wires=estimation_wires
                )
                qp.probs(estimation_wires)

            tape = qp.tape.QuantumScript.from_queue(q)
            tapes, _ = dev.preprocess_transforms()([tape])
            assert len(tapes) == 1

            res = dev.execute(tapes)[0].flatten()
            initial_estimate = np.argmax(res) / 2 ** (wires - 1)

            # We need to rescale because RX is exp(- i theta X / 2) and we expect a unitary of the
            # form exp(2 pi i theta X)
            rescaled_estimate = (1 - initial_estimate) * np.pi * 4
            estimates.append(rescaled_estimate)

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - phase)
            err2 = np.abs(estimates[i + 1] - phase)
            assert err1 >= err2

        # This is quite a large error, but we'd need to push the qubit number up more to get it
        # lower
        assert np.allclose(estimates[-1], phase, rtol=1e-2)

    def test_phase_estimated_two_qubit(self):
        """Tests that the QPE circuit can correctly estimate the phase of a random two-qubit
        unitary."""
        # pylint: disable=cell-var-from-loop

        unitary = unitary_group.rvs(4, random_state=1967)
        eigvals, eigvecs = np.linalg.eig(unitary)

        state = eigvecs[:, 0]
        eigval = eigvals[0]
        phase = np.real_if_close(np.log(eigval) / (2 * np.pi * 1j))

        estimates = []
        wire_range = range(3, 11)

        for wires in wire_range:
            dev = qp.device("default.qubit", wires=wires)

            target_wires = [0, 1]
            estimation_wires = range(2, wires)

            with qp.queuing.AnnotatedQueue() as q:
                # We want to prepare an eigenstate of RX, in this case |+>
                qp.StatePrep(state, wires=target_wires)

                qp.QuantumPhaseEstimation(
                    unitary, target_wires=target_wires, estimation_wires=estimation_wires
                )
                qp.probs(estimation_wires)

            tape = qp.tape.QuantumScript.from_queue(q)
            tapes, _ = dev.preprocess_transforms()([tape])
            assert len(tapes) == 1
            res = dev.execute(tapes)[0].flatten()

            if phase < 0:
                estimate = np.argmax(res) / 2 ** (wires - 2) - 1
            else:
                estimate = np.argmax(res) / 2 ** (wires - 2)
            estimates.append(estimate)

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - phase)
            err2 = np.abs(estimates[i + 1] - phase)
            assert err1 >= err2

        # This is quite a large error, but we'd need to push the qubit number up more to get it
        # lower
        assert np.allclose(estimates[-1], phase, rtol=1e-2)

    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 4))
    def test_phase_estimated_single_ops(self, param):
        """Tests that the QPE works correctly for a single operator"""
        # pylint: disable=cell-var-from-loop

        unitary = qp.RX(param, wires=[0])

        # Analytical eigenvectors and phase of the unitary
        eig_vec = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])
        phase = param / (4 * np.pi)

        estimates = []
        wire_range = range(3, 11)

        for wires in wire_range:
            dev = qp.device("default.qubit", wires=wires)

            estimation_wires = range(1, wires - 1)
            target_wires = [0]

            tape = qp.tape.QuantumScript(
                [
                    qp.StatePrep(eig_vec, wires=target_wires),
                    qp.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires),
                ],
                [qp.probs(estimation_wires)],
            )

            tapes, _ = dev.preprocess_transforms()([tape])
            res = dev.execute(tapes)[0].flatten()
            assert len(tapes) == 1

            estimate = np.argmax(res) / 2 ** (wires - 2)
            estimates.append(estimate)

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - phase)
            err2 = np.abs(estimates[i + 1] - phase)
            assert err1 >= err2

        # This is a large error, but we'd need to push the qubit number up more to get it lower
        assert np.allclose(estimates[-1], phase, rtol=1e-2)

    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 4))
    def test_phase_estimated_ops(self, param):
        """Tests that the QPE works correctly for compound operators"""
        # pylint: disable=cell-var-from-loop

        unitary = qp.RX(param, wires=[0]) @ qp.CNOT(wires=[0, 1])

        # Analytical eigenvectors and phase of the unitary
        eig_vec = np.array([-1 / 2, -1 / 2, 1 / 2, 1 / 2])
        phase = param / (4 * np.pi)

        estimates = []
        wire_range = range(3, 11)

        for wires in wire_range:
            dev = qp.device("default.qubit", wires=wires)

            # Offset the index of target wires to test the wire maÏp
            estimation_wires = range(2, wires)
            target_wires = [0, 1]

            tape = qp.tape.QuantumScript(
                [
                    qp.StatePrep(eig_vec, wires=target_wires),
                    qp.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires),
                ],
                [qp.probs(estimation_wires)],
            )

            tapes, _ = dev.preprocess_transforms()([tape])
            assert len(tapes) == 1
            res = dev.execute(tapes)[0].flatten()

            estimate = np.argmax(res) / 2 ** (wires - 2)
            estimates.append(estimate)

        # Check that the error is monotonically decreasing
        for i in range(len(estimates) - 1):
            err1 = np.abs(estimates[i] - phase)
            err2 = np.abs(estimates[i + 1] - phase)
            assert err1 >= err2

        # This is a large error, but we'd need to push the qubit number up more to get it lower
        assert np.allclose(estimates[-1], phase, rtol=1e-2)

    def test_wires_specified(self):
        """Tests errors with specifying target_wires and estimation_wires"""

        unitary = unitary_group.rvs(4, random_state=1967)

        with pytest.raises(
            QuantumFunctionError,
            match="Target wires must be specified if the unitary is expressed as a matrix.",
        ):
            qp.QuantumPhaseEstimation(unitary, estimation_wires=[2, 3])

        unitary = qp.RX(3, wires=[0])
        with pytest.raises(
            QuantumFunctionError,
            match="The unitary is expressed as an operator, which already has target wires "
            "defined, do not additionally specify target wires.",
        ):
            qp.QuantumPhaseEstimation(unitary, target_wires=[1], estimation_wires=[2, 3])

        with pytest.raises(
            QuantumFunctionError,
            match="No estimation wires specified.",
        ):
            qp.QuantumPhaseEstimation(unitary)

    def test_map_wires(self):
        """Tests that QPE behaves correctly in a wire map"""
        # pylint: disable=protected-access

        unitary = qp.RX(np.pi / 4, wires=[0]) @ qp.CNOT(wires=[0, 1])
        qpe = qp.QuantumPhaseEstimation(unitary, estimation_wires=[2, 3])
        new_qpe = qp.map_wires(
            qpe,
            wire_map={
                0: 2,
                1: 3,
                2: 4,
                3: 5,
            },
        )

        assert list(new_qpe.wires) == [2, 3, 4, 5]
        assert list(new_qpe._hyperparameters["target_wires"]) == [2, 3]
        assert list(new_qpe._hyperparameters["estimation_wires"]) == [4, 5]
        assert list(new_qpe._hyperparameters["unitary"].wires) == [2, 3]

    def test_adjoint(self):
        """Test that the QPE adjoint works."""
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def qpe_circuit():
            qp.Hadamard(wires=0)
            qp.PauliX(wires=1)
            qp.QuantumPhaseEstimation(
                qp.PauliX.compute_matrix(),
                target_wires=[0],
                estimation_wires=[1, 2],
            )

            qp.adjoint(qp.QuantumPhaseEstimation)(
                qp.PauliX.compute_matrix(),
                target_wires=[0],
                estimation_wires=[1, 2],
            )
            qp.Hadamard(wires=0)
            qp.PauliX(wires=1)

            return qp.state()

        assert qp.math.isclose(qpe_circuit()[0], 1)  # pylint: disable=unsubscriptable-object

    @pytest.mark.jax
    def test_jit(self):
        """Tests the template correctly compiles with JAX JIT."""
        import jax

        phase = 5
        target_wires = [0]
        unitary = qp.RX(phase, wires=0).matrix()
        n_estimation_wires = 5
        estimation_wires = range(1, n_estimation_wires + 1)

        dev = qp.device("default.qubit", wires=n_estimation_wires + 1)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=target_wires)

            qp.QuantumPhaseEstimation(
                unitary, target_wires=target_wires, estimation_wires=estimation_wires
            )

            return qp.probs(estimation_wires)

        jit_circuit = jax.jit(circuit)

        assert qp.math.allclose(circuit(), jit_circuit())


class TestInputs:
    """Test inputs and pre-processing."""

    def test_same_wires(self):
        """Tests if a QuantumFunctionError is raised if target_wires and estimation_wires contain a
        common element"""

        with pytest.raises(QuantumFunctionError, match="The target wires and estimation wires"):
            qp.QuantumPhaseEstimation(np.eye(4), target_wires=[0, 1], estimation_wires=[1, 2])

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qp.QuantumPhaseEstimation(
            np.eye(4), target_wires=[0, 1], estimation_wires=[2, 3], id="a"
        )
        assert template.id == "a"
