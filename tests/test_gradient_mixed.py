import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.qubit import qchem_ops, parametric_ops
from pennylane.ops import channel


param_ops = [
    qchem_ops.SingleExcitation,
    qchem_ops.SingleExcitationPlus,
    qchem_ops.SingleExcitationMinus,
    qchem_ops.DoubleExcitation,
    qchem_ops.DoubleExcitationPlus,
    qchem_ops.DoubleExcitationMinus,
    qchem_ops.OrbitalRotation,
    parametric_ops.RX,
    parametric_ops.RY,
    parametric_ops.RZ,
    parametric_ops.PhaseShift,
    parametric_ops.ControlledPhaseShift,
    parametric_ops.Rot,
    parametric_ops.MultiRZ,
    parametric_ops.PauliRot,
    parametric_ops.CRX,
    parametric_ops.CRY,
    parametric_ops.CRZ,
    parametric_ops.CRot,
    parametric_ops.U1,
    parametric_ops.U2,
    parametric_ops.U3,
    parametric_ops.IsingXX,
    parametric_ops.IsingYY,
    parametric_ops.IsingZZ,
]

channel_ops = [
    channel.AmplitudeDamping,
    channel.GeneralizedAmplitudeDamping,
    channel.PhaseDamping,
    channel.DepolarizingChannel,
    channel.BitFlip,
    channel.ResetError,
    channel.PauliError,
    channel.PhaseFlip,
    channel.QubitChannel,
    channel.ThermalRelaxationError,
]


def parametric_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)


class TestMixedGradientBackprop():

    @pytest.mark.parametrize("op", param_ops)
    def test_param_ops(self, op, tol):
        """Test that we can apply all parametric qubit ops and compute gradients"""
        num_params = op.num_params
        num_qubits = op.num_wires

        pure_dev = qml.device("default.qubit", wires=num_qubits)
        mixed_dev = qml.device("default.mixed", wires=num_qubits)

        def circuit(params):
            op(*params, wires=range(num_qubits))
            return qml.expval(qml.PauliZ(wires=0))

        pure_circ = qml.QNode(circuit, device=pure_dev, interface="autograd", diff_method="backprop")
        grad_pure_circ = qml.grad(pure_circ)
        param_shift_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="parameter-shift")
        grad_param_shift_circ = qml.grad(param_shift_circ)
        mixed_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="backprop")
        grad_mixed_circ = qml.grad(mixed_circ)

        params = np.random.rand(num_params)
        params.requires_grad = True
        print(params)

        res = grad_mixed_circ(params)
        print(f"Mixed circ grad: {res}")
        param_shft_res = grad_param_shift_circ(params)
        print(f"Mixed param shift grad: {param_shft_res}")
        pure_res = grad_pure_circ(params)
        print(f"Pure state grad: {pure_res}")

        assert np.allclose(res, param_shft_res, atol=tol, rtol=0)
        assert np.allclose(res, pure_res, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", channel_ops)
    def test_channel_ops(self, op, tol):
        """Test that we can apply all channel ops and compute gradients"""
        num_params = op.num_params
        num_qubits = op.num_wires

        mixed_dev = qml.device("default.mixed", wires=num_qubits)

        def circuit(params):
            op(*params, wires=range(num_qubits))
            return qml.expval(qml.PauliZ(wires=0))

        param_shift_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="parameter-shift")
        grad_param_shift_circ = qml.grad(param_shift_circ)
        mixed_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="backprop")
        grad_mixed_circ = qml.grad(mixed_circ)

        params = np.random.rand(num_params)
        params.requires_grad = True
        print(params)

        res = grad_mixed_circ(params)
        print(f"Mixed circ grad: {res}")
        param_shft_res = grad_param_shift_circ(params)
        print(f"Mixed param shift grad: {param_shft_res}")

        assert np.allclose(res, param_shft_res, atol=tol, rtol=0)

    def test_expval(self, tol):
        """Test that we can compute expval and gradient of parametric circ"""
        num_qubits = 3
        num_params = 3
        pure_dev = qml.device("default.qubit", wires=num_qubits)
        mixed_dev = qml.device("default.mixed", wires=num_qubits)

        def circuit(params):
            parametric_circuit(params)
            return qml.expval(qml.PauliZ(wires=0) @ qml.PauliX(wires=1) @ qml.PauliY(wires=2))

        pure_circ = qml.QNode(circuit, device=pure_dev, interface="autograd", diff_method="backprop")
        grad_pure_circ = qml.grad(pure_circ)
        param_shift_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="parameter-shift")
        grad_param_shift_circ = qml.grad(param_shift_circ)
        mixed_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="backprop")
        grad_mixed_circ = qml.grad(mixed_circ)

        params = np.random.rand(num_params)
        params.requires_grad = True
        print(params)

        res = grad_mixed_circ(params)
        print(f"Mixed circ grad: {res}")
        param_shft_res = grad_param_shift_circ(params)
        print(f"Mixed param shift grad: {param_shft_res}")
        pure_res = grad_pure_circ(params)
        print(f"Pure state grad: {pure_res}")

        assert np.allclose(res, param_shft_res, atol=tol, rtol=0)
        assert np.allclose(res, pure_res, atol=tol, rtol=0)

    def test_var(self, tol):
        """Test that we can compute var and gradient of parametric circ"""
        num_qubits = 3
        num_params = 3
        pure_dev = qml.device("default.qubit", wires=num_qubits)
        mixed_dev = qml.device("default.mixed", wires=num_qubits)

        def circuit(params):
            parametric_circuit(params)
            return qml.var(qml.PauliZ(wires=0) @ qml.PauliX(wires=1) @ qml.PauliY(wires=2))

        pure_circ = qml.QNode(circuit, device=pure_dev, interface="autograd", diff_method="backprop")
        grad_pure_circ = qml.grad(pure_circ)
        param_shift_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="parameter-shift")
        grad_param_shift_circ = qml.grad(param_shift_circ)
        mixed_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="backprop")
        grad_mixed_circ = qml.grad(mixed_circ)

        params = np.random.rand(num_params)
        params.requires_grad = True
        print(params)

        res = grad_mixed_circ(params)
        print(f"Mixed circ grad: {res}")
        param_shft_res = grad_param_shift_circ(params)
        print(f"Mixed param shift grad: {param_shft_res}")
        pure_res = grad_pure_circ(params)
        print(f"Pure state grad: {pure_res}")

        assert np.allclose(res, param_shft_res, atol=tol, rtol=0)
        assert np.allclose(res, pure_res, atol=tol, rtol=0)

    def test_probs(self, tol):
        """Test that we can compute probs and gradient of parametric circ"""
        num_qubits = 3
        num_params = 3
        pure_dev = qml.device("default.qubit", wires=num_qubits)
        mixed_dev = qml.device("default.mixed", wires=num_qubits)

        def circuit(params):
            parametric_circuit(params)
            return qml.probs(wires=[0, 1, 2])

        pure_circ = qml.QNode(circuit, device=pure_dev, interface="autograd", diff_method="backprop")
        grad_pure_circ = qml.grad(pure_circ)
        param_shift_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="parameter-shift")
        grad_param_shift_circ = qml.grad(param_shift_circ)
        mixed_circ = qml.QNode(circuit, device=mixed_dev, interface="autograd", diff_method="backprop")
        grad_mixed_circ = qml.grad(mixed_circ)

        params = np.random.rand(num_params)
        params.requires_grad = True
        print(params)

        res = grad_mixed_circ(params)
        print(f"Mixed circ grad: {res}")
        param_shft_res = grad_param_shift_circ(params)
        print(f"Mixed param shift grad: {param_shft_res}")
        pure_res = grad_pure_circ(params)
        print(f"Pure state grad: {pure_res}")

        assert np.allclose(res, param_shft_res, atol=tol, rtol=0)
        assert np.allclose(res, pure_res, atol=tol, rtol=0)

    def test_state(self, tol):
        """Test that we can compute density matrix and gradient of parametric circ"""
        num_qubits = 3
        num_params = 3
        pure_dev = qml.device("default.qubit", wires=num_qubits)
        mixed_dev = qml.device("default.mixed", wires=num_qubits)

        def pure_circuit(params):
            parametric_circuit(params)
            return qml.density_matrix(wires=[0, 1, 2])

        def mixed_circuit(params):
            parametric_circuit(params)
            return qml.state()

        pure_circ = qml.QNode(pure_circuit, device=pure_dev, interface="autograd", diff_method="backprop")
        grad_pure_circ = qml.grad(pure_circ)
        param_shift_circ = qml.QNode(mixed_circuit, device=mixed_dev, interface="autograd", diff_method="parameter-shift")
        grad_param_shift_circ = qml.grad(param_shift_circ)
        mixed_circ = qml.QNode(mixed_circuit, device=mixed_dev, interface="autograd", diff_method="backprop")
        grad_mixed_circ = qml.grad(mixed_circ)

        params = np.random.rand(num_params)
        params.requires_grad = True
        print(params)

        res = grad_mixed_circ(params)
        print(f"Mixed circ grad: {res}")
        param_shft_res = grad_param_shift_circ(params)
        print(f"Mixed param shift grad: {param_shft_res}")
        pure_res = grad_pure_circ(params)
        print(f"Pure state grad: {pure_res}")

        assert np.allclose(res, param_shft_res, atol=tol, rtol=0)
        assert np.allclose(res, pure_res, atol=tol, rtol=0)
