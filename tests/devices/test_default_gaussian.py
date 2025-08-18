# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop,no-self-use

import numpy as np
import numpy.random
import pytest
from scipy.linalg import block_diag
from scipy.special import factorial as fac

import pennylane as qml
from pennylane.devices.default_gaussian import (
    beamsplitter,
    coherent_state,
    controlled_addition,
    controlled_phase,
    displaced_squeezed_state,
    fock_prob,
    quadratic_phase,
    rotation,
    squeezed_state,
    squeezing,
    thermal_state,
    two_mode_squeezing,
    vacuum_state,
)
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.wires import Wires

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array(
    [
        [
            -0.07843244 - 3.57825948e-01j,
            0.71447295 - 5.38069384e-02j,
            0.20949966 + 6.59100734e-05j,
            -0.50297381 + 2.35731613e-01j,
        ],
        [
            -0.26626692 + 4.53837083e-01j,
            0.27771991 - 2.40717436e-01j,
            0.41228017 - 1.30198687e-01j,
            0.01384490 - 6.33200028e-01j,
        ],
        [
            -0.69254712 - 2.56963068e-02j,
            -0.15484858 + 6.57298384e-02j,
            -0.53082141 + 7.18073414e-02j,
            -0.41060450 - 1.89462315e-01j,
        ],
        [
            -0.09686189 - 3.15085273e-01j,
            -0.53241387 - 1.99491763e-01j,
            0.56928622 + 3.97704398e-01j,
            -0.28671074 - 6.01574497e-02j,
        ],
    ]
)


H = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

hbar = 2


@pytest.fixture(scope="function", name="gaussian_device_1_wire")
def gaussian_device_1_wire_fixture():
    """Fixture of a default.gaussian device with 1 wire."""
    return qml.device("default.gaussian", wires=1)


@pytest.fixture(scope="function", name="gaussian_device_2_wires")
def gaussian_device_2_wires_fixture():
    """Fixture of a default.gaussian device with 2 wires."""
    return qml.device("default.gaussian", wires=2)


@pytest.fixture(scope="function", name="gaussian_device_3_wires")
def gaussian_device_3_wires_fixture():
    """Fixture of a default.gaussian device with 3 wires."""
    return qml.device("default.gaussian", wires=3)


@pytest.fixture(scope="function", name="gaussian_dev")
def gaussian_dev_fixture():  # alias
    """Fixture of a default.gaussian device with 2 wires."""
    return qml.device("default.gaussian", wires=2)


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.gaussian", wires=1, shots=1, analytic=True)


# pylint: disable=too-few-public-methods
class TestExceptions:
    """Tests that default.gaussian throws the correct error messages"""

    def test_sample_exception(self):
        """Test that default.gaussian raises an exception if sampling is attempted."""

        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.NumberOperator(0))

        with pytest.raises(
            NotImplementedError, match="default.gaussian does not support sampling NumberOperator"
        ):
            circuit()


class TestAuxillaryFunctions:
    """Tests the auxillary functions"""

    def test_fock_prob(self, tol):
        """Test fock_prob returns the correct Fock probabilities"""

        # an arbitrary two-mode Gaussian state generated using Strawberry Fields
        mu = np.array([0.6862, 0.4002, 0.09, 0.558]) * np.sqrt(hbar)
        cov = (
            np.array(
                [
                    [0.50750512, -0.04125979, -0.21058229, -0.07866912],
                    [-0.04125979, 0.50750512, -0.07866912, -0.21058229],
                    [-0.21058229, -0.07866912, 0.95906208, 0.27133391],
                    [-0.07866912, -0.21058229, 0.27133391, 0.95906208],
                ]
            )
            * hbar
        )

        # expected Fock state probabilities
        events = [(0, 0), (0, 1), (1, 1), (2, 3)]
        probs = [0.430461524043, 0.163699407559, 0.0582788388927, 0.00167706931355]

        for idx, e in enumerate(events):
            res = fock_prob(cov, mu, e, hbar=hbar)
            assert res == pytest.approx(probs[idx], abs=tol)


class TestGates:
    """Gate tests."""

    input_state = [vacuum_state(1), coherent_state(a=0.5)]

    @pytest.mark.parametrize("inp_state", input_state)
    def test_identity(self, inp_state, tol):
        inp_cov_mat = inp_state[0]
        inp_means = inp_state[1]

        O = qml.Identity.identity_op()
        out_means = O @ inp_means
        out_cov_mat = O @ inp_cov_mat @ O.T

        assert np.allclose(out_means, inp_means, atol=tol)
        assert np.allclose(
            out_cov_mat, inp_cov_mat, atol=tol
        )  # Identity op shouldn't change means or cov mat

    def test_rotation(self, tol):
        """Test the Fourier transform of a displaced state."""
        # pylint: disable=invalid-unary-operand-type

        alpha = 0.23 + 0.12j
        S = rotation(np.pi / 2)

        # apply to a coherent state. F{x, p} -> {-p, x}
        out = S @ np.array([alpha.real, alpha.imag]) * np.sqrt(2 * hbar)
        expected = np.array([-alpha.imag, alpha.real]) * np.sqrt(2 * hbar)
        assert out == pytest.approx(expected, abs=tol)

    def test_squeezing(self, tol):
        """Test the squeezing symplectic transform."""

        r = 0.543
        phi = 0.123
        S = squeezing(r, phi)

        # apply to an identity covariance matrix
        out = S @ S.T
        expected = rotation(phi / 2) @ np.diag(np.exp([-2 * r, 2 * r])) @ rotation(phi / 2).T
        assert out == pytest.approx(expected, abs=tol)

    def test_quadratic_phase(self, tol):
        """Test the quadratic phase symplectic transform."""

        s = 0.543
        S = quadratic_phase(s)

        # apply to a coherent state. P[x, p] -> [x, p+sx]
        alpha = 0.23 + 0.12j
        out = S @ np.array([alpha.real, alpha.imag]) * np.sqrt(2 * hbar)
        expected = np.array([alpha.real, alpha.imag + s * alpha.real]) * np.sqrt(2 * hbar)
        assert out == pytest.approx(expected, abs=tol)

    def test_beamsplitter(self, tol):
        """Test the beamsplitter symplectic transform."""

        theta = 0.543
        phi = 0.312
        S = beamsplitter(theta, phi)

        # apply to a coherent state. BS|a1, a2> -> |ta1-r^*a2, ra1+ta2>
        a1 = 0.23 + 0.12j
        a2 = 0.23 + 0.12j
        out = S @ np.array([a1.real, a2.real, a1.imag, a2.imag]) * np.sqrt(2 * hbar)

        T = np.cos(theta)
        R = np.exp(1j * phi) * np.sin(theta)
        a1out = T * a1 - R.conj() * a2
        a2out = R * a2 + T * a1
        expected = np.array([a1out.real, a2out.real, a1out.imag, a2out.imag]) * np.sqrt(2 * hbar)
        assert out == pytest.approx(expected, abs=tol)

    def test_two_mode_squeezing(self, tol):
        """Test the two mode squeezing symplectic transform."""

        r = 0.543
        phi = 0.123
        S = two_mode_squeezing(r, phi)

        # test that S = B^\dagger(pi/4, 0) [S(z) x S(-z)] B(pi/4)
        B = beamsplitter(np.pi / 4, 0)
        Sz = block_diag(squeezing(r, phi), squeezing(-r, phi))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]
        expected = B.conj().T @ Sz @ B
        assert S == pytest.approx(expected, abs=tol)

        # test that S |a1, a2> = |ta1+ra2, ta2+ra1>
        a1 = 0.23 + 0.12j
        a2 = 0.23 + 0.12j
        out = S @ np.array([a1.real, a2.real, a1.imag, a2.imag]) * np.sqrt(2 * hbar)

        T = np.cosh(r)
        R = np.exp(1j * phi) * np.sinh(r)
        a1out = T * a1 + R * np.conj(a2)
        a2out = T * a2 + R * np.conj(a1)
        expected = np.array([a1out.real, a2out.real, a1out.imag, a2out.imag]) * np.sqrt(2 * hbar)
        assert out == pytest.approx(expected, abs=tol)

    def test_controlled_addition(self, tol):
        """Test the CX symplectic transform."""

        s = 0.543
        S = controlled_addition(s)

        # test that S = B(theta+pi/2, 0) [S(z) x S(-z)] B(theta, 0)
        r = np.arcsinh(-s / 2)
        theta = 0.5 * np.arctan2(-1 / np.cosh(r), -np.tanh(r))
        Sz = block_diag(squeezing(r, 0), squeezing(-r, 0))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]

        expected = beamsplitter(theta + np.pi / 2, 0) @ Sz @ beamsplitter(theta, 0)
        assert S == pytest.approx(expected, abs=tol)

        # test that S[x1, x2, p1, p2] -> [x1, x2+sx1, p1-sp2, p2]
        x1 = 0.5432
        x2 = -0.453
        p1 = 0.154
        p2 = -0.123
        out = S @ np.array([x1, x2, p1, p2]) * np.sqrt(2 * hbar)
        expected = np.array([x1, x2 + s * x1, p1 - s * p2, p2]) * np.sqrt(2 * hbar)
        assert out == pytest.approx(expected, abs=tol)

    def test_controlled_phase(self, tol):
        """Test the CZ symplectic transform."""

        s = 0.543
        S = controlled_phase(s)

        # test that S = R_2(pi/2) CX(s) R_2(pi/2)^\dagger
        R2 = block_diag(np.identity(2), rotation(np.pi / 2))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]
        expected = R2 @ controlled_addition(s) @ R2.conj().T
        assert S == pytest.approx(expected, abs=tol)

        # test that S[x1, x2, p1, p2] -> [x1, x2, p1+sx2, p2+sx1]
        x1 = 0.5432
        x2 = -0.453
        p1 = 0.154
        p2 = -0.123
        out = S @ np.array([x1, x2, p1, p2]) * np.sqrt(2 * hbar)
        expected = np.array([x1, x2, p1 + s * x2, p2 + s * x1]) * np.sqrt(2 * hbar)
        assert out == pytest.approx(expected, abs=tol)


class TestStates:
    """State tests."""

    def test_vacuum_state(self, tol):
        """Test the vacuum state is correct."""
        wires = 3
        cov, means = vacuum_state(wires, hbar=hbar)
        assert means == pytest.approx(np.zeros([2 * wires]), abs=tol)
        assert cov == pytest.approx(np.identity(2 * wires) * hbar / 2, abs=tol)

    def test_coherent_state(self, tol):
        """Test the coherent state is correct."""
        a = 0.432 - 0.123j
        cov, means = coherent_state(a, hbar=hbar)
        assert means == pytest.approx(np.array([a.real, a.imag]) * np.sqrt(2 * hbar), abs=tol)
        assert cov == pytest.approx(np.identity(2) * hbar / 2, abs=tol)

    def test_squeezed_state(self, tol):
        """Test the squeezed state is correct."""
        r = 0.432
        phi = 0.123
        cov, means = squeezed_state(r, phi, hbar=hbar)

        # test vector of means is zero
        assert means == pytest.approx(np.zeros([2]), abs=tol)

        R = rotation(phi / 2)
        expected = R @ np.array([[np.exp(-2 * r), 0], [0, np.exp(2 * r)]]) * hbar / 2 @ R.T
        # test covariance matrix is correct
        assert cov == pytest.approx(expected, abs=tol)

    def test_displaced_squeezed_state(self, tol):
        """Test the displaced squeezed state is correct."""
        alpha = 0.541 + 0.109j
        a = abs(alpha)
        phi_a = np.angle(alpha)
        r = 0.432
        phi_r = 0.123
        cov, means = displaced_squeezed_state(a, phi_a, r, phi_r, hbar=hbar)

        # test vector of means is correct
        assert means == pytest.approx(
            np.array([alpha.real, alpha.imag]) * np.sqrt(2 * hbar), abs=tol
        )

        R = rotation(phi_r / 2)
        expected = R @ np.array([[np.exp(-2 * r), 0], [0, np.exp(2 * r)]]) * hbar / 2 @ R.T
        # test covariance matrix is correct
        assert cov == pytest.approx(expected, abs=tol)

    def thermal_state(self, tol):
        """Test the thermal state is correct."""
        nbar = 0.5342
        cov, means = thermal_state(nbar, hbar=hbar)
        assert means == pytest.approx(np.zeros([2]), abs=tol)
        assert np.all((cov.diag * 2 / hbar - 1) / 2 == nbar)


class TestDefaultGaussianDevice:
    """Test the default gaussian device. The test ensures that the device is properly
    applying gaussian operations and calculating the correct observables."""

    def test_operation_map(self, gaussian_dev):
        """Test that default Gaussian device supports all PennyLane Gaussian CV gates."""

        non_supported = {
            "FockDensityMatrix",
            "FockStateVector",
            "FockState",
            "CrossKerr",
            "CatState",
            "CubicPhase",
            "Kerr",
        }

        cv_ops = set(qml.ops._cv__ops__) | {"Identity"}  # we also support identity operation
        assert cv_ops - non_supported == set(gaussian_dev._operation_map)

    def test_observable_map(self, gaussian_dev):
        """Test that default Gaussian device supports all PennyLane Gaussian continuous observables."""
        assert set(qml.ops._cv__obs__) - {"TensorN"} | {"Identity"} - {"Heterodyne"} == set(
            gaussian_dev._observable_map
        )

    def test_apply_general(self, tol):
        """Test the application of gates to a state, using gates that use the default logic in
        the apply function."""

        gaussian_dev = qml.device("default.gaussian", wires=1)

        def dummy_gate_fn():
            # dummy implementation of a symplectic matrix
            return np.array([[2, 0], [0, 3]])

        # temporarily add the gate to the device using an arbitrary implementation
        gaussian_dev._operation_map["DummyGate"] = dummy_gate_fn

        gaussian_dev.apply("DummyGate", wires=qml.wires.Wires([0]), par=[])

        exp_cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        exp_mu = np.array([0.0, 0.0])

        # verify the device is now in the expected state
        assert gaussian_dev._state[0] == pytest.approx(exp_cov, abs=tol)
        assert gaussian_dev._state[1] == pytest.approx(exp_mu, abs=tol)

    def test_apply_gaussianstate(self, tol):
        """Test the application of the GaussianState gate to a state, since it
        uses a forked logic in the apply function."""

        gaussian_dev = qml.device("default.gaussian", wires=2)

        # start in the displaced squeezed state
        alpha = 0.542 + 0.123j
        a = abs(alpha)
        phi_a = np.angle(alpha)
        r = 0.652
        phi_r = -0.124
        gaussian_dev.apply("DisplacedSqueezedState", wires=Wires([0]), par=[a, phi_a, r, phi_r])
        gaussian_dev.apply("DisplacedSqueezedState", wires=Wires([1]), par=[a, phi_a, r, phi_r])

        cov = np.diag([0.5234] * 4)
        mu = np.array([0.432, 0.123, 0.342, 0.123])
        p = [cov, mu]
        w = list(range(2))
        expected_out = [cov, mu]

        gaussian_dev.apply("GaussianState", wires=Wires(w), par=p)

        # verify the device is now in the expected state
        assert gaussian_dev._state[0] == pytest.approx(expected_out[0], abs=tol)
        assert gaussian_dev._state[1] == pytest.approx(expected_out[1], abs=tol)

    def test_apply_squeezedstate(self, tol):
        """Test the application of one of the state preparation gates, since they
        use a forked logic in the apply function."""

        gaussian_dev = qml.device("default.gaussian", wires=2)

        # start in the displaced squeezed state
        alpha = 0.542 + 0.123j
        a = abs(alpha)
        phi_a = np.angle(alpha)
        r = 0.652
        phi_r = -0.124

        gaussian_dev.apply("DisplacedSqueezedState", wires=Wires([0]), par=[a, phi_a, r, phi_r])
        gaussian_dev.apply("DisplacedSqueezedState", wires=Wires([1]), par=[a, phi_a, r, phi_r])

        w = [0]
        p = [0.432423, -0.12312]
        fn = gaussian_dev._operation_map["SqueezedState"]
        cov, mu = fn(*p, hbar=hbar)
        expected_out = gaussian_dev._state
        expected_out[1][[w[0], w[0] + 2]] = mu

        ind = np.concatenate([np.array([w[0]]), np.array([w[0]]) + 2])
        rows = ind.reshape(-1, 1)
        cols = ind.reshape(1, -1)
        expected_out[0][rows, cols] = cov

        gaussian_dev.apply("SqueezedState", wires=Wires(w), par=p)

        # verify the device is now in the expected state
        assert gaussian_dev._state[0] == pytest.approx(expected_out[0], abs=tol)
        assert gaussian_dev._state[1] == pytest.approx(expected_out[1], abs=tol)

    def test_apply_displacement(self, tol):
        """Test the application of the displacement gate to a state, since it
        uses a forked logic in the apply function."""

        gaussian_dev = qml.device("default.gaussian", wires=2)

        # start in the displaced squeezed state
        alpha = 0.542 + 0.123j
        a = abs(alpha)
        phi_a = np.angle(alpha)
        r = 0.652
        phi_r = -0.124
        gaussian_dev.apply("DisplacedSqueezedState", wires=Wires([0]), par=[a, phi_a, r, phi_r])
        gaussian_dev.apply("DisplacedSqueezedState", wires=Wires([1]), par=[a, phi_a, r, phi_r])

        w = [0]
        p = [0.432423, -0.12312]
        alpha = p[0] * np.exp(1j * p[1])
        state = gaussian_dev._state
        mu = state[1].copy()
        mu[w[0]] += alpha.real * np.sqrt(2 * hbar)
        mu[w[0] + 2] += alpha.imag * np.sqrt(2 * hbar)
        expected_out = state[0], mu

        gaussian_dev.apply("Displacement", wires=Wires(w), par=p)

        # verify the device is now in the expected state
        assert gaussian_dev._state[0] == pytest.approx(expected_out[0], abs=tol)
        assert gaussian_dev._state[1] == pytest.approx(expected_out[1], abs=tol)

    def test_apply_errors(self, gaussian_dev):
        """Test that apply fails for incorrect state preparation"""

        with pytest.raises(ValueError, match="incorrect size for the number of subsystems"):
            p = [thermal_state(0.5)]
            gaussian_dev.apply("GaussianState", wires=Wires([0]), par=[p])

        with pytest.raises(ValueError, match="Incorrect number of subsystems"):
            p = U
            gaussian_dev.apply("InterferometerUnitary", wires=Wires([0]), par=[p])

        with pytest.raises(qml.wires.WireError, match="Did not find some of the wires"):
            p = U2
            # dev = DefaultGaussian(wires=4, shots=1000, hbar=hbar)
            gaussian_dev.apply("InterferometerUnitary", wires=Wires([0, 1, 2]), par=[p])

    def test_expectation(self, tol):
        """Test that expectation values are calculated correctly"""

        dev = qml.device("default.gaussian", wires=1, hbar=hbar)

        # test correct mean for <n> of a displaced thermal state
        nbar = 0.5431
        alpha = 0.324 - 0.59j
        dev.apply("ThermalState", wires=Wires([0]), par=[nbar])
        dev.apply("Displacement", wires=Wires([0]), par=[alpha, 0])
        mean = dev.expval("NumberOperator", Wires([0]), [])
        assert mean == pytest.approx(np.abs(alpha) ** 2 + nbar, abs=tol)

        # test correct mean for Homodyne P measurement
        alpha = 0.324 - 0.59j
        dev.apply("CoherentState", wires=Wires([0]), par=[alpha])
        mean = dev.expval("QuadP", Wires([0]), [])
        assert mean == pytest.approx(alpha.imag * np.sqrt(2 * hbar), abs=tol)

        # test correct mean for Homodyne measurement
        mean = dev.expval("QuadOperator", Wires([0]), [np.pi / 2])
        assert mean == pytest.approx(alpha.imag * np.sqrt(2 * hbar), abs=tol)

        # test correct mean for number state expectation |<n|alpha>|^2
        # on a coherent state
        for n in range(3):
            mean = dev.expval("FockStateProjector", Wires([0]), [np.array([n])])
            expected = np.abs(np.exp(-np.abs(alpha) ** 2 / 2) * alpha**n / np.sqrt(fac(n))) ** 2
            assert mean == pytest.approx(expected, abs=tol)

        # test correct mean for number state expectation |<n|S(r)>|^2
        # on a squeezed state
        n = 1
        r = 0.4523
        dev.apply("SqueezedState", wires=Wires([0]), par=[r, 0])
        mean = dev.expval("FockStateProjector", Wires([0]), [np.array([2 * n])])
        base = np.sqrt(fac(2 * n)) / (2**n * fac(n)) * (-np.tanh(r)) ** n / np.sqrt(np.cosh(r))
        expected = np.abs(base) ** 2
        assert mean == pytest.approx(expected, abs=tol)

    def test_variance_displaced_thermal_mean_photon(self, tol):
        """test correct variance for <n> of a displaced thermal state"""
        dev = qml.device("default.gaussian", wires=1, hbar=hbar)

        nbar = 0.5431
        alpha = 0.324 - 0.59j
        dev.apply("ThermalState", wires=Wires([0]), par=[nbar])
        dev.apply("Displacement", wires=Wires([0]), par=[alpha, 0])
        var = dev.var("NumberOperator", Wires([0]), [])
        assert var == pytest.approx(nbar**2 + nbar + np.abs(alpha) ** 2 * (1 + 2 * nbar), abs=tol)

    def test_variance_coherent_homodyne(self, tol):
        """test correct variance for Homodyne P measurement"""
        dev = qml.device("default.gaussian", wires=1, hbar=hbar)

        alpha = 0.324 - 0.59j
        dev.apply("CoherentState", wires=Wires([0]), par=[alpha])
        var = dev.var("QuadP", Wires([0]), [])
        assert var == pytest.approx(hbar / 2, abs=tol)

        # test correct mean and variance for Homodyne measurement
        var = dev.var("QuadOperator", Wires([0]), [np.pi / 2])
        assert var == pytest.approx(hbar / 2, abs=tol)

    def test_variance_coherent_numberstate(self, tol):
        """test correct variance for number state expectation |<n|alpha>|^2
        on a coherent state
        """
        dev = qml.device("default.gaussian", wires=1, hbar=hbar)

        alpha = 0.324 - 0.59j

        dev.apply("CoherentState", wires=Wires([0]), par=[alpha])

        for n in range(3):
            var = dev.var("FockStateProjector", Wires([0]), [np.array([n])])
            mean = np.abs(np.exp(-np.abs(alpha) ** 2 / 2) * alpha**n / np.sqrt(fac(n))) ** 2
            assert var == pytest.approx(mean * (1 - mean), abs=tol)

    def test_variance_squeezed_numberstate(self, tol):
        """test correct variance for number state expectation |<n|S(r)>|^2
        on a squeezed state
        """
        dev = qml.device("default.gaussian", wires=1, hbar=hbar)

        n = 1
        r = 0.4523
        dev.apply("SqueezedState", wires=Wires([0]), par=[r, 0])
        var = dev.var("FockStateProjector", Wires([0]), [np.array([2 * n])])
        base = np.sqrt(fac(2 * n)) / (2**n * fac(n)) * (-np.tanh(r)) ** n / np.sqrt(np.cosh(r))
        mean = np.abs(base) ** 2
        assert var == pytest.approx(mean * (1 - mean), abs=tol)

    def test_reduced_state(self, gaussian_dev, tol):
        """Test reduced state"""

        # Test requesting via an integer
        res = gaussian_dev.reduced_state(Wires(0))
        expected = gaussian_dev.reduced_state(Wires([0]))
        assert res[0] == pytest.approx(expected[0], abs=tol)
        assert res[1] == pytest.approx(expected[1], abs=tol)

        # Test requesting all wires returns the full state
        res = gaussian_dev.reduced_state(Wires([0, 1]))
        expected = gaussian_dev._state
        assert res[0] == pytest.approx(expected[0], abs=tol)
        assert res[1] == pytest.approx(expected[1], abs=tol)


def input_logger(*args):
    """Helper function for monkeypatch: logs its input."""
    input_logger.args = args
    return np.array([1, 2, 3, 4, 5])


class TestSample:
    """Tests that sampling is correctly implemented."""

    @pytest.mark.parametrize("alpha", [0.324 - 0.59j, 2.3 + 1.2j, 1.3j, -1.2])
    def test_sampling_parameters_coherent(self, tol, gaussian_device_1_wire, alpha, monkeypatch):
        """Tests that the np.random.normal is called with the correct parameters that reflect
        the underlying distribution for a coherent state."""

        mean = alpha.imag * np.sqrt(2 * gaussian_device_1_wire.hbar)
        std = gaussian_device_1_wire.hbar / 2
        gaussian_device_1_wire.apply("CoherentState", wires=Wires([0]), par=[alpha])

        with monkeypatch.context() as m:
            m.setattr(numpy.random, "normal", input_logger)
            gaussian_device_1_wire.sample("QuadP", Wires([0]), [])
            assert np.isclose(input_logger.args[0], mean, atol=tol, rtol=0)
            assert np.isclose(input_logger.args[1], std, atol=tol, rtol=0)
            assert gaussian_device_1_wire.shots == qml.measurements.Shots(input_logger.args[2])

    @pytest.mark.parametrize("alpha", [0.324 - 0.59j, 2.3 + 1.2j, 1.3j, -1.2])
    def test_sampling_parameters_coherent_quad_operator(
        self, tol, gaussian_device_1_wire, alpha, monkeypatch
    ):
        """Tests that the np.random.normal is called with the correct parameters that reflect
        the underlying distribution for a coherent state when using QuadOperator."""

        mean = alpha.imag * np.sqrt(2 * gaussian_device_1_wire.hbar)
        std = gaussian_device_1_wire.hbar / 2
        gaussian_device_1_wire.apply("CoherentState", wires=Wires([0]), par=[alpha])

        with monkeypatch.context() as m:
            m.setattr(numpy.random, "normal", input_logger)
            gaussian_device_1_wire.sample("QuadOperator", Wires([0]), [np.pi / 2])
            assert np.isclose(input_logger.args[0], mean, atol=tol, rtol=0)
            assert np.isclose(input_logger.args[1], std, atol=tol, rtol=0)
            assert gaussian_device_1_wire.shots == qml.measurements.Shots(input_logger.args[2])

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize("r,phi", [(1.0, 0.0)])
    def test_sampling_parameters_squeezed(self, tol, gaussian_device_1_wire, r, phi, monkeypatch):
        """Tests that the np.random.normal is called with the correct parameters that reflect
        the underlying distribution for a squeezed state."""

        mean = 0.0
        std = np.sqrt(gaussian_device_1_wire.hbar * np.exp(2 * r) / 2)
        gaussian_device_1_wire.apply("SqueezedState", wires=Wires([0]), par=[r, phi])

        with monkeypatch.context() as m:
            m.setattr(numpy.random, "normal", input_logger)
            gaussian_device_1_wire.sample("QuadP", Wires([0]), [])
            assert np.isclose(input_logger.args[0], mean, atol=tol, rtol=0)
            assert np.isclose(input_logger.args[1], std, atol=tol, rtol=0)
            assert gaussian_device_1_wire.shots == qml.measurements.Shots(input_logger.args[2])

    @pytest.mark.parametrize(
        "observable,n_sample", [("QuadP", 10), ("QuadP", 25), ("QuadX", 1), ("QuadX", 16)]
    )
    def test_sample_shape_and_dtype(self, gaussian_device_2_wires, observable, n_sample):
        """Test that the sample function outputs samples of the right size"""

        gaussian_device_2_wires.target_device._shots = n_sample
        sample = gaussian_device_2_wires.sample(observable, Wires([0]), [])

        assert np.array_equal(sample.shape, (n_sample,))
        assert sample.dtype == np.dtype("float")

    def test_sample_error_multi_wire(self, gaussian_device_2_wires):
        """Test that the sample function raises an error if multiple wires are given"""

        with pytest.raises(ValueError, match="Only one mode can be measured in homodyne"):
            gaussian_device_2_wires.sample("QuadP", [0, 1], [])

    @pytest.mark.parametrize(
        "observable",
        sorted(set(qml.ops.cv.__obs__) - {"QuadP", "QuadX", "QuadOperator"}),
    )
    def test_sample_error_unsupported_observable(self, gaussian_device_2_wires, observable):
        """Test that the sample function raises an error if the given observable is not supported"""

        with pytest.raises(NotImplementedError, match="default.gaussian does not support sampling"):
            gaussian_device_2_wires.sample(observable, [0], [])


class TestDefaultGaussianIntegration:
    """Integration tests for default.gaussian. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.gaussian", wires=1)
        cap = dev.target_device.capabilities()
        capabilities = {
            "model": "cv",
            "supports_finite_shots": True,
            "returns_probs": False,
            "returns_state": False,
            "supports_analytic_computation": True,
            "supports_broadcasting": False,
        }
        assert cap == capabilities

    def test_load_default_gaussian_device(self):
        """Test that the default plugin loads correctly"""

        dev = qml.device("default.gaussian", wires=2, hbar=2)
        assert dev.num_wires == 2
        assert dev.shots == qml.measurements.Shots(None)
        assert dev.hbar == 2
        assert dev.short_name == "default.gaussian"

    def test_args(self):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(TypeError, match="missing 1 required positional argument: 'wires'"):
            qml.device("default.gaussian")

    def test_gaussian_circuit(self, tol):
        """Test that the default gaussian plugin provides correct result for simple circuit"""
        dev = qml.device("default.gaussian", wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.QuadX(0))

        assert circuit(p) == pytest.approx(p * np.sqrt(2 * hbar), abs=tol)

    def test_gaussian_identity(self, tol):
        """Test that the default gaussian plugin provides correct result for the identity expectation"""
        dev = qml.device("default.gaussian", wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.Identity(0))

        assert circuit(p) == pytest.approx(1, abs=tol)

    def test_vacuum_x_squared_variance(self):
        """Test that variance of X^2 is correct for the vacuum

        The expected analytic expression of hbar^2/ 2 follows as:

        Var[X^2] = E[X^4] - E[X^2]

        We assume that we've prepared the vaccuum state.

        X = sqrt(hbar/2) * (a + a_{dagger}), where a and a_{dagger} are the
        ladder operators.

        Var[X^2] = <0|(sqrt(hbar/2) * (a + a_{dagger}))^ 4|0> -
        (<0|(sqrt(hbar/2) * (a + a_{dagger})) ^ 2|0>) ^ 2

        Let's label the two terms with (1) and (2).

        In (1), only (a*a*a_{dagger}*a_{dagger} + a*a_{dagger}*a*a_{dagger}),
        in (2), only (a*a_{dagger}) contributes (normalization excluded). The
        rest of the terms are zero.

        The a*a_{dagger}=Id+a_{dagger}*a equation is also used to yield
        Var[X^2] = hbar^2/2.
        """
        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PolyXP(np.diag([0, 1, 0]), wires=0))

        assert circuit() == pytest.approx(dev.hbar**2 / 2)

    def test_nonzero_shots(self, tol_stochastic):
        """Test that the default gaussian plugin provides correct result for high shot number"""

        shots = 10**4
        dev = qml.device("default.gaussian", wires=1)

        p = 0.543

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.QuadX(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.mean(runs) == pytest.approx(p * np.sqrt(2 * hbar), abs=tol_stochastic)

    def test_shot_list_warns(self):
        """Test that specifying a list of shots is unsupported for
        default.gaussian and emits a warning"""

        shots = [10, 10, 10]
        dev = qml.device("default.gaussian", wires=1)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit():
            """Test quantum function"""
            return qml.sample(qml.QuadX(0))

        with pytest.warns(
            UserWarning,
            match="Specifying a list of shots is only supported for QubitDevice based devices.",
        ):
            circuit()
        assert circuit._shots.total_shots == sum(shots)

    def test_new_return_type_error_multi_measurements(self):
        """Test that multiple measurements raise an error with the new return type."""
        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev)
        def circuit():
            """Test quantum function"""
            return qml.sample(qml.QuadX(0)), qml.expval(qml.QuadX(1))

        with pytest.raises(
            QuantumFunctionError,
            match="Default gaussian only support single measurements.",
        ):
            circuit()
