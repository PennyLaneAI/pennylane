# Copyright 2018 Xanadu Quantum Technologies Inc.

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
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log

from pennylane import numpy as np
from scipy.special import factorial as fac
from scipy.linalg import block_diag

from defaults import pennylane as qml, BaseTest

from pennylane.plugins.default_gaussian import fock_prob

from pennylane.plugins.default_gaussian import (rotation, squeezing, quadratic_phase,
                                              beamsplitter, two_mode_squeezing,
                                              controlled_addition, controlled_phase)
from pennylane.plugins.default_gaussian import (vacuum_state, coherent_state,
                                              squeezed_state, displaced_squeezed_state,
                                              thermal_state)

from pennylane.plugins.default_gaussian import DefaultGaussian


log.getLogger('defaults')


U = np.array([[0.83645892-0.40533293j, -0.20215326+0.30850569j],
              [-0.23889780-0.28101519j, -0.88031770-0.29832709j]])


U2 = np.array([[-0.07843244-3.57825948e-01j, 0.71447295-5.38069384e-02j, 0.20949966+6.59100734e-05j, -0.50297381+2.35731613e-01j],
               [-0.26626692+4.53837083e-01j, 0.27771991-2.40717436e-01j, 0.41228017-1.30198687e-01j, 0.01384490-6.33200028e-01j],
               [-0.69254712-2.56963068e-02j, -0.15484858+6.57298384e-02j, -0.53082141+7.18073414e-02j, -0.41060450-1.89462315e-01j],
               [-0.09686189-3.15085273e-01j, -0.53241387-1.99491763e-01j, 0.56928622+3.97704398e-01j, -0.28671074-6.01574497e-02j]])


H = np.array([[1.02789352, 1.61296440-0.3498192j],
              [1.61296440+0.3498192j, 1.23920938+0j]])


hbar = 2

def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == 'A':
        return [np.diag([x, 1]) for x in par]
    return par


class TestAuxillaryFunctions(BaseTest):
    """Tests the auxillary functions"""

    def setUp(self):
        self.hbar = 2.

        # an arbitrary two-mode Gaussian state generated using Strawberry Fields
        self.mu = np.array([0.6862, 0.4002, 0.09, 0.558])*np.sqrt(self.hbar)
        self.cov = np.array([[0.50750512, -0.04125979, -0.21058229, -0.07866912],
                             [-0.04125979, 0.50750512, -0.07866912, -0.21058229],
                             [-0.21058229, -0.07866912, 0.95906208, 0.27133391],
                             [-0.07866912, -0.21058229, 0.27133391, 0.95906208]])*self.hbar

        # expected Fock state probabilities
        self.events = [(0, 0), (0, 1), (1, 1), (2, 3)]
        self.probs = [0.430461524043, 0.163699407559, 0.0582788388927, 0.00167706931355]

    def test_fock_prob(self):
        """Test fock_prob returns the correct Fock probabilities"""
        for idx, e in enumerate(self.events):
            res = fock_prob(self.mu, self.cov, e, hbar=self.hbar)
            self.assertAlmostEqual(res, self.probs[idx], delta=self.tol)


class TestGates(BaseTest):
    """Gate tests."""

    def test_rotation(self):
        """Test the Fourier transform of a displaced state."""
        # pylint: disable=invalid-unary-operand-type
        self.logTestName()

        alpha = 0.23+0.12j
        S = rotation(np.pi/2)

        # apply to a coherent state. F{x, p} -> {-p, x}
        out = S @ np.array([alpha.real, alpha.imag])*np.sqrt(2*hbar)
        expected = np.array([-alpha.imag, alpha.real])*np.sqrt(2*hbar)
        self.assertAllAlmostEqual(out, expected, delta=self.tol)

    def test_squeezing(self):
        """Test the squeezing symplectic transform."""
        self.logTestName()

        r = 0.543
        phi = 0.123
        S = squeezing(r, phi)

        # apply to an identity covariance matrix
        out = S @ S.T
        expected = rotation(phi/2) @ np.diag(np.exp([-2*r, 2*r])) @ rotation(phi/2).T
        self.assertAllAlmostEqual(out, expected, delta=self.tol)

    def test_quadratic_phase(self):
        """Test the quadratic phase symplectic transform."""
        self.logTestName()

        s = 0.543
        S = quadratic_phase(s)

        # apply to a coherent state. P[x, p] -> [x, p+sx]
        alpha = 0.23+0.12j
        out = S @ np.array([alpha.real, alpha.imag])*np.sqrt(2*hbar)
        expected = np.array([alpha.real, alpha.imag+s*alpha.real])*np.sqrt(2*hbar)
        self.assertAllAlmostEqual(out, expected, delta=self.tol)

    def test_beamsplitter(self):
        """Test the beamsplitter symplectic transform."""
        self.logTestName()

        theta = 0.543
        phi = 0.312
        S = beamsplitter(theta, phi)

        # apply to a coherent state. BS|a1, a2> -> |ta1-r^*a2, ra1+ta2>
        a1 = 0.23+0.12j
        a2 = 0.23+0.12j
        out = S @ np.array([a1.real, a2.real, a1.imag, a2.imag])*np.sqrt(2*hbar)

        T = np.cos(theta)
        R = np.exp(1j*phi)*np.sin(theta)
        a1out = T*a1 - R.conj()*a2
        a2out = R*a2 + T*a1
        expected = np.array([a1out.real, a2out.real, a1out.imag, a2out.imag])*np.sqrt(2*hbar)
        self.assertAllAlmostEqual(out, expected, delta=self.tol)

    def test_two_mode_squeezing(self):
        """Test the two mode squeezing symplectic transform."""
        self.logTestName()

        r = 0.543
        phi = 0.123
        S = two_mode_squeezing(r, phi)

        # test that S = B^\dagger(pi/4, 0) [S(z) x S(-z)] B(pi/4)
        B = beamsplitter(np.pi/4, 0)
        Sz = block_diag(squeezing(r, phi), squeezing(-r, phi))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]
        expected = B.conj().T @ Sz @ B
        self.assertAllAlmostEqual(S, expected, delta=self.tol)

        # test that S |a1, a2> = |ta1+ra2, ta2+ra1>
        a1 = 0.23+0.12j
        a2 = 0.23+0.12j
        out = S @ np.array([a1.real, a2.real, a1.imag, a2.imag])*np.sqrt(2*hbar)

        T = np.cosh(r)
        R = np.exp(1j*phi)*np.sinh(r)
        a1out = T*a1 + R*np.conj(a2)
        a2out = T*a2 + R*np.conj(a1)
        expected = np.array([a1out.real, a2out.real, a1out.imag, a2out.imag])*np.sqrt(2*hbar)
        self.assertAllAlmostEqual(out, expected, delta=self.tol)

    def test_controlled_addition(self):
        """Test the CX symplectic transform."""
        self.logTestName()

        s = 0.543
        S = controlled_addition(s)

        # test that S = B(theta+pi/2, 0) [S(z) x S(-z)] B(theta, 0)
        r = np.arcsinh(-s/2)
        theta = 0.5*np.arctan2(-1/np.cosh(r), -np.tanh(r))
        Sz = block_diag(squeezing(r, 0), squeezing(-r, 0))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]

        expected = beamsplitter(theta+np.pi/2, 0) @ Sz @ beamsplitter(theta, 0)
        self.assertAllAlmostEqual(S, expected, delta=self.tol)

        # test that S[x1, x2, p1, p2] -> [x1, x2+sx1, p1-sp2, p2]
        x1 = 0.5432
        x2 = -0.453
        p1 = 0.154
        p2 = -0.123
        out = S @ np.array([x1, x2, p1, p2])*np.sqrt(2*hbar)
        expected = np.array([x1, x2+s*x1, p1-s*p2, p2])*np.sqrt(2*hbar)
        self.assertAllAlmostEqual(out, expected, delta=self.tol)

    def test_controlled_phase(self):
        """Test the CZ symplectic transform."""
        self.logTestName()

        s = 0.543
        S = controlled_phase(s)

        # test that S = R_2(pi/2) CX(s) R_2(pi/2)^\dagger
        R2 = block_diag(np.identity(2), rotation(np.pi/2))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]
        expected = R2 @ controlled_addition(s) @ R2.conj().T
        self.assertAllAlmostEqual(S, expected, delta=self.tol)

        # test that S[x1, x2, p1, p2] -> [x1, x2, p1+sx2, p2+sx1]
        x1 = 0.5432
        x2 = -0.453
        p1 = 0.154
        p2 = -0.123
        out = S @ np.array([x1, x2, p1, p2])*np.sqrt(2*hbar)
        expected = np.array([x1, x2, p1+s*x2, p2+s*x1])*np.sqrt(2*hbar)
        self.assertAllAlmostEqual(out, expected, delta=self.tol)


class TestStates(BaseTest):
    """State tests."""

    def test_vacuum_state(self):
        """Test the vacuum state is correct."""
        self.logTestName()
        wires = 3
        means, cov = vacuum_state(wires, hbar=hbar)
        self.assertAllAlmostEqual(means, np.zeros([2*wires]), delta=self.tol)
        self.assertAllAlmostEqual(cov, np.identity(2*wires)*hbar/2, delta=self.tol)

    def test_coherent_state(self):
        """Test the coherent state is correct."""
        self.logTestName()
        a = 0.432-0.123j
        means, cov = coherent_state(a, hbar=hbar)
        self.assertAllAlmostEqual(means, np.array([a.real, a.imag])*np.sqrt(2*hbar), delta=self.tol)
        self.assertAllAlmostEqual(cov, np.identity(2)*hbar/2, delta=self.tol)

    def test_squeezed_state(self):
        """Test the squeezed state is correct."""
        self.logTestName()
        r = 0.432
        phi = 0.123
        means, cov = squeezed_state(r, phi, hbar=hbar)

        # test vector of means is zero
        self.assertAllAlmostEqual(means, np.zeros([2]), delta=self.tol)

        R = rotation(phi/2)
        expected = R @ np.array([[np.exp(-2*r), 0],
                                 [0, np.exp(2*r)]]) * hbar/2 @ R.T
        # test covariance matrix is correct
        self.assertAllAlmostEqual(cov, expected, delta=self.tol)

    def test_displaced_squeezed_state(self):
        """Test the displaced squeezed state is correct."""
        self.logTestName()
        a = 0.541+0.109j
        r = 0.432
        phi = 0.123
        means, cov = displaced_squeezed_state(a, r, phi, hbar=hbar)

        # test vector of means is correct
        self.assertAllAlmostEqual(means, np.array([a.real, a.imag])*np.sqrt(2*hbar), delta=self.tol)

        R = rotation(phi/2)
        expected = R @ np.array([[np.exp(-2*r), 0],
                                 [0, np.exp(2*r)]]) * hbar/2 @ R.T
        # test covariance matrix is correct
        self.assertAllAlmostEqual(cov, expected, delta=self.tol)

    def thermal_state(self):
        """Test the thermal state is correct."""
        self.logTestName()
        nbar = 0.5342
        means, cov = thermal_state(nbar, hbar=hbar)
        self.assertAllAlmostEqual(means, np.zeros([2]), delta=self.tol)
        self.assertTrue(np.all((cov.diag*2/hbar-1)/2 == nbar))



class TestDefaultGaussianDevice(BaseTest):
    """Test the default gaussian device. The test ensures that the device is properly
    applying gaussian operations and calculating the correct observables."""
    def setUp(self):
        self.dev = DefaultGaussian(wires=2, shots=0, hbar=hbar)

    def test_operation_map(self):
        """Test that default Gaussian device supports all PennyLane Gaussian CV gates."""
        self.logTestName()

        nonGaussian = {'FockDensityMatrix',
                       'FockStateVector',
                       'FockState',
                       'CrossKerr',
                       'CatState',
                       'CubicPhase',
                       'Kerr'}

        self.assertEqual(set(qml.ops.cv.__all__) - nonGaussian,
                         set(self.dev._operation_map))

    def test_expectation_map(self):
        """Test that default Gaussian device supports all PennyLane Gaussian continuous expectations."""
        self.logTestName()
        self.assertEqual(set(qml.expval.cv.__all__)-{'Heterodyne'},
                         set(self.dev._expectation_map))

    def test_apply(self):
        """Test the application of gates to a state"""
        self.logTestName()

        # loop through all supported operations
        for gate_name, fn in self.dev._operation_map.items():
            log.debug("\tTesting %s gate...", gate_name)
            self.dev.reset()

            # start in the displaced squeezed state
            a = 0.542+0.123j
            r = 0.652
            phi = -0.124

            self.dev.apply('DisplacedSqueezedState', wires=[0], par=[a, r, phi])
            self.dev.apply('DisplacedSqueezedState', wires=[1], par=[a, r, phi])

            # get the equivalent pennylane operation class
            op = qml.ops.__getattribute__(gate_name)
            # the list of wires to apply the operation to
            w = list(range(op.num_wires))

            if op.par_domain == 'A':
                # the parameter is an array
                if gate_name == 'GaussianState':
                    p = [np.array([0.432, 0.123, 0.342, 0.123]), np.diag([0.5234]*4)]
                    w = list(range(2))
                    expected_out = p
            else:
                # the parameter is a float
                p = [0.432423, -0.12312, 0.324][:op.num_params]

                if gate_name == 'Displacement':
                    alpha = p[0]*np.exp(1j*p[1])
                    state = self.dev._state
                    mu = state[0].copy()
                    mu[w[0]] += alpha.real*np.sqrt(2*hbar)
                    mu[w[0]+2] += alpha.imag*np.sqrt(2*hbar)
                    expected_out = mu, state[1]
                elif 'State' in gate_name:
                    mu, cov = fn(*p, hbar=hbar)
                    expected_out = self.dev._state
                    expected_out[0][[w[0], w[0]+2]] = mu

                    ind = np.concatenate([np.array([w[0]]), np.array([w[0]])+2])
                    rows = ind.reshape(-1, 1)
                    cols = ind.reshape(1, -1)
                    expected_out[1][rows, cols] = cov
                else:
                    # if the default.gaussian is an operation accepting parameters,
                    # initialise it using the parameters generated above.
                    S = fn(*p)

                    # calculate the expected output
                    if op.num_wires == 1:
                        # reorder from symmetric ordering to xp-ordering
                        S = block_diag(S, np.identity(2))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]

                    expected_out = S @ self.dev._state[0], S @ self.dev._state[1] @ S.T

            self.dev.apply(gate_name, wires=w, par=p)

            # verify the device is now in the expected state
            self.assertAllAlmostEqual(self.dev._state[0], expected_out[0], delta=self.tol)
            self.assertAllAlmostEqual(self.dev._state[1], expected_out[1], delta=self.tol)

    def test_apply_errors(self):
        """Test that apply fails for incorrect state preparation"""
        self.logTestName()

        with self.assertRaisesRegex(ValueError, 'incorrect size for the number of subsystems'):
            p = [thermal_state(0.5)]
            self.dev.apply('GaussianState', wires=[0], par=[p])

    def test_expectation(self):
        """Test that expectation values are calculated correctly"""
        self.logTestName()

        dev = qml.device('default.gaussian', wires=1, hbar=hbar)

        # test correct mean for <n> of a displaced thermal state
        nbar = 0.5431
        alpha = 0.324-0.59j
        dev.apply('ThermalState', wires=[0], par=[nbar])
        dev.apply('Displacement', wires=[0], par=[alpha, 0])
        mean = dev.expval('MeanPhoton', [0], [])
        self.assertAlmostEqual(mean, np.abs(alpha)**2+nbar, delta=self.tol)

        # test correct mean for Homodyne P measurement
        alpha = 0.324-0.59j
        dev.apply('CoherentState', wires=[0], par=[alpha])
        mean = dev.expval('P', [0], [])
        self.assertAlmostEqual(mean, alpha.imag*np.sqrt(2*hbar), delta=self.tol)

        # test correct mean for Homodyne measurement
        mean = dev.expval('Homodyne', [0], [np.pi/2])
        self.assertAlmostEqual(mean, alpha.imag*np.sqrt(2*hbar), delta=self.tol)

        # test correct mean for number state expectation |<n|alpha>|^2
        # on a coherent state
        for n in range(3):
            mean = dev.expval('NumberState', [0], [np.array([n])])
            expected = np.abs(np.exp(-np.abs(alpha)**2/2)*alpha**n/np.sqrt(fac(n)))**2
            self.assertAlmostEqual(mean, expected, delta=self.tol)

        # test correct mean for number state expectation |<n|S(r)>|^2
        # on a squeezed state
        n = 1
        r = 0.4523
        dev.apply('SqueezedState', wires=[0], par=[r, 0])
        mean = dev.expval('NumberState', [0], [np.array([2*n])])
        expected = np.abs(np.sqrt(fac(2*n))/(2**n*fac(n))*(-np.tanh(r))**n/np.sqrt(np.cosh(r)))**2
        self.assertAlmostEqual(mean, expected, delta=self.tol)

    def test_var(self):
        """Test that variance values are calculated correctly"""
        self.logTestName()

        dev = qml.device('default.gaussian', wires=1, hbar=hbar)

        # test correct variance for <n> of a displaced thermal state
        nbar = 0.5431
        alpha = 0.324-0.59j
        dev.apply('ThermalState', wires=[0], par=[nbar])
        dev.apply('Displacement', wires=[0], par=[alpha, 0])

        var = dev.var('MeanPhoton', [0], [])
        self.assertAlmostEqual(var, nbar**2+nbar+np.abs(alpha)**2*(1+2*nbar), delta=self.tol)

        # test the same quantity but using PolyXP
        var = dev.var('PolyXP', [0], [np.array([[-0.5, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])])
        self.assertAlmostEqual(var, nbar**2+nbar+np.abs(alpha)**2*(1+2*nbar), delta=self.tol)

        # test correct variance for Homodyne P measurement
        alpha = 0.324-0.59j
        dev.apply('CoherentState', wires=[0], par=[alpha])
        var = dev.var('P', [0], [])
        self.assertAlmostEqual(var, hbar/2, delta=self.tol)

        # test correct variance for Homodyne measurement
        var = dev.var('Homodyne', [0], [np.pi/2])
        self.assertAlmostEqual(var, hbar/2, delta=self.tol)

        # test correct variance for number state expectation |<n|alpha>|^2
        # on a coherent state
        for n in range(3):
            var = dev.var('NumberState', [0], [np.array([n])])
            mean = np.abs(np.exp(-np.abs(alpha)**2/2)*alpha**n/np.sqrt(fac(n)))**2
            expected = mean - mean**2
            self.assertAlmostEqual(var, expected, delta=self.tol)

        # test correct mean and variance for number state expectation |<n|S(r)>|^2
        # on a squeezed state
        n = 1
        r = 0.4523
        dev.apply('SqueezedState', wires=[0], par=[r, 0])
        var = dev.var('NumberState', [0], [np.array([2*n])])
        mean = np.abs(np.sqrt(fac(2*n))/(2**n*fac(n))*(-np.tanh(r))**n/np.sqrt(np.cosh(r)))**2
        expected = mean - mean**2
        self.assertAlmostEqual(var, expected, delta=self.tol)

    def test_reduced_state(self):
        """Test reduced state"""
        self.logTestName()

        # Test error is raised if requesting a non-existant subsystem
        with self.assertRaisesRegex(ValueError, "specified wires cannot be larger than the number of subsystems"):
            self.dev.reduced_state([6, 4])

        # Test requesting via an integer
        res = self.dev.reduced_state(0)
        expected = self.dev.reduced_state([0])
        self.assertAllAlmostEqual(res[0], expected[0], delta=self.tol)
        self.assertAllAlmostEqual(res[1], expected[1], delta=self.tol)

        # Test requesting all wires returns the full state
        res = self.dev.reduced_state([0, 1])
        expected = self.dev._state
        self.assertAllAlmostEqual(res[0], expected[0], delta=self.tol)
        self.assertAllAlmostEqual(res[1], expected[1], delta=self.tol)


class TestDefaultGaussianIntegration(BaseTest):
    """Integration tests for default.gaussian. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_gaussian_device(self):
        """Test that the default plugin loads correctly"""
        self.logTestName()

        dev = qml.device('default.gaussian', wires=2, hbar=2)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 0)
        self.assertEqual(dev.hbar, 2)
        self.assertEqual(dev.short_name, 'default.gaussian')

    def test_args(self):
        """Test that the plugin requires correct arguments"""
        self.logTestName()

        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'wires'"):
            qml.device('default.gaussian')

    def test_unsupported_gates(self):
        """Test error is raised with unsupported gates"""
        self.logTestName()
        dev = qml.device('default.gaussian', wires=2)

        gates = set(dev._operation_map.keys())
        all_gates = {m[0] for m in inspect.getmembers(qml.ops, inspect.isclass)}

        for g in all_gates - gates:
            op = getattr(qml.ops, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Test quantum function"""
                x = prep_par(x, op)
                op(*x, wires=wires)

                if issubclass(op, qml.operation.CV):
                    return qml.expval.X(0)

                return qml.expval.PauliZ(0)

            with self.assertRaisesRegex(qml.DeviceError, "Gate {} not supported on device default.gaussian".format(g)):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_unsupported_observables(self):
        """Test error is raised with unsupported observables"""
        self.logTestName()
        dev = qml.device('default.gaussian', wires=2)

        obs = set(dev._expectation_map.keys())
        all_obs = {m[0] for m in inspect.getmembers(qml.expval, inspect.isclass)}

        for g in all_obs - obs:
            op = getattr(qml.expval, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Test quantum function"""
                x = prep_par(x, op)
                return op(*x, wires=wires)

            with self.assertRaisesRegex(qml.DeviceError, "Expectation {} not supported on device default.gaussian".format(g)):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_gaussian_circuit(self):
        """Test that the default gaussian plugin provides correct result for simple circuit"""
        self.logTestName()
        dev = qml.device('default.gaussian', wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.Displacement(x, 0, wires=0)
            return qml.expval.X(0)

        self.assertAlmostEqual(circuit(p), p*np.sqrt(2*hbar), delta=self.tol)

    def test_nonzero_shots(self):
        """Test that the default gaussian plugin provides correct result for high shot number"""
        self.logTestName()

        shots = 10**4
        dev = qml.device('default.gaussian', wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.Displacement(x, 0, wires=0)
            return qml.expval.X(0)

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        self.assertAlmostEqual(np.mean(runs), p*np.sqrt(2*hbar), delta=0.01)

    def test_supported_gates(self):
        """Test that all supported gates work correctly"""
        self.logTestName()
        a = 0.312

        dev = qml.device('default.gaussian', wires=2)

        for g, qop in dev._operation_map.items():
            log.debug('\tTesting gate %s...', g)
            self.assertTrue(dev.supported(g))
            dev.reset()

            op = getattr(qml.ops, g)
            if op.num_wires == 0:
                wires = list(range(2))
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Reference quantum function"""
                qml.Displacement(a, 0, wires=[0])
                op(*x, wires=wires)
                return qml.expval.X(0)

            # compare to reference result
            def reference(*x):
                """reference circuit"""
                if g == 'GaussianState':
                    return x[0][0]

                if g == 'Displacement':
                    alpha = x[0]*np.exp(1j*x[1])
                    return (alpha+a).real*np.sqrt(2*hbar)

                if 'State' in g:
                    mu, _ = qop(*x, hbar=hbar)
                    return mu[0]

                S = qop(*x)

                # calculate the expected output
                if op.num_wires == 1:
                    S = block_diag(S, np.identity(2))[:, [0, 2, 1, 3]][[0, 2, 1, 3]]

                return (S @ np.array([a.real, a.imag, 0, 0])*np.sqrt(2*hbar))[0]

            if g == 'GaussianState':
                p = [np.array([0.432, 0.123, 0.342, 0.123]), np.diag([0.5234]*4)]
            else:
                p = [0.432423, -0.12312, 0.324][:op.num_params]

            self.assertAllEqual(circuit(*p), reference(*p))


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', default.gaussian plugin.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestAuxillaryFunctions,
              TestGates,
              TestStates,
              TestDefaultGaussianDevice,
              TestDefaultGaussianIntegration):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)
