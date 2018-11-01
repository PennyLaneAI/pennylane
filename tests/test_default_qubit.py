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
Unit tests for the :mod:`pennylane.plugin.DefaultQubit` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
import unittest
import inspect
import logging as log

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest
from pennylane.plugins.default_qubit import (spectral_decomposition_qubit,
                                     I, X, Z, CNOT, Rphi, frx, fry, frz, fr3,
                                     unitary, hermitian, DefaultQubit)

log.getLogger('defaults')


U = np.array([[0.83645892-0.40533293j, -0.20215326+0.30850569j],
              [-0.23889780-0.28101519j, -0.88031770-0.29832709j]])


U2 = np.array([[-0.07843244-3.57825948e-01j, 0.71447295-5.38069384e-02j, 0.20949966+6.59100734e-05j, -0.50297381+2.35731613e-01j],
               [-0.26626692+4.53837083e-01j, 0.27771991-2.40717436e-01j, 0.41228017-1.30198687e-01j, 0.01384490-6.33200028e-01j],
               [-0.69254712-2.56963068e-02j, -0.15484858+6.57298384e-02j, -0.53082141+7.18073414e-02j, -0.41060450-1.89462315e-01j],
               [-0.09686189-3.15085273e-01j, -0.53241387-1.99491763e-01j, 0.56928622+3.97704398e-01j, -0.28671074-6.01574497e-02j]])


H = np.array([[1.02789352, 1.61296440-0.3498192j],
              [1.61296440+0.3498192j, 1.23920938+0j]])


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == 'A':
        return [np.diag([x, 1]) for x in par]
    return par


class TestAuxillaryFunctions(BaseTest):
    """Test auxillary functions."""

    def test_spectral_decomposition_qubit(self):
        """Test that the correct spectral decomposition is returned."""
        self.logTestName()

        a, P = spectral_decomposition_qubit(H)

        # verify that H = \sum_k a_k P_k
        self.assertAllAlmostEqual(H, np.einsum('i,ijk->jk', a, P), delta=self.tol)

    def test_phase_shift(self):
        """Test phase shift is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(Rphi(0), np.identity(2), delta=self.tol)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j*phi)]])
        self.assertAllAlmostEqual(Rphi(phi), expected, delta=self.tol)

    def test_x_rotation(self):
        """Test x rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(frx(0), np.identity(2), delta=self.tol)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]])/np.sqrt(2)
        self.assertAllAlmostEqual(frx(np.pi/2), expected, delta=self.tol)

        # test identity for theta=pi
        expected = -1j*np.array([[0, 1], [1, 0]])
        self.assertAllAlmostEqual(frx(np.pi), expected, delta=self.tol)

    def test_y_rotation(self):
        """Test y rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(fry(0), np.identity(2), delta=self.tol)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]])/np.sqrt(2)
        self.assertAllAlmostEqual(fry(np.pi/2), expected, delta=self.tol)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        self.assertAllAlmostEqual(fry(np.pi), expected, delta=self.tol)

    def test_z_rotation(self):
        """Test z rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(frz(0), np.identity(2), delta=self.tol)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j*np.pi/4, 1j*np.pi/4]))
        self.assertAllAlmostEqual(frz(np.pi/2), expected, delta=self.tol)

        # test identity for theta=pi
        self.assertAllAlmostEqual(frz(np.pi), -1j*Z, delta=self.tol)

    def test_arbitrary_rotation(self):
        """Test arbitrary single qubit rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(fr3(0, 0, 0), np.identity(2), delta=self.tol)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y/2)
            s = np.sin(y/2)
            return np.array([[np.exp(-0.5j*(x+z))*c, -np.exp(0.5j*(x-z))*s],
                             [np.exp(-0.5j*(x-z))*s, np.exp(0.5j*(x+z))*c]])

        a, b, c = 0.432, -0.152, 0.9234
        self.assertAllAlmostEqual(fr3(a, b, c), arbitrary_rotation(a, b, c), delta=self.tol)


class TestStateFunctions(BaseTest):
    """Arbitrary state and operator tests."""

    def test_unitary(self):
        """Test that the unitary function produces the correct output."""
        self.logTestName()

        out = unitary(U)

        # verify output type
        self.assertTrue(isinstance(out, np.ndarray))

        # verify equivalent to input state
        self.assertAllAlmostEqual(out, U, delta=self.tol)

        # test non-square matrix
        with self.assertRaisesRegex(ValueError, "must be a square matrix"):
            unitary(U[1:])

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "must be unitary"):
            unitary(U3)

    def test_hermitian(self):
        """Test that the Hermitian function produces the correct output."""
        self.logTestName()

        out = hermitian(H)

        # verify output type
        self.assertTrue(isinstance(out, np.ndarray))

        # verify equivalent to input state
        self.assertAllAlmostEqual(out, H, delta=self.tol)

        # test non-square matrix
        with self.assertRaisesRegex(ValueError, "must be a square matrix"):
            hermitian(H[1:])

        # test non-Hermitian matrix
        H2 = H.copy()
        H2[0, 1] = H2[0, 1].conj()
        with self.assertRaisesRegex(ValueError, "must be Hermitian"):
            hermitian(H2)


class TestDefaultQubitDevice(BaseTest):
    """Test the default qubit device. The test ensures that the device is properly
    applying qubit operations and calculating the correct observables."""
    def setUp(self):
        self.dev = DefaultQubit(wires=2, shots=0)

    def test_operation_map(self):
        """Test that default qubit device supports all PennyLane discrete gates."""
        self.logTestName()

        self.assertEqual(set(qml.ops.qubit.__all__), set(self.dev._operation_map))

    def test_expectation_map(self):
        """Test that default qubit device supports all PennyLane discrete expectations."""
        self.logTestName()

        self.assertEqual(set(qml.expval.qubit.__all__), set(self.dev._expectation_map))

    def test_expand_one(self):
        """Test that a 1 qubit gate correctly expands to 3 qubits."""
        self.logTestName()

        dev = DefaultQubit(wires=3)

        # test applied to wire 0
        res = dev.expand_one(U, [0])
        expected = np.kron(np.kron(U, I), I)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # test applied to wire 1
        res = dev.expand_one(U, [1])
        expected = np.kron(np.kron(I, U), I)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # test applied to wire 2
        res = dev.expand_one(U, [2])
        expected = np.kron(np.kron(I, I), U)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # test exception raised if U is not 2x2 matrix
        with self.assertRaisesRegex(ValueError, "2x2 matrix required"):
            dev.expand_one(U2, [0])

        # test exception raised if more than one subsystem provided
        with self.assertRaisesRegex(ValueError, "One target subsystem required"):
            dev.expand_one(U, [0, 1])

    def test_expand_two(self):
        """Test that a 2 qubit gate correctly expands to 3 qubits."""
        self.logTestName()

        dev = DefaultQubit(wires=4)

        # test applied to wire 0+1
        res = dev.expand_two(U2, [0, 1])
        expected = np.kron(np.kron(U2, I), I)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # test applied to wire 1+2
        res = dev.expand_two(U2, [1, 2])
        expected = np.kron(np.kron(I, U2), I)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # test applied to wire 2+3
        res = dev.expand_two(U2, [2, 3])
        expected = np.kron(np.kron(I, I), U2)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # CNOT with target on wire 1
        res = dev.expand_two(CNOT, [1, 0])
        rows = np.array([0, 2, 1, 3])
        expected = np.kron(np.kron(CNOT[:, rows][rows], I), I)
        self.assertAllAlmostEqual(res, expected, delta=self.tol)

        # test exception raised if U is not 4x4 matrix
        with self.assertRaisesRegex(ValueError, "4x4 matrix required"):
            dev.expand_two(U, [0])

        # test exception raised if two subsystems are not provided
        with self.assertRaisesRegex(ValueError, "Two target subsystems required"):
            dev.expand_two(U2, [0])

        # test exception raised if unphysical subsystems provided
        with self.assertRaisesRegex(ValueError, "Bad target subsystems."):
            dev.expand_two(U2, [-1, 5])

    def test_get_operator_matrix(self):
        """Test the the correct matrix is returned given an operation name"""
        self.logTestName()

        for name, fn in {**self.dev._operation_map, **self.dev._expectation_map}.items():
            try:
                op = qml.ops.__getattribute__(name)
            except AttributeError:
                op = qml.expval.__getattribute__(name)

            p = [0.432423, -0.12312, 0.324][:op.num_params]

            if name == 'QubitStateVector':
                p = [np.array([1, 0, 0, 0])]
            elif name == 'QubitUnitary':
                p = [U]
            elif name == 'Hermitian':
                p = [H]

            res = self.dev._get_operator_matrix(name, p)

            if callable(fn):
                # if the default.qubit is an operation accepting parameters,
                # initialise it using the parameters generated above.
                expected = fn(*p)
            else:
                # otherwise, the operation is simply an array.
                expected = fn

            self.assertAllAlmostEqual(res, expected, delta=self.tol)

    def test_apply(self):
        """Test the application of gates to a state"""
        self.logTestName()
        self.dev.reset()

        # loop through all supported operations
        for gate_name, fn in self.dev._operation_map.items():
            log.debug("\tTesting %s gate...", gate_name)

            # start in the state |00>
            self.dev._state = np.array([1, 0, 0, 0])

            # get the equivalent pennylane operation class
            op = qml.ops.__getattribute__(gate_name)
            # the list of wires to apply the operation to
            w = list(range(op.num_wires))

            if op.par_domain == 'A':
                # the parameter is an array
                if gate_name == 'QubitStateVector':
                    p = [np.array([1, 0, 1, 1])/np.sqrt(3)]
                    expected_out = p
                elif gate_name == 'QubitUnitary':
                    p = [U]
                    w = [0]
                    expected_out = np.kron(U @ np.array([1, 0]), np.array([1, 0]))
                elif gate_name == 'BasisState':
                    p = [np.array([1, 1])]
                    expected_out = np.array([0, 0, 0, 1])

            elif op.par_domain == 'N':
                # the parameter is an integer
                p = [1, 3, 4][:op.num_params]
            else:
                # the parameter is a float
                p = [0.432423, -0.12312, 0.324][:op.num_params]

                if callable(fn):
                    # if the default.qubit is an operation accepting parameters,
                    # initialise it using the parameters generated above.
                    O = fn(*p)
                else:
                    # otherwise, the operation is simply an array.
                    O = fn

                # calculate the expected output
                if op.num_wires == 1:
                    expected_out = np.kron(O @ np.array([1, 0]), np.array([1, 0]))
                elif op.num_wires == 2:
                    expected_out = O @ self.dev._state

            self.dev.apply(gate_name, wires=w, par=p)

            # verify the device is now in the expected state
            self.assertAllAlmostEqual(self.dev._state, expected_out, delta=self.tol)

    def test_apply_errors(self):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""
        self.logTestName()

        with self.assertRaisesRegex(ValueError, r'State vector must be of length 2\*\*wires.'):
            p = [np.array([1, 0, 1, 1, 1])/np.sqrt(3)]
            self.dev.apply('QubitStateVector', wires=[0, 1], par=[p])

        with self.assertRaisesRegex(ValueError, "BasisState parameter must be an array of 0/1 integers"):
            self.dev.apply('BasisState', wires=[0, 1], par=[np.array([0, 1, 4.2])])

        with self.assertRaisesRegex(ValueError, "This plugin supports only one- and two-qubit gates."):
            self.dev.apply('QubitUnitary', wires=[0, 1, 2], par=[U2])

    def test_ev(self):
        """Test that expectation values are calculated correctly"""
        self.logTestName()
        self.dev.reset()

        # loop through all supported observables
        for name, fn in self.dev._expectation_map.items():
            log.debug("\tTesting %s observable...", name)

            # start in the state |00>
            self.dev._state = np.array([1, 0, 1, 1])/np.sqrt(3)

            # get the equivalent pennylane operation class
            op = qml.expval.__getattribute__(name)

            if op.par_domain == 'A':
                # the parameter is an array
                p = [H]
            else:
                # the parameter is a float
                p = [0.432423, -0.12312, 0.324][:op.num_params]

            if callable(fn):
                # if the default.qubit is an operation accepting parameters,
                # initialise it using the parameters generated above.
                O = fn(*p)
            else:
                # otherwise, the operation is simply an array.
                O = fn

            # calculate the expected output
            if op.num_wires == 1:
                expected_out = self.dev._state.conj() @ np.kron(O, I) @ self.dev._state
            elif op.num_wires == 2:
                expected_out = self.dev._state.conj() @ O @ self.dev._state

            res = self.dev.ev(O, wires=[0])

            # verify the device is now in the expected state
            self.assertAllAlmostEqual(res, expected_out, delta=self.tol)

            # text exception raised if matrix is not 2x2
            with self.assertRaisesRegex(ValueError, "2x2 matrix required"):
                self.dev.ev(U2, [0])

            # text warning raised if matrix is complex
            with self.assertLogs(level='WARNING') as l:
                self.dev.ev(H+1j, [0])
                self.assertEqual(len(l.output), 1)
                self.assertEqual(len(l.records), 1)
                self.assertIn('Nonvanishing imaginary part', l.output[0])

    def test_var(self):
        """Test that the correct variances are returned"""
        self.logTestName()

        # variance for sigma_z on eigenstate {1, 0} should be zero
        @qml.qnode(self.dev)
        def qfunc():
            return qml.expval.PauliZ(0)

        self.assertAllAlmostEqual(qfunc.var(), 0, delta=self.tol)

        # variance for sigma_y on Rx(x)|0> is cos^2(x)
        @qml.qnode(self.dev)
        def qfunc(x):
            qml.RX(x, wires=0)
            return qml.expval.PauliY(0)

        x = 0.5432
        self.assertAllAlmostEqual(qfunc.var(x), np.cos(x)**2, delta=self.tol)

        # variance for sigma_z on Rx(x)|0> is sin^2(x)
        @qml.qnode(self.dev)
        def qfunc(x):
            qml.RX(x, wires=0)
            return qml.expval.PauliZ(0)

        self.assertAllAlmostEqual(qfunc.var(x), np.sin(x)**2, delta=self.tol)

        # text exception raised if matrix is not 2x2
        with self.assertRaisesRegex(ValueError, "2x2 matrix required"):
            print('here')
            self.dev.var('Hermitian', [0], [np.identity(4)])


class TestDefaultQubitIntegration(BaseTest):
    """Integration tests for default.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""
        self.logTestName()

        dev = qml.device('default.qubit', wires=2)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 0)
        self.assertEqual(dev.short_name, 'default.qubit')

    def test_args(self):
        """Test that the plugin requires correct arguments"""
        self.logTestName()

        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'wires'"):
            qml.device('default.qubit')

    def test_unsupported_gates(self):
        """Test error is raised with unsupported gates"""
        self.logTestName()
        dev = qml.device('default.qubit', wires=2)

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

            with self.assertRaisesRegex(qml.DeviceError, "Gate {} not supported on device default.qubit".format(g)):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_unsupported_observables(self):
        """Test error is raised with unsupported observables"""
        self.logTestName()
        dev = qml.device('default.qubit', wires=2)

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

            with self.assertRaisesRegex(qml.DeviceError, "Expectation {} not supported on device default.qubit".format(g)):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_qubit_circuit(self):
        """Test that the default qubit plugin provides correct result for simple circuit"""
        self.logTestName()
        dev = qml.device('default.qubit', wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval.PauliY(0)

        # <0|RX(p)^\dagger.PauliY.RX(p)|0> = -sin(p)
        expected = -np.sin(p)
        self.assertAlmostEqual(circuit(p), expected, delta=self.tol)

    def test_nonzero_shots(self):
        """Test that the default qubit plugin provides correct result for high shot number"""
        self.logTestName()

        shots = 10**4
        dev = qml.device('default.qubit', wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval.PauliY(0)

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        self.assertAlmostEqual(np.mean(runs), -np.sin(p), delta=0.01)

    def test_supported_gates(self):
        """Test that all supported gates work correctly"""
        self.logTestName()
        a = 0.312
        b = 0.123

        dev = qml.device('default.qubit', wires=2)

        for g, qop in dev._operation_map.items():
            log.debug('\tTesting gate %s...', g)
            self.assertTrue(dev.supported(g))
            dev.reset()

            op = getattr(qml.ops, g)
            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Reference quantum function"""
                op(*x, wires=wires)
                return qml.expval.PauliX(0)

            # compare to reference result
            def reference(*x):
                """reference circuit"""
                if callable(qop):
                    # if the default.qubit is an operation accepting parameters,
                    # initialise it using the parameters generated above.
                    O = qop(*x)
                else:
                    # otherwise, the operation is simply an array.
                    O = qop

                # calculate the expected output
                if op.num_wires == 1 or g == 'QubitUnitary':
                    out_state = np.kron(O @ np.array([1, 0]), np.array([1, 0]))
                elif g == 'QubitStateVector':
                    out_state = x[0]
                elif g == 'BasisState':
                    out_state = np.array([0, 0, 0, 1])
                else:
                    out_state = O @ dev._state

                expectation = out_state.conj() @ np.kron(X, np.identity(2)) @ out_state
                return expectation

            if g == 'QubitStateVector':
                p = np.array([1, 0, 0, 1])/np.sqrt(2)
                self.assertAllEqual(circuit(p), reference(p))
            elif g == 'BasisState':
                p = np.array([1, 1])
                self.assertAllEqual(circuit(p), reference(p))
            elif g == 'QubitUnitary':
                self.assertAllEqual(circuit(U), reference(U))
            elif op.num_params == 1:
                self.assertAllEqual(circuit(a), reference(a))
            elif op.num_params == 2:
                self.assertAllEqual(circuit(a, b), reference(a, b))

    def test_supported_observables(self):
        """Test that all supported observables work correctly"""
        self.logTestName()
        a = 0.312

        dev = qml.device('default.qubit', wires=2)

        for g, qop in dev._expectation_map.items():
            log.debug('\tTesting observable %s...', g)
            self.assertTrue(dev.supported(g))
            dev.reset()

            op = getattr(qml.expval, g)
            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Reference quantum function"""
                qml.RX(a, wires=0)
                return op(*x, wires=wires)

            # compare to reference result
            def reference(*x):
                """reference circuit"""
                if callable(qop):
                    # if the default.qubit is an operation accepting parameters,
                    # initialise it using the parameters generated above.
                    O = qop(*x)
                else:
                    # otherwise, the operation is simply an array.
                    O = qop

                # calculate the expected output
                out_state = np.kron(frx(a) @ np.array([1, 0]), np.array([1, 0]))
                expectation = out_state.conj() @ np.kron(O, np.identity(2)) @ out_state
                return expectation

            if op.num_params == 0:
                self.assertAllEqual(circuit(), reference())
            elif g == 'Hermitian':
                self.assertAllEqual(circuit(H), reference(H))


if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', default.qubit plugin.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestAuxillaryFunctions,
              TestStateFunctions,
              TestDefaultQubitDevice,
              TestDefaultQubitIntegration):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)
