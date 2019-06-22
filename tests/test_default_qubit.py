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
import logging as log

import pytest

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest
from pennylane.plugins.default_qubit import (
    spectral_decomposition_qubit,
    I,
    X,
    Z,
    CNOT,
    Rphi,
    Rotx,
    Roty,
    Rotz,
    Rot3,
    unitary,
    hermitian,
    DefaultQubit,
)

log.getLogger("defaults")


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


U_toffoli = np.diag([1 for i in range(8)])
U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])

U_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


H = np.array(
    [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
)


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == "A":
        return [np.diag([x, 1]) for x in par]
    return par


class TestAuxillaryFunctions(BaseTest):
    """Test auxillary functions."""

    def test_spectral_decomposition_qubit(self):
        """Test that the correct spectral decomposition is returned."""
        self.logTestName()

        a, P = spectral_decomposition_qubit(H)

        # verify that H = \sum_k a_k P_k
        self.assertAllAlmostEqual(H, np.einsum("i,ijk->jk", a, P), delta=self.tol)

    def test_phase_shift(self):
        """Test phase shift is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(Rphi(0), np.identity(2), delta=self.tol)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        self.assertAllAlmostEqual(Rphi(phi), expected, delta=self.tol)

    def test_x_rotation(self):
        """Test x rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(Rotx(0), np.identity(2), delta=self.tol)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        self.assertAllAlmostEqual(Rotx(np.pi / 2), expected, delta=self.tol)

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1], [1, 0]])
        self.assertAllAlmostEqual(Rotx(np.pi), expected, delta=self.tol)

    def test_y_rotation(self):
        """Test y rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(Roty(0), np.identity(2), delta=self.tol)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        self.assertAllAlmostEqual(Roty(np.pi / 2), expected, delta=self.tol)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        self.assertAllAlmostEqual(Roty(np.pi), expected, delta=self.tol)

    def test_z_rotation(self):
        """Test z rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(Rotz(0), np.identity(2), delta=self.tol)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4]))
        self.assertAllAlmostEqual(Rotz(np.pi / 2), expected, delta=self.tol)

        # test identity for theta=pi
        self.assertAllAlmostEqual(Rotz(np.pi), -1j * Z, delta=self.tol)

    def test_arbitrary_rotation(self):
        """Test arbitrary single qubit rotation is correct"""
        self.logTestName()

        # test identity for theta=0
        self.assertAllAlmostEqual(Rot3(0, 0, 0), np.identity(2), delta=self.tol)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        self.assertAllAlmostEqual(
            Rot3(a, b, c), arbitrary_rotation(a, b, c), delta=self.tol
        )


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

        self.assertEqual(set(qml.ops._qubit__ops__), set(self.dev._operation_map))

    def test_observable_map(self):
        """Test that default qubit device supports all PennyLane discrete observables."""
        self.logTestName()

        self.assertEqual(
            set(qml.ops._qubit__obs__) | {"Identity"}, set(self.dev._observable_map)
        )

    def test_get_operator_matrix(self):
        """Test the the correct matrix is returned given an operation name"""
        self.logTestName()

        for name, fn in {
            **self.dev._operation_map,
            **self.dev._observable_map,
        }.items():
            try:
                op = getattr(qml.ops, name)
            except AttributeError:
                op = getattr(qml.expval, name)

            p = [0.432423, -0.12312, 0.324][: op.num_params]

            if name == "QubitStateVector":
                p = [np.array([1, 0, 0, 0])]
            elif name == "QubitUnitary":
                p = [U]
            elif name == "Hermitian":
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
            op = getattr(qml.ops, gate_name)
            # the list of wires to apply the operation to
            w = list(range(op.num_wires))

            if op.par_domain == "A":
                # the parameter is an array
                if gate_name == "QubitStateVector":
                    p = [np.array([1, 0, 1, 1]) / np.sqrt(3)]
                    expected_out = p
                elif gate_name == "QubitUnitary":
                    p = [U]
                    w = [0]
                    expected_out = np.kron(U @ np.array([1, 0]), np.array([1, 0]))
                elif gate_name == "BasisState":
                    p = [np.array([1, 1])]
                    expected_out = np.array([0, 0, 0, 1])

            elif op.par_domain == "N":
                # the parameter is an integer
                p = [1, 3, 4][: op.num_params]
            else:
                # the parameter is a float
                p = [0.432423, -0.12312, 0.324][: op.num_params]

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

        with self.assertRaisesRegex(
            ValueError, r"State vector must be of length 2\*\*wires."
        ):
            p = [np.array([1, 0, 1, 1, 1]) / np.sqrt(3)]
            self.dev.apply("QubitStateVector", wires=[0, 1], par=[p])

        with self.assertRaisesRegex(
            ValueError,
            "BasisState parameter must be an array of 0 or 1 integers of length at most 2.",
        ):
            self.dev.apply("BasisState", wires=[0, 1], par=[np.array([-0.2, 4.2])])

        with self.assertRaisesRegex(
            ValueError,
            "The default.qubit plugin can apply BasisState only to all of the 2 wires.",
        ):
            self.dev.apply("BasisState", wires=[0, 1, 2], par=[np.array([0, 1])])

    def test_ev(self):
        """Test that expectation values are calculated correctly"""
        self.logTestName()
        self.dev.reset()

        # loop through all supported observables
        for name, fn in self.dev._observable_map.items():
            log.debug("\tTesting %s observable...", name)

            # start in the state |00>
            self.dev._state = np.array([1, 0, 1, 1]) / np.sqrt(3)

            # get the equivalent pennylane operation class
            op = getattr(qml.ops, name)

            if op.par_domain == "A":
                # the parameter is an array
                p = [H]
            else:
                # the parameter is a float
                p = [0.432423, -0.12312, 0.324][: op.num_params]

            if callable(fn):
                # if the default.qubit is an operation accepting parameters,
                # initialise it using the parameters generated above.
                O = fn(*p)
            else:
                # otherwise, the operation is simply an array.
                O = fn

            # calculate the expected output
            if op.num_wires == 1 or op.num_wires == 0:
                expected_out = self.dev._state.conj() @ np.kron(O, I) @ self.dev._state
            elif op.num_wires == 2:
                expected_out = self.dev._state.conj() @ O @ self.dev._state
            else:
                raise NotImplementedError(
                    "Test for operations with num_wires="
                    + op.num_wires
                    + " not implemented."
                )

            res = self.dev.ev(O, wires=[0])

            # verify the device is now in the expected state
            self.assertAllAlmostEqual(res, expected_out, delta=self.tol)

            # text warning raised if matrix is complex
            with pytest.warns(RuntimeWarning, match='Nonvanishing imaginary part'):
                self.dev.ev(H + 1j, [0])

    def test_var_pauliz(self):
        """Test that variance of PauliZ is the same as I-<Z>^2"""
        self.logTestName()
        self.dev.reset()

        phi = 0.543
        theta = 0.6543
        self.dev.apply('RX', wires=[0], par=[phi])
        self.dev.apply('RY', wires=[0], par=[theta])

        var = self.dev.var('PauliZ', [0], [])
        mean = self.dev.expval('PauliZ', [0], [])

        self.assertAlmostEqual(var, 1-mean**2, delta=self.tol)

    def test_var_pauliz_rotated_state(self):
        """test correct variance for <Z> of a rotated state"""
        self.logTestName()
        self.dev.reset()

        phi = 0.543
        theta = 0.6543
        self.dev.apply('RX', wires=[0], par=[phi])
        self.dev.apply('RY', wires=[0], par=[theta])
        var = self.dev.var('PauliZ', [0], [])
        expected = 0.25*(3-np.cos(2*theta)-2*np.cos(theta)**2*np.cos(2*phi))
        self.assertAlmostEqual(var, expected, delta=self.tol)

    def test_var_hermitian_rotated_state(self):
        """test correct variance for <H> of a rotated state"""
        self.logTestName()
        self.dev.reset()

        phi = 0.543
        theta = 0.6543

        H = np.array([[4, -1+6j], [-1-6j, 2]])
        self.dev.apply('RX', wires=[0], par=[phi])
        self.dev.apply('RY', wires=[0], par=[theta])
        var = self.dev.var('Hermitian', [0], [H])
        expected = 0.5*(2*np.sin(2*theta)*np.cos(phi)**2+24*np.sin(phi)\
                    *np.cos(phi)*(np.sin(theta)-np.cos(theta))+35*np.cos(2*phi)+39)
        self.assertAlmostEqual(var, expected, delta=self.tol)

class TestDefaultQubitIntegration(BaseTest):
    """Integration tests for default.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""
        self.logTestName()

        dev = qml.device("default.qubit", wires=2)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.shots, 0)
        self.assertEqual(dev.short_name, "default.qubit")

    def test_args(self):
        """Test that the plugin requires correct arguments"""
        self.logTestName()

        with self.assertRaisesRegex(
            TypeError, "missing 1 required positional argument: 'wires'"
        ):
            qml.device("default.qubit")

    def test_unsupported_gates(self):
        """Test error is raised with unsupported gates"""
        self.logTestName()
        dev = qml.device("default.qubit", wires=2)

        gates = set(dev._operation_map.keys())
        all_gates = set(qml.ops.__all_ops__)

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
                    return qml.expval(qml.X(0))

                return qml.expval(qml.PauliZ(0))

            with self.assertRaisesRegex(
                qml.DeviceError,
                "Gate {} not supported on device default.qubit".format(g),
            ):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_unsupported_observables(self):
        """Test error is raised with unsupported observables"""
        self.logTestName()
        dev = qml.device("default.qubit", wires=2)

        obs = set(dev._observable_map.keys())
        all_obs = set(qml.ops.__all_obs__)

        for g in all_obs - obs:
            op = getattr(qml.ops, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Test quantum function"""
                x = prep_par(x, op)
                return qml.expval(op(*x, wires=wires))

            with self.assertRaisesRegex(
                qml.DeviceError,
                "Observable {} not supported on device default.qubit".format(g),
            ):
                x = np.random.random([op.num_params])
                circuit(*x)

    def test_qubit_circuit(self):
        """Test that the default qubit plugin provides correct result for simple circuit"""
        self.logTestName()
        dev = qml.device("default.qubit", wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        # <0|RX(p)^\dagger.PauliY.RX(p)|0> = -sin(p)
        expected = -np.sin(p)
        self.assertAlmostEqual(circuit(p), expected, delta=self.tol)

    def test_qubit_identity(self):
        """Test that the default qubit plugin provides correct result for the Identity expectation"""
        self.logTestName()
        dev = qml.device("default.qubit", wires=1)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        self.assertAlmostEqual(circuit(p), 1, delta=self.tol)

    def test_nonzero_shots(self):
        """Test that the default qubit plugin provides correct result for high shot number"""
        self.logTestName()

        shots = 10 ** 4
        dev = qml.device("default.qubit", wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        self.assertAlmostEqual(np.mean(runs), -np.sin(p), delta=0.01)

    def test_supported_gates(self):
        """Test that all supported gates work correctly"""
        self.logTestName()
        a = 0.312
        b = 0.123

        dev = qml.device("default.qubit", wires=2)

        for g, qop in dev._operation_map.items():
            log.debug("\tTesting gate %s...", g)
            self.assertTrue(dev.supported(g))
            dev.reset()

            op = getattr(qml.ops, g)
            if op.num_wires == 0:
                if g == "BasisState":
                    wires = [0, 1]
                else:
                    wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*x):
                """Reference quantum function"""
                op(*x, wires=wires)
                return qml.expval(qml.PauliX(0))

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
                if op.num_wires == 1 or g == "QubitUnitary":
                    out_state = np.kron(O @ np.array([1, 0]), np.array([1, 0]))
                elif g == "QubitStateVector":
                    out_state = x[0]
                elif g == "BasisState":
                    out_state = np.array([0, 0, 0, 1])
                else:
                    out_state = O @ dev._state

                expectation = out_state.conj() @ np.kron(X, np.identity(2)) @ out_state
                return expectation

            if g == "QubitStateVector":
                p = np.array([1, 0, 0, 1]) / np.sqrt(2)
                self.assertAllEqual(circuit(p), reference(p))
            elif g == "BasisState":
                p = np.array([1, 1])
                self.assertAllEqual(circuit(p), reference(p))
            elif g == "QubitUnitary":
                self.assertAllEqual(circuit(U), reference(U))
            elif op.num_params == 1:
                self.assertAllEqual(circuit(a), reference(a))
            elif op.num_params == 2:
                self.assertAllEqual(circuit(a, b), reference(a, b))

    def test_supported_observables(self):
        """Test that all supported observables work correctly"""
        self.logTestName()
        a = 0.312

        dev = qml.device("default.qubit", wires=2)

        for g, qop in dev._observable_map.items():
            log.debug("\tTesting observable %s...", g)
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
                qml.RX(a, wires=0)
                return qml.expval(op(*x, wires=wires))

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
                out_state = np.kron(Rotx(a) @ np.array([1, 0]), np.array([1, 0]))
                expectation = out_state.conj() @ np.kron(O, np.identity(2)) @ out_state
                return expectation

            if op.num_params == 0:
                self.assertAllEqual(circuit(), reference())
            elif g == "Hermitian":
                self.assertAllEqual(circuit(H), reference(H))

    def test_two_qubit_observable(self):
        """Tests expval for two-qubit observables """
        self.logTestName()
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, target_observable=None):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.RZ(x[2], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(target_observable, wires=[0, 1]))

        target_state = 1 / np.sqrt(2) * np.array([1, 0, 0, 1])
        target_herm_op = np.outer(target_state.conj(), target_state)
        weights = np.array([0.5, 0.1, 0.2])
        expval = circuit(weights, target_observable=target_herm_op)
        self.assertAlmostEqual(expval, 0.590556, delta=self.tol)


if __name__ == "__main__":
    print("Testing PennyLane version " + qml.version() + ", default.qubit plugin.")
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (
        TestAuxillaryFunctions,
        TestStateFunctions,
        TestDefaultQubitDevice,
        TestDefaultQubitIntegration,
    ):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)
    unittest.TextTestRunner().run(suite)
