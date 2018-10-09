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
Unit tests for the Gaussian plugin.
"""
import inspect
import unittest
import logging as log
log.getLogger()

import strawberryfields as sf

import openqml as qm
from openqml import numpy as np

from defaults import openqml_sf as qmsf, BaseTest


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == 'A':
        return [np.diag([x, 1]) for x in par]
    return par


class GaussianTests(BaseTest):
    """Test the Gaussian simulator."""

    def test_load_gaussian_device(self):
        """Test that the gaussian plugin loads correctly"""
        log.info('test_load_gaussian_device')

        dev = qm.device('strawberryfields.gaussian', wires=2)
        self.assertEqual(dev.wires, 2)
        self.assertEqual(dev.hbar, 2)
        self.assertEqual(dev.shots, 0)
        self.assertEqual(dev.short_name, 'strawberryfields.gaussian')

    def test_gaussian_args(self):
        """Test that the gaussian plugin requires correct arguments"""
        log.info('test_gaussian_args')

        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'wires'"):
            dev = qm.device('strawberryfields.gaussian')

    def test_unsupported_gates(self):
        """Test error is raised with unsupported gates"""
        log.info('test_unsupported_ops')

        dev = qm.device('strawberryfields.gaussian', wires=2)
        gates = set(dev._operator_map.keys())
        all_gates = {m[0] for m in inspect.getmembers(qm.ops, inspect.isclass)}

        for g in all_gates - gates:
            op = getattr(qm.ops, g)

            if op.n_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.n_wires))

            @qm.qfunc(dev)
            def circuit(*x):
                x = prep_par(x, op)
                op(*x, wires=wires)

                if issubclass(op, qm.operation.CV):
                    return qm.expectation.X(0)
                else:
                    return qm.expectation.PauliZ(0)

            with self.assertRaisesRegex(qm.DeviceError,
                "Gate {} not supported on device strawberryfields.gaussian".format(g)):
                x = np.random.random([op.n_params])
                circuit(*x)

    def test_unsupported_observables(self):
        """Test error is raised with unsupported observables"""
        log.info('test_unsupported_observables')

        dev = qm.device('strawberryfields.gaussian', wires=2)
        obs = set(dev._observable_map.keys())
        all_obs = {m[0] for m in inspect.getmembers(qm.expectation, inspect.isclass)}

        for g in all_obs - obs:
            op = getattr(qm.expectation, g)

            if op.n_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.n_wires))

            @qm.qfunc(dev)
            def circuit(*x):
                x = prep_par(x, op)
                return op(*x, wires=wires)

            with self.assertRaisesRegex(qm.DeviceError,
                "Observable {} not supported on device strawberryfields.gaussian".format(g)):
                x = np.random.random([op.n_params])
                circuit(*x)

    def test_gaussian_circuit(self):
        """Test that the gaussian plugin provides correct result for simple circuit"""
        log.info('test_gaussian_circuit')

        dev = qm.device('strawberryfields.gaussian', wires=1)

        @qm.qfunc(dev)
        def circuit(x):
            qm.Displacement(x, 0, wires=0)
            return qm.expectation.PhotonNumber(0)

        self.assertAlmostEqual(circuit(1), 1, delta=self.tol)

    def test_nonzero_shots(self):
        """Test that the gaussian plugin provides correct result for high shot number"""
        log.info('test_gaussian_circuit')

        shots = 10**2
        dev = qm.device('strawberryfields.gaussian', wires=1, shots=shots)

        @qm.qfunc(dev)
        def circuit(x):
            qm.Displacement(x, 0, wires=0)
            return qm.expectation.PhotonNumber(0)

        expected_var = np.sqrt(1/shots)
        self.assertAlmostEqual(circuit(1), 1, delta=expected_var)

    def test_supported_gaussian_gates(self):
        """Test that all supported gates work correctly"""
        log.info('test_supported_gaussian_gates')
        a = 0.312
        b = 0.123

        dev = qm.device('strawberryfields.gaussian', wires=2)

        gates = list(dev._operator_map.items())
        for g, sfop in gates:
            log.info('\tTesting gate {}...'.format(g))
            self.assertTrue(dev.supported(g))

            op = getattr(qm.ops, g)
            if op.n_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.n_wires))

            @qm.qfunc(dev)
            def circuit(*x):
                qm.TwoModeSqueezing(0.1, 0, wires=[0, 1])
                op(*x, wires=wires)
                return qm.expectation.PhotonNumber(0), qm.expectation.PhotonNumber(1)

            # compare to reference SF engine
            def SF_reference(*x):
                """SF reference circuit"""
                eng, q = sf.Engine(2)
                with eng:
                    sf.ops.S2gate(0.1) | q
                    sfop(*x) | [q[i] for i in wires]

                state = eng.run('gaussian')
                return state.mean_photon(0)[0], state.mean_photon(1)[0]

            if g == 'GaussianState':
                r = np.array([0, 0])
                V = np.array([[0.5, 0], [0, 2]])
                self.assertAllEqual(circuit(V, r), SF_reference(V, r))
            elif op.n_params == 1:
                self.assertAllEqual(circuit(a), SF_reference(a))
            elif op.n_params == 2:
                self.assertAllEqual(circuit(a, b), SF_reference(a, b))

    def test_supported_gaussian_observables(self):
        """Test that all supported observables work correctly"""
        log.info('test_supported_gaussian_observables')
        a = 0.312
        a_array = np.eye(3)

        dev = qm.device('strawberryfields.gaussian', wires=2)

        observables = list(dev._observable_map.items())
        for g, sfop in observables:
            log.info('\tTesting observable {}...'.format(g))
            self.assertTrue(dev.supported(g))

            op = getattr(qm.expectation, g)
            if op.n_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.n_wires))

            @qm.qfunc(dev)
            def circuit(*x):
                qm.Displacement(0.1, 0, wires=0)
                qm.TwoModeSqueezing(0.1, 0, wires=[0, 1])
                return op(*x, wires=wires)

            # compare to reference SF engine
            def SF_reference(*x):
                """SF reference circuit"""
                eng, q = sf.Engine(2)
                with eng:
                    sf.ops.Xgate(0.2) | q[0]
                    sf.ops.S2gate(0.1) | q

                state = eng.run('gaussian')
                return sfop(state, wires, x)[0]

            if op.n_params == 0:
                self.assertAllEqual(circuit(), SF_reference())
            elif op.n_params == 1:
                p = a_array if op.par_domain == 'A' else a
                self.assertAllEqual(circuit(p), SF_reference(p))


if __name__ == '__main__':
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (GaussianTests,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
