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
Unit tests for the :mod:`pennylane` :class:`Device` class.
"""
import unittest
from unittest.mock import patch
import inspect
import logging as log
log.getLogger('defaults')

import autograd
from autograd import numpy as np

from defaults import pennylane as qml, BaseTest
from pennylane.plugins import DefaultQubit


class DeviceTest(BaseTest):
    """Device tests."""
    def setUp(self):
        self.default_devices = ['default.qubit', 'default.gaussian']

        self.dev = {}

        for device_name in self.default_devices:
            self.dev[device_name] = qml.device(device_name, wires=2)

    def test_reset(self):
        """Test reset works (no error is raised). Does not verify
        that the circuit is actually reset."""
        self.logTestName()

        for dev in self.dev.values():
            dev.reset()

    def test_short_name(self):
        """test correct short name"""
        self.logTestName()

        for name, dev in self.dev.items():
            self.assertEqual(dev.short_name, name)

    def test_supported(self):
        """check that a nonempty set of operations/observables are supported"""
        self.logTestName()

        for dev in self.dev.values():
            ops = dev.operations
            exps = dev.observables
            self.assertTrue(len(ops) > 0)
            self.assertTrue(len(exps) > 0)

            for op in ops.union(exps):
                self.assertTrue(dev.supported(op))

    def test_check_validity(self):
        """test that the check_validity method correctly
        determines what operations/observables are supported."""
        self.logTestName()

        dev = qml.device('default.qubit', wires=2)
        # overwrite the device supported operations and observables
        dev._operation_map = {'RX':0, 'PauliX':0, 'PauliY':0, 'PauliZ':0, 'Hadamard':0}
        dev._observable_map = {'PauliZ':0, 'Identity':0}

        # test a valid queue
        queue = [
            qml.RX(1., wires=0, do_queue=False),
            qml.PauliY(wires=1, do_queue=False),
            qml.PauliZ(wires=2, do_queue=False),
        ]

        observables = [qml.expval(qml.PauliZ(0, do_queue=False))]

        dev.check_validity(queue, observables)

        # test an invalid operation
        queue = [qml.RY(1., wires=0, do_queue=False)]
        with self.assertRaisesRegex(qml.DeviceError, "Gate RY not supported"):
            dev.check_validity(queue, observables)

        # test an invalid observable with the same name
        # as a valid operation
        queue = [qml.PauliY(wires=0, do_queue=False)]
        observables = [qml.expval(qml.PauliY(0, do_queue=False))]
        with self.assertRaisesRegex(qml.DeviceError, "Observable PauliY not supported"):
            dev.check_validity(queue, observables)

    def test_capabilities(self):
        """check that device can give a dict of further capabilities"""
        self.logTestName()

        for dev in self.dev.values():
            caps = dev.capabilities()
            self.assertTrue(isinstance(caps, dict))

    @patch.object(DefaultQubit, 'pre_measure', lambda self: log.info(self.op_queue))
    def test_op_queue(self):
        """Check that peaking at the operation queue works correctly"""
        self.logTestName()

        # queue some gates
        queue = []
        queue.append(qml.RX(0.543, wires=[0], do_queue=False))
        queue.append(qml.CNOT(wires=[0, 1], do_queue=False))

        dev = qml.device('default.qubit', wires=2)

        # outside of an execution context, error will be raised
        with self.assertRaisesRegex(ValueError, "Cannot access the operation queue outside of the execution context!"):
            dev.op_queue

        # inside of the execute method, it works
        with self.assertLogs(level='INFO') as l:
            dev.execute(queue, [qml.expval(qml.PauliX(0, do_queue=False))])
            self.assertEqual(len(l.output), 1)
            self.assertEqual(len(l.records), 1)
            self.assertIn('INFO:root:[<pennylane.ops.qubit.RX object', l.output[0])

    @patch.object(DefaultQubit, 'pre_measure', lambda self: log.info(self.obs_queue))
    def test_obs_queue(self):
        """Check that peaking at the obs queue works correctly"""
        self.logTestName()

        # queue some gates
        queue = []
        queue.append(qml.RX(0.543, wires=[0], do_queue=False))
        queue.append(qml.CNOT(wires=[0, 1], do_queue=False))

        dev = qml.device('default.qubit', wires=2)

        # outside of an execution context, error will be raised
        with self.assertRaisesRegex(ValueError, "Cannot access the observable value queue outside of the execution context!"):
            dev.obs_queue

        # inside of the execute method, it works
        with self.assertLogs(level='INFO') as l:
            dev.execute(queue, [qml.expval(qml.PauliX(0, do_queue=False))])
            self.assertEqual(len(l.output), 1)
            self.assertEqual(len(l.records), 1)
            self.assertIn('INFO:root:[<pennylane.ops.qubit.PauliX object', l.output[0])

    def test_execute(self):
        """check that execution works on supported operations/observables"""
        self.logTestName()

        for dev in self.dev.values():
            ops = dev.operations
            exps = dev.observables

            queue = []
            for o in ops:
                log.debug('Queueing gate %s...', o)
                op = qml.ops.__getattribute__(o)

                if op.par_domain == 'A':
                    # skip operations with array parameters, as there are too
                    # many constraints to consider. These should be tested
                    # directly within the plugin tests.
                    continue
                elif op.par_domain == 'N':
                    params = np.asarray(np.random.random([op.num_params]), dtype=np.int64)
                else:
                    params = np.random.random([op.num_params])

                queue.append(op(*params, wires=list(range(op.num_wires)), do_queue=False))

            temp = [isinstance(op, qml.operation.CV) for op in queue]
            if all(temp):
                expval = dev.execute(queue, [qml.expval(qml.X(0, do_queue=False))])
            else:
                expval = dev.execute(queue, [qml.expval(qml.PauliX(0, do_queue=False))])

            self.assertTrue(isinstance(expval, np.ndarray))
            
    def test_sample_attribute_error(self):
        """Check that an error is raised if a required attribute
           is not present in a sampled observable"""
        self.logTestName()

        dev = qml.device('default.qubit', wires=2)

        queue = [qml.RX(0.543, wires=[0], do_queue=False)]

        # Make a sampling observable but delete its num_samples attribute
        obs = qml.sample(qml.PauliZ(0, do_queue=False), n = 10)
        del obs.num_samples
        obs = [obs]

        with self.assertRaisesRegex(qml.DeviceError, "Number of samples not specified for observable"):
            dev.execute(queue, obs)

    def test_validity(self):
        """check that execution throws error on unsupported operations/observables"""
        self.logTestName()

        for dev in self.dev.values():
            ops = dev.operations
            all_ops = set(qml.ops.__all_ops__)

            for o in all_ops-ops:
                op = getattr(qml.ops, o)

                if op.par_domain == 'A':
                    # skip operations with array parameters, as there are too
                    # many constraints to consider. These should be tested
                    # directly within the plugin tests.
                    continue
                elif op.par_domain == 'N':
                    params = np.asarray(np.random.random([op.num_params]), dtype=np.int64)
                else:
                    params = np.random.random([op.num_params])

                queue = [op(*params, wires=list(range(op.num_wires)), do_queue=False)]

                temp = isinstance(queue[0], qml.operation.CV)

                with self.assertRaisesRegex(qml.DeviceError, 'not supported on device'):
                    if temp:
                        expval = dev.execute(queue, [qml.expval(qml.X(0, do_queue=False))])
                    else:
                        expval = dev.execute(queue, [qml.expval(qml.PauliX(0, do_queue=False))])

            exps = dev.observables
            all_exps = set(qml.ops.__all_obs__)

            for g in all_exps-exps:
                op = getattr(qml.ops, g)

                if op.par_domain == 'A':
                    # skip observables with array parameters, as there are too
                    # many constraints to consider. These should be tested
                    # directly within the plugin tests.
                    continue
                elif op.par_domain == 'N':
                    params = np.asarray(np.random.random([op.num_params]), dtype=np.int64)
                else:
                    params = np.random.random([op.num_params])

                queue = [op(*params, wires=list(range(op.num_wires)), do_queue=False)]

                temp = isinstance(queue[0], qml.operation.CV)

                with self.assertRaisesRegex(qml.DeviceError, 'not supported on device'):
                    if temp:
                        expval = dev.execute([qml.Rotation(0.5, wires=0, do_queue=False)], queue)
                    else:
                        expval = dev.execute([qml.RX(0.5, wires=0, do_queue=False)], queue)


class InitDeviceTests(BaseTest):
    """Tests for device loader in __init__.py"""

    def test_no_device(self):
        """Test exception raised for a device that doesn't exist"""
        self.logTestName()

        with self.assertRaisesRegex(qml.DeviceError, 'Device does not exist'):
            qml.device('None', wires=0)

    @patch.object(qml, 'version', return_value='0.0.1')
    def test_outdated_API(self, n):
        """Test exception raised if plugin that targets an old API is loaded"""
        self.logTestName()

        with self.assertRaisesRegex(qml.DeviceError, 'plugin requires PennyLane versions'):
            qml.device('default.qubit', wires=0)



if __name__ == '__main__':
    print('Testing PennyLane version ' + qml.version() + ', Device class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (DeviceTest, InitDeviceTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
