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
Unit tests for the :mod:`openqml` :class:`Device` class.
"""
import unittest
from unittest.mock import patch
import inspect
import logging as log
log.getLogger('defaults')

import autograd
from autograd import numpy as np

from defaults import openqml as qm, BaseTest


class DeviceTest(BaseTest):
    """Device tests."""
    def setUp(self):
        self.default_devices = ['default.qubit', 'default.gaussian']

        self.dev = {}

        for device_name in self.default_devices:
            self.dev[device_name] = qm.device(device_name, wires=2)

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
        """check that a nonempty set of operations/expectations are supported"""
        self.logTestName()

        for dev in self.dev.values():
            ops = dev.operations
            exps = dev.expectations
            self.assertTrue(len(ops) > 0)
            self.assertTrue(len(exps) > 0)

            for op in ops.union(exps):
                self.assertTrue(dev.supported(op))

    def test_capabilities(self):
        """check that device can give a dict of further capabilities"""
        self.logTestName()

        for dev in self.dev.values():
            caps = dev.capabilities()
            self.assertTrue(isinstance(caps, dict))

    def test_execute(self):
        """check that execution works on supported operations/expectations"""
        self.logTestName()

        for dev in self.dev.values():
            ops = dev.operations
            exps = dev.expectations

            queue = []
            for o in ops:
                log.debug('Queueing gate %s...', o)
                op = qm.ops.__getattribute__(o)

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

            temp = [isinstance(op, qm.operation.CV) for op in queue]
            if all(temp):
                expval = dev.execute(queue, [qm.expval.X(0, do_queue=False)])
            else:
                expval = dev.execute(queue, [qm.expval.PauliX(0, do_queue=False)])

            self.assertTrue(isinstance(expval, np.ndarray))

    def test_validity(self):
        """check that execution throws error on unsupported operations/expectations"""
        self.logTestName()

        for dev in self.dev.values():
            ops = dev.operations
            all_ops = {m[0] for m in inspect.getmembers(qm.ops, inspect.isclass)}

            for o in all_ops-ops:
                op = qm.ops.__getattribute__(o)

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

                temp = isinstance(queue[0], qm.operation.CV)

                with self.assertRaisesRegex(qm.DeviceError, 'not supported on device'):
                    if temp:
                        expval = dev.execute(queue, [qm.expval.X(0, do_queue=False)])
                    else:
                        expval = dev.execute(queue, [qm.expval.PauliX(0, do_queue=False)])

            exps = dev.expectations
            all_exps = {m[0] for m in inspect.getmembers(qm.expval, inspect.isclass)}

            for g in all_exps-exps:
                op = qm.expval.__getattribute__(g)

                if op.par_domain == 'A':
                    # skip expectations with array parameters, as there are too
                    # many constraints to consider. These should be tested
                    # directly within the plugin tests.
                    continue
                elif op.par_domain == 'N':
                    params = np.asarray(np.random.random([op.num_params]), dtype=np.int64)
                else:
                    params = np.random.random([op.num_params])

                queue = [op(*params, wires=list(range(op.num_wires)), do_queue=False)]

                temp = isinstance(queue[0], qm.operation.CV)

                with self.assertRaisesRegex(qm.DeviceError, 'not supported on device'):
                    if temp:
                        expval = dev.execute([qm.Rotation(0.5, wires=0, do_queue=False)], queue)
                    else:
                        expval = dev.execute([qm.RX(0.5, wires=0, do_queue=False)], queue)


class InitDeviceTests(BaseTest):
    """Tests for device loader in __init__.py"""

    def test_no_device(self):
        """Test exception raised for a device that doesn't exist"""
        self.logTestName()

        with self.assertRaisesRegex(qm.DeviceError, 'Device does not exist'):
            qm.device('None', wires=0)

    @patch.object(qm, '__version__', return_value='0')
    def test_outdated_API(self, n):
        """Test exception raised if plugin that targets an old API is loaded"""
        self.logTestName()

        with self.assertLogs(level='WARNING') as l:
            qm.device('default.qubit', wires=0)
            self.assertEqual(len(l.output), 1)
            self.assertEqual(len(l.records), 1)
            self.assertIn('Plugin API version', l.output[0])


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', Device class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (DeviceTest, InitDeviceTests):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
