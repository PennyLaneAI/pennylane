# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for the conversion of plxpr to a tape.
"""

import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

jax = pytest.importorskip("jax")

from pennylane.capture.to_tape import CollectOpsandMeas  # pylint: disable=wrong-import-position


class TestCollectOpsandMeas:
    """Tests for the CollectOpsandMeas class."""

    def test_flat_func(self):
        """Test a function without classical structure."""

        def f(x):
            qml.RX(x, 0)
            qml.CNOT((0, 1))
            qml.QFT(wires=(0, 1, 2))
            return qml.expval(qml.Z(0))

        obj = CollectOpsandMeas()
        obj(f)(1.2)
        qml.assert_equal(obj.state["ops"][0], qml.RX(1.2, 0))
        qml.assert_equal(obj.state["ops"][1], qml.CNOT((0, 1)))
        qml.assert_equal(obj.state["ops"][2], qml.QFT((0, 1, 2)))
        assert len(obj.state["ops"]) == 3

        qml.assert_equal(obj.state["measurements"][0], qml.expval(qml.Z(0)))

    def test_for_loop(self):

        def f(n):
            @qml.for_loop(n)
            def g(i):
                qml.X(i)

            g()

        obj = CollectOpsandMeas()
        obj(f)(3)
        assert len(obj.state["ops"]) == 3
        qml.assert_equal(obj.state["ops"][0], qml.X(0))
        qml.assert_equal(obj.state["ops"][1], qml.X(1))
        qml.assert_equal(obj.state["ops"][2], qml.X(2))

        assert len(obj.state["measurements"]) == 0

    def test_while_loop(self):

        def g(x):
            @qml.while_loop(lambda x, i: i < 3)
            def loop(x, i):
                qml.RX(x, i)
                return 2 * x, i + 1

            loop(x, 0)

        obj = CollectOpsandMeas()
        x = jax.numpy.array(1.2)
        obj(g)(x)

        assert len(obj.state["ops"]) == 3
        assert len(obj.state["measurements"]) == 0

        qml.assert_equal(obj.state["ops"][0], qml.RX(x, 0))
        qml.assert_equal(obj.state["ops"][1], qml.RX(2 * x, 1))
        qml.assert_equal(obj.state["ops"][2], qml.RX(4 * x, 2))

    def test_cond_bool(self):

        def f(x, value):
            qml.cond(value, qml.RX, false_fn=qml.RY)(x, 0)

        obj1 = CollectOpsandMeas()
        x = jax.numpy.array(-0.5)
        obj1(f)(x, True)
        assert len(obj1.state["ops"]) == 1
        qml.assert_equal(obj1.state["ops"][0], qml.RX(x, 0))

        obj2 = CollectOpsandMeas()
        obj2(f)(x, False)
        assert len(obj2.state["ops"]) == 1
        qml.assert_equal(obj2.state["ops"][0], qml.RY(x, 0))

    def test_measure(self):
        """Test capturing measurements."""

        def f():
            m0 = qml.measure(0)
            return qml.sample(op=m0)

        obj = CollectOpsandMeas()
        obj(f)()

        assert len(obj.state["ops"]) == 1
        assert isinstance(obj.state["ops"][0], qml.measurements.MidMeasureMP)
        assert obj.state["ops"][0].wires == qml.wires.Wires(0)

        assert isinstance(obj.state["measurements"][0], qml.measurements.SampleMP)
        assert obj.state["measurements"][0].mv is not None

    def test_cond_mcm(self):
        """Test capturing a conditional of a mid circuit measurement."""

        def rx(x, w):
            qml.RX(x, w)

        def f(x):
            m0 = qml.measure(0)
            qml.cond(m0, rx)(x, 2)
            return m0

        x = jax.numpy.array(0.987)

        obj = CollectOpsandMeas()
        mv = obj(f)(x)

        assert len(obj.state["ops"]) == 2
        assert isinstance(obj.state["ops"][0], qml.measurements.MidMeasureMP)
        assert mv.measurements[0] is obj.state["ops"][0]

        qml.assert_equal(obj.state["ops"][1], qml.ops.Conditional(mv, qml.RX(x, 2)))

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):

        def qfunc(x):
            qml.RX(x, 0)
            qml.RY(2 * x, 0)
            qml.X(2)

        def f(x):
            qml.adjoint(qfunc, lazy=lazy)(x)

        obj = CollectOpsandMeas()
        x = jax.numpy.array(2.1)
        obj(f)(x)

        assert len(obj.state["ops"]) == 3
        qml.assert_equal(obj.state["ops"][0], qml.adjoint(qml.X(2), lazy=lazy))
        qml.assert_equal(obj.state["ops"][1], qml.adjoint(qml.RY(2 * x, 0), lazy=lazy))
        qml.assert_equal(obj.state["ops"][2], qml.adjoint(qml.RX(x, 0), lazy=lazy))

    def test_control_transform(self):

        def qfunc(x, wire):
            qml.RX(x, wire)
            qml.X(wire)

        def f(x):
            qml.ctrl(qfunc, control=[1, 2], control_values=[False, False])(x, 0)

        obj = CollectOpsandMeas()
        x = jax.numpy.array(-0.98)
        obj(f)(x)

        assert len(obj.state["ops"]) == 2
        expected0 = qml.ctrl(qml.RX(x, 0), [1, 2], control_values=[False, False])
        qml.assert_equal(obj.state["ops"][0], expected0)
        expected1 = qml.ctrl(qml.X(0), [1, 2], control_values=[False, False])
        qml.assert_equal(obj.state["ops"][1], expected1)
