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
Tests for capturing measurements.
"""
import numpy as np

# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane.capture.measure import _get_measure_primitive
from pennylane.capture.primitives import _get_abstract_measurement
from pennylane.measurements import (
    ClassicalShadowMP,
    DensityMatrixMP,
    ExpectationMP,
    MidMeasureMP,
    MutualInfoMP,
    ProbabilityMP,
    PurityMP,
    SampleMP,
    ShadowExpvalMP,
    StateMP,
    VarianceMP,
    VnEntropyMP,
)

jax = pytest.importorskip("jax")

pytestmark = pytest.mark.jax

measure_prim = _get_measure_primitive()
AbstractMeasurement = _get_abstract_measurement()


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


# pylint: disable=unnecessary-lambda
creation_funcs = [
    lambda: qml.state(),
    lambda: qml.density_matrix(wires=(0, 1)),
    lambda: qml.expval(qml.X(0)),
    lambda: ExpectationMP(wires=qml.wires.Wires((0, 1)), eigvals=np.array([-1.0, -0.5, 0.5, 1.0])),
    # lambda : qml.expval(qml.measure(0)+qml.measure(1)),
    lambda: qml.var(qml.X(0)),
    lambda: VarianceMP(wires=qml.wires.Wires((0, 1)), eigvals=np.array([-1.0, -0.5, 0.5, 1.0])),
    # lambda : qml.var(qml.measure(0)+qml.measure(1)),
    lambda: qml.probs(wires=(0, 1)),
    lambda: qml.probs(op=qml.X(0)),
    # lambda : qml.probs(op=[qml.measure(0), qml.measure(1)]),
    lambda: ProbabilityMP(wires=qml.wires.Wires((0, 1)), eigvals=np.array([-1.0, -0.5, 0.5, 1.0])),
    lambda: qml.sample(wires=(3, 4)),
    lambda: qml.shadow_expval(np.array(2) * qml.X(0)),
    lambda: qml.vn_entropy(wires=(1, 2)),
    lambda: qml.purity(wires=(0, 1)),
    lambda: qml.mutual_info(wires0=(1, 3), wires1=(2, 4), log_base=2),
    lambda: qml.classical_shadow(wires=(0, 1), seed=84),
]


@pytest.mark.parametrize("func", creation_funcs)
def test_capture_and_eval(func):
    """Test that captured jaxpr can be evaluated to restore the initial measurement."""

    mp = func()

    jaxpr = jax.make_jaxpr(func)()
    out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0]

    assert qml.equal(mp, out)


def test_state():
    """Test the capture of a state measurement."""

    def f():
        return qml.capture.measure(qml.state(), num_device_wires=4)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive == StateMP._wires_primitive
    assert len(jaxpr.eqns[0].invars) == 0
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 0
    assert mp._abstract_eval == StateMP._abstract_eval

    assert jaxpr.eqns[1].primitive == measure_prim
    assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(None)}
    assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((16,), jax.numpy.complex64)


@pytest.mark.parametrize("wires, shape", [([0, 1], (4, 4)), ([], (16, 16))])
def test_density_matrix(wires, shape):
    """Test the capture of a density matrix."""

    def f():
        mp = qml.density_matrix(wires=wires)
        return qml.capture.measure(mp, num_device_wires=4)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive == DensityMatrixMP._wires_primitive
    assert len(jaxpr.eqns[0].invars) == len(wires)
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == len(wires)
    assert mp._abstract_eval == DensityMatrixMP._abstract_eval

    assert jaxpr.eqns[1].primitive == measure_prim
    assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(None)}
    assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray(shape, jax.numpy.complex64)


@pytest.mark.parametrize("m_type", (ExpectationMP, VarianceMP))
class TestExpvalVar:
    """Tests for capturing an expectation value."""

    def test_capture_obs(self, m_type):
        """Test that the expectation value of an observable can be captured."""

        def f():
            mp = m_type(obs=qml.X(0))
            return qml.capture.measure(mp, shots=50)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 3
        assert jaxpr.eqns[0].primitive == qml.X._primitive

        assert jaxpr.eqns[1].primitive == m_type._obs_primitive
        assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars

        am = jaxpr.eqns[1].outvars[0].aval
        assert isinstance(am, AbstractMeasurement)
        assert am.n_wires is None
        assert am._abstract_eval == m_type._abstract_eval

        assert jaxpr.eqns[2].primitive == measure_prim
        assert jaxpr.eqns[2].params == {"num_device_wires": 0, "shots": qml.measurements.Shots(50)}
        assert jaxpr.eqns[2].invars == jaxpr.eqns[1].outvars
        out_aval = jaxpr.eqns[2].outvars[0].aval
        assert out_aval == jax.core.ShapedArray((), jax.numpy.float32)

    def test_capture_eigvals_wires(self, m_type):
        """Test that we can capture an expectation value of eigvals+wires."""

        def f(eigs):
            mp = m_type(eigvals=eigs, wires=qml.wires.Wires((0, 1)))
            return qml.capture.measure(mp, shots=50)

        eigs = np.array([1.0, 0.5, -0.5, -1.0])
        jaxpr = jax.make_jaxpr(f)(eigs)

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == m_type._wires_primitive
        assert jaxpr.eqns[0].params == {"has_eigvals": True}
        assert [x.val for x in jaxpr.eqns[0].invars[:-1]] == [0, 1]  # the wires
        assert jaxpr.eqns[0].invars[-1] == jaxpr.jaxpr.invars[0]  # the eigvals

        am = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(am, AbstractMeasurement)
        assert am.n_wires == 2
        assert am._abstract_eval == m_type._abstract_eval

        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[1].invars == jaxpr.eqns[0].outvars
        out_aval = jaxpr.eqns[1].outvars[0].aval
        assert out_aval == jax.core.ShapedArray((), jax.numpy.float32)

    def test_simple_single_mcm(self, m_type):
        """Test that we can take the expectation value of a mid circuit measurement."""

        def f():
            m0 = qml.measure(0)
            mp = m_type(obs=m0)
            return qml.capture.measure(mp)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == MidMeasureMP._wires_primitive
        assert jaxpr.eqns[0].params == {"postselect": None, "reset": False}
        assert jaxpr.eqns[0].invars[0].val == 0  # the wire
        am0 = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(am0, AbstractMeasurement)
        assert am0.n_wires == 1
        assert am0._abstract_eval == MidMeasureMP._abstract_eval

        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[1].invars == jaxpr.eqns[0].outvars
        aval0 = jaxpr.eqns[1].outvars[0].aval
        assert aval0 == jax.core.ShapedArray((), jax.numpy.int32)

        assert jaxpr.eqns[2].primitive == m_type._mcm_primitive
        assert jaxpr.eqns[2].invars == jaxpr.eqns[1].outvars
        aval1 = jaxpr.eqns[2].outvars[0].aval
        assert isinstance(aval1, AbstractMeasurement)
        assert aval1.n_wires == 1
        assert aval1._abstract_eval == m_type._abstract_eval

        assert jaxpr.eqns[3].primitive == measure_prim
        assert jaxpr.eqns[3].params == {
            "num_device_wires": 0,
            "shots": qml.measurements.Shots(None),
        }
        assert jaxpr.eqns[3].invars == jaxpr.eqns[2].outvars
        aval2 = jaxpr.eqns[3].outvars[0].aval
        assert aval2 == jax.core.ShapedArray((), jax.numpy.float32)


class TestProbs:

    @pytest.mark.parametrize("wires, shape", [([0, 1, 2], 8), ([], 16)])
    def test_wires(self, wires, shape):
        """Tests capturing probabilities on wires."""

        def f():
            mp = qml.probs(wires=wires)
            return qml.capture.measure(mp, num_device_wires=4, shots=50)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 2

        assert jaxpr.eqns[0].primitive == ProbabilityMP._wires_primitive
        assert [x.val for x in jaxpr.eqns[0].invars] == wires
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == len(wires)
        assert mp._abstract_eval == ProbabilityMP._abstract_eval

        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(50)}
        assert jaxpr.eqns[1].invars == jaxpr.eqns[0].outvars
        assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((shape,), jax.numpy.float32)

    def test_eigvals(self):
        """Test capturing probabilities eith eigenvalues."""

        def f(eigs):
            mp = ProbabilityMP(eigvals=eigs, wires=qml.wires.Wires((0, 1)))
            return qml.capture.measure(mp, num_device_wires=4, shots=qml.measurements.Shots(50))

        eigvals = np.array([-1.0, -0.5, 0.5, 1.0])
        jaxpr = jax.make_jaxpr(f)(eigvals)

        assert len(jaxpr.eqns) == 2

        assert jaxpr.eqns[0].primitive == ProbabilityMP._wires_primitive
        assert jaxpr.eqns[0].params == {"has_eigvals": True}
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == 2
        assert mp._abstract_eval == ProbabilityMP._abstract_eval

        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[1].invars == jaxpr.eqns[0].outvars
        assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((4,), jax.numpy.float32)

    def test_multiple_mcms(self):
        """Test measuring multiple mcms."""

        def f():
            m0 = qml.measure(0)
            m1 = qml.measure(0)
            mp = qml.probs(op=[m0, m1])
            return qml.capture.measure(mp, num_device_wires=4, shots=50)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 6

        for i in [0, 2]:
            assert jaxpr.eqns[i].primitive == MidMeasureMP._wires_primitive
            assert jaxpr.eqns[i].params == {"postselect": None, "reset": False}
            assert jaxpr.eqns[i].invars[0].val == 0
            mp = jaxpr.eqns[i].outvars[0].aval
            assert isinstance(mp, AbstractMeasurement)
            assert mp.n_wires == 1
            assert mp._abstract_eval == MidMeasureMP._abstract_eval

        for i in [1, 3]:
            assert jaxpr.eqns[i].primitive == measure_prim
            assert jaxpr.eqns[i].invars == jaxpr.eqns[i - 1].outvars
            out = jaxpr.eqns[i].outvars[0].aval
            assert out == jax.core.ShapedArray((), jax.numpy.int32)

        assert jaxpr.eqns[4].primitive == ProbabilityMP._mcm_primitive
        assert jaxpr.eqns[4].invars[0] == jaxpr.eqns[1].outvars[0]
        assert jaxpr.eqns[4].invars[1] == jaxpr.eqns[3].outvars[0]
        out = jaxpr.eqns[4].outvars[0].aval
        assert isinstance(out, AbstractMeasurement)
        assert out.n_wires == 2
        assert out._abstract_eval == ProbabilityMP._abstract_eval

        assert jaxpr.eqns[5].primitive == measure_prim
        assert jaxpr.eqns[5].invars == jaxpr.eqns[4].outvars
        out = jaxpr.eqns[5].outvars[0].aval
        assert out == jax.core.ShapedArray((4,), jax.numpy.float32)


class TestSample:

    @pytest.mark.parametrize("wires, dim1_len", [([0, 1, 2], 3), ([], 4)])
    def test_wires(self, wires, dim1_len):
        """Tests capturing probabilities on wires."""

        def f():
            mp = qml.sample(wires=wires)
            return qml.capture.measure(mp, num_device_wires=4, shots=50)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 2

        assert jaxpr.eqns[0].primitive == SampleMP._wires_primitive
        assert [x.val for x in jaxpr.eqns[0].invars] == wires
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == len(wires)
        assert mp._abstract_eval == SampleMP._abstract_eval

        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(50)}
        assert jaxpr.eqns[1].invars == jaxpr.eqns[0].outvars
        assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray(
            (50, dim1_len), jax.numpy.int32
        )

    def test_eigvals(self):
        """Test capturing probabilities eith eigenvalues."""

        def f(eigs):
            mp = SampleMP(eigvals=eigs, wires=qml.wires.Wires((0, 1)))
            return qml.capture.measure(mp, num_device_wires=4, shots=qml.measurements.Shots(50))

        eigvals = np.array([-1.0, -0.5, 0.5, 1.0])
        jaxpr = jax.make_jaxpr(f)(eigvals)

        assert len(jaxpr.eqns) == 2

        assert jaxpr.eqns[0].primitive == SampleMP._wires_primitive
        assert jaxpr.eqns[0].params == {"has_eigvals": True}
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == 2
        assert mp._abstract_eval == SampleMP._abstract_eval

        assert jaxpr.eqns[1].primitive == measure_prim
        assert jaxpr.eqns[1].invars == jaxpr.eqns[0].outvars
        assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((50,), jax.numpy.float32)

    def test_multiple_mcms(self):
        """Test measuring multiple mcms."""

        def f():
            m0 = qml.measure(0)
            m1 = qml.measure(0)
            mp = qml.sample(op=[m0, m1])
            return qml.capture.measure(mp, num_device_wires=4, shots=50)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 6

        for i in [0, 2]:
            assert jaxpr.eqns[i].primitive == MidMeasureMP._wires_primitive
            assert jaxpr.eqns[i].params == {"postselect": None, "reset": False}
            assert jaxpr.eqns[i].invars[0].val == 0
            mp = jaxpr.eqns[i].outvars[0].aval
            assert isinstance(mp, AbstractMeasurement)
            assert mp.n_wires == 1
            assert mp._abstract_eval == MidMeasureMP._abstract_eval

        for i in [1, 3]:
            assert jaxpr.eqns[i].primitive == measure_prim
            assert jaxpr.eqns[i].invars == jaxpr.eqns[i - 1].outvars
            out = jaxpr.eqns[i].outvars[0].aval
            assert out == jax.core.ShapedArray((), jax.numpy.int32)

        assert jaxpr.eqns[4].primitive == SampleMP._mcm_primitive
        assert jaxpr.eqns[4].invars[0] == jaxpr.eqns[1].outvars[0]
        assert jaxpr.eqns[4].invars[1] == jaxpr.eqns[3].outvars[0]
        out = jaxpr.eqns[4].outvars[0].aval
        assert isinstance(out, AbstractMeasurement)
        assert out.n_wires == 2
        assert out._abstract_eval == SampleMP._abstract_eval

        assert jaxpr.eqns[5].primitive == measure_prim
        assert jaxpr.eqns[5].invars == jaxpr.eqns[4].outvars
        out = jaxpr.eqns[5].outvars[0].aval
        assert out == jax.core.ShapedArray((50, 2), jax.numpy.int32)


def test_shadow_expval():
    """Test that the shadow expval of an observable can be captured."""

    def f():
        mp = qml.shadow_expval(qml.X(0), seed=887, k=4)
        return qml.capture.measure(mp, shots=50)

    jaxpr = jax.make_jaxpr(f)()

    assert len(jaxpr.eqns) == 3
    assert jaxpr.eqns[0].primitive == qml.X._primitive

    assert jaxpr.eqns[1].primitive == ShadowExpvalMP._obs_primitive
    assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars
    assert jaxpr.eqns[1].params == {"seed": 887, "k": 4}

    am = jaxpr.eqns[1].outvars[0].aval
    assert isinstance(am, AbstractMeasurement)
    assert am.n_wires is None
    assert am._abstract_eval == ShadowExpvalMP._abstract_eval

    assert jaxpr.eqns[2].primitive == measure_prim
    assert jaxpr.eqns[2].params == {"num_device_wires": 0, "shots": qml.measurements.Shots(50)}
    assert jaxpr.eqns[2].invars == jaxpr.eqns[1].outvars
    out_aval = jaxpr.eqns[2].outvars[0].aval
    assert out_aval == jax.core.ShapedArray((), jax.numpy.float32)


@pytest.mark.parametrize("mtype, kwargs", [(VnEntropyMP, {"log_base": 2}), (PurityMP, {})])
def test_qinfo_measurements(mtype, kwargs):
    """Test the capture of a vn entropy and purity measurement."""

    def f():
        mp = mtype(wires=qml.wires.Wires([0, 1]), **kwargs)
        return qml.capture.measure(mp, num_device_wires=4)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive == mtype._wires_primitive
    assert jaxpr.eqns[0].params == kwargs
    assert len(jaxpr.eqns[0].invars) == 2
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 2
    assert mp._abstract_eval == mtype._abstract_eval

    assert jaxpr.eqns[1].primitive == measure_prim
    assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(None)}
    assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((), jax.numpy.float32)


def test_MutualInfo():
    """Test the capture of a vn entropy and purity measurement."""

    def f():
        mp = qml.mutual_info(wires0=[0, 1], wires1=[2, 3], log_base=2)
        return qml.capture.measure(mp, num_device_wires=4)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive == MutualInfoMP._wires_primitive
    assert jaxpr.eqns[0].params == {"log_base": 2, "n_wires0": 2}
    assert len(jaxpr.eqns[0].invars) == 4
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp._abstract_eval == MutualInfoMP._abstract_eval

    assert jaxpr.eqns[1].primitive == measure_prim
    assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(None)}
    assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((), jax.numpy.float32)


def test_ClassicalShadow():
    """Test that the classical shadow measurement can be captured."""

    def f():
        mp = qml.classical_shadow(wires=(0, 1, 2), seed=95)
        return qml.capture.measure(mp, shots=50, num_device_wires=4)

    jaxpr = jax.make_jaxpr(f)()

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 2

    assert jaxpr.eqns[0].primitive == ClassicalShadowMP._wires_primitive
    assert jaxpr.eqns[0].params == {"seed": 95}
    assert len(jaxpr.eqns[0].invars) == 3
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 3
    assert mp._abstract_eval == ClassicalShadowMP._abstract_eval

    assert jaxpr.eqns[1].primitive == measure_prim
    assert jaxpr.eqns[1].params == {"num_device_wires": 4, "shots": qml.measurements.Shots(50)}
    assert jaxpr.eqns[1].outvars[0].aval == jax.core.ShapedArray((2, 50, 3), np.int8)
