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

AbstractMeasurement = _get_abstract_measurement()


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def _get_shapes_for(*measurements, shots=qml.measurements.Shots(None), num_device_wires=0):
    if jax.config.jax_enable_x64:
        dtype_map = {
            float: jax.numpy.float64,
            int: jax.numpy.int64,
            complex: jax.numpy.complex128,
        }
    else:
        dtype_map = {
            float: jax.numpy.float32,
            int: jax.numpy.int32,
            complex: jax.numpy.complex64,
        }

    shapes = []
    if not shots:
        shots = [None]

    for s in shots:
        for m in measurements:
            shape, dtype = m.abstract_eval(shots=s, num_device_wires=num_device_wires)
            shapes.append(jax.core.ShapedArray(shape, dtype_map.get(dtype, dtype)))
    return shapes


def test_abstract_measurement():
    """Tests for the AbstractMeasurement class."""
    am = AbstractMeasurement(ExpectationMP._abstract_eval, n_wires=2, has_eigvals=True)

    assert am.n_wires == 2
    assert am.has_eigvals is True

    expected_repr = "AbstractMeasurement(n_wires=2, has_eigvals=True)"
    assert repr(am) == expected_repr

    assert am.abstract_eval(2, 50) == ((), float)

    with pytest.raises(NotImplementedError):
        am.at_least_vspace()

    with pytest.raises(NotImplementedError):
        am.join(am)

    with pytest.raises(NotImplementedError):
        am.update(key="value")

    am2 = AbstractMeasurement(ExpectationMP._abstract_eval)
    expected_repr2 = "AbstractMeasurement(n_wires=None)"
    assert repr(am2) == expected_repr2

    assert am == am2
    assert hash(am) == hash("AbstractMeasurement")


def test_counts_no_measure():
    """Test that counts can't be measured and raises a NotImplementedError."""

    with pytest.raises(NotImplementedError, match=r"CountsMP returns a dictionary"):
        qml.counts()._abstract_eval()


def test_mid_measure_not_implemented():
    """Test that measure raises a NotImplementedError if capture is enabled."""
    with pytest.raises(NotImplementedError):
        qml.measure(0)


def test_primitive_none_behavior():
    """Test that if the obs primitive is None, the measurement can still
    be created, but it just won't be captured into jaxpr.
    """

    # pylint: disable=too-few-public-methods
    class MyMeasurement(qml.measurements.MeasurementProcess):
        pass

    MyMeasurement._obs_primitive = None

    def f():
        return MyMeasurement(wires=qml.wires.Wires((0, 1)))

    mp = f()
    assert isinstance(mp, MyMeasurement)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 0


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
    lambda: MidMeasureMP(qml.wires.Wires((0, 1))),
]


@pytest.mark.parametrize("func", creation_funcs)
def test_capture_and_eval(func):
    """Test that captured jaxpr can be evaluated to restore the initial measurement."""

    mp = func()

    jaxpr = jax.make_jaxpr(func)()
    out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0]

    assert qml.equal(mp, out)


@pytest.mark.parametrize("x64_mode", [True, False])
def test_mid_measure(x64_mode):
    """Test that mid circuit measurements can be captured and executed.x"""
    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    def f(w):
        return MidMeasureMP(qml.wires.Wires((w,)), reset=True, postselect=1)

    jaxpr = jax.make_jaxpr(f)(2)

    assert len(jaxpr.eqns) == 1
    assert jaxpr.eqns[0].primitive == MidMeasureMP._wires_primitive
    assert jaxpr.eqns[0].params == {"reset": True, "postselect": 1}
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 1
    assert mp._abstract_eval == MidMeasureMP._abstract_eval

    shapes = _get_shapes_for(*jaxpr.out_avals, shots=qml.measurements.Shots(1))
    assert shapes[0] == jax.core.ShapedArray((), jax.numpy.int64 if x64_mode else jax.numpy.int32)

    mp = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1)[0]
    assert mp == f(1)

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize(
    "x64_mode, expected", [(True, jax.numpy.complex128), (False, jax.numpy.complex64)]
)
@pytest.mark.parametrize("state_wires, shape", [(None, 16), (qml.wires.Wires((0, 1, 2, 3, 4)), 32)])
def test_state(x64_mode, expected, state_wires, shape):
    """Test the capture of a state measurement."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    def f():
        return StateMP(wires=state_wires)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 1

    assert jaxpr.eqns[0].primitive == StateMP._wires_primitive
    assert len(jaxpr.eqns[0].invars) == 0 if state_wires is None else 5
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 0 if state_wires is None else 5
    assert mp._abstract_eval == StateMP._abstract_eval

    shapes = _get_shapes_for(
        *jaxpr.out_avals, shots=qml.measurements.Shots(None), num_device_wires=4
    )[0]
    assert shapes == jax.core.ShapedArray((shape,), expected)
    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize(
    "x64_mode, expected", [(True, jax.numpy.complex128), (False, jax.numpy.complex64)]
)
@pytest.mark.parametrize("wires, shape", [([0, 1], (4, 4)), ([], (16, 16))])
def test_density_matrix(wires, shape, x64_mode, expected):
    """Test the capture of a density matrix."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    def f():
        return qml.density_matrix(wires=wires)

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 1

    assert jaxpr.eqns[0].primitive == DensityMatrixMP._wires_primitive
    assert len(jaxpr.eqns[0].invars) == len(wires)
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == len(wires)
    assert mp._abstract_eval == DensityMatrixMP._abstract_eval

    shapes = _get_shapes_for(
        *jaxpr.out_avals, shots=qml.measurements.Shots(None), num_device_wires=4
    )
    assert shapes[0] == jax.core.ShapedArray(shape, expected)
    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize(
    "x64_mode, expected", [(True, jax.numpy.float64), (False, jax.numpy.float32)]
)
@pytest.mark.parametrize("m_type", (ExpectationMP, VarianceMP))
class TestExpvalVar:
    """Tests for capturing an expectation value or variance."""

    def test_capture_obs(self, m_type, x64_mode, expected):
        """Test that the expectation value of an observable can be captured."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f():
            return m_type(obs=qml.X(0))

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.X._primitive

        assert jaxpr.eqns[1].primitive == m_type._obs_primitive
        assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars

        am = jaxpr.eqns[1].outvars[0].aval
        assert isinstance(am, AbstractMeasurement)
        assert am.n_wires is None
        assert am._abstract_eval == m_type._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, num_device_wires=0, shots=qml.measurements.Shots(50)
        )[0]
        assert shapes == jax.core.ShapedArray((), expected)
        jax.config.update("jax_enable_x64", initial_mode)

    def test_capture_eigvals_wires(self, m_type, x64_mode, expected):
        """Test that we can capture an expectation value of eigvals+wires."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f(eigs):
            return m_type(eigvals=eigs, wires=qml.wires.Wires((0, 1)))

        eigs = np.array([1.0, 0.5, -0.5, -1.0])
        jaxpr = jax.make_jaxpr(f)(eigs)

        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == m_type._wires_primitive
        assert jaxpr.eqns[0].params == {"has_eigvals": True}
        assert [x.val for x in jaxpr.eqns[0].invars[:-1]] == [0, 1]  # the wires
        assert jaxpr.eqns[0].invars[-1] == jaxpr.jaxpr.invars[0]  # the eigvals

        am = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(am, AbstractMeasurement)
        assert am.n_wires == 2
        assert am._abstract_eval == m_type._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, num_device_wires=0, shots=qml.measurements.Shots(50)
        )[0]
        assert shapes == jax.core.ShapedArray((), expected)
        jax.config.update("jax_enable_x64", initial_mode)

    def test_simple_single_mcm(self, m_type, x64_mode, expected):
        """Test that we can take the expectation value of a mid circuit measurement."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f():
            # using integer to represent classical mcm value
            return m_type(obs=1)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == m_type._mcm_primitive
        aval1 = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(aval1, AbstractMeasurement)
        assert aval1.n_wires == 1
        assert aval1._abstract_eval == m_type._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, num_device_wires=0, shots=qml.measurements.Shots(50)
        )[0]
        assert shapes == jax.core.ShapedArray((), expected)

        with pytest.raises(NotImplementedError):
            f()

        jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
class TestProbs:

    @pytest.mark.parametrize("wires, shape", [([0, 1, 2], 8), ([], 16)])
    def test_wires(self, wires, shape, x64_mode):
        """Tests capturing probabilities on wires."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f():
            return qml.probs(wires=wires)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == ProbabilityMP._wires_primitive
        assert [x.val for x in jaxpr.eqns[0].invars] == wires
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == len(wires)
        assert mp._abstract_eval == ProbabilityMP._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, shots=qml.measurements.Shots(50), num_device_wires=4
        )[0]
        assert shapes == jax.core.ShapedArray(
            (shape,), jax.numpy.float64 if x64_mode else jax.numpy.float32
        )

        jax.config.update("jax_enable_x64", initial_mode)

    def test_eigvals(self, x64_mode):
        """Test capturing probabilities with eigenvalues."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f(eigs):
            return ProbabilityMP(eigvals=eigs, wires=qml.wires.Wires((0, 1)))

        eigvals = np.array([-1.0, -0.5, 0.5, 1.0])
        jaxpr = jax.make_jaxpr(f)(eigvals)

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == ProbabilityMP._wires_primitive
        assert jaxpr.eqns[0].params == {"has_eigvals": True}
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == 2
        assert mp._abstract_eval == ProbabilityMP._abstract_eval

        shapes = _get_shapes_for(*jaxpr.out_avals)
        assert shapes[0] == jax.core.ShapedArray(
            (4,), jax.numpy.float64 if x64_mode else jax.numpy.float32
        )

        jax.config.update("jax_enable_x64", initial_mode)

    def test_multiple_mcms(self, x64_mode):
        """Test measuring probabilities of multiple mcms."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f(c1, c2):
            return qml.probs(op=[c1, c2])

        jaxpr = jax.make_jaxpr(f)(1, 2)

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == ProbabilityMP._mcm_primitive
        out = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(out, AbstractMeasurement)
        assert out.n_wires == 2
        assert out._abstract_eval == ProbabilityMP._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, shots=qml.measurements.Shots(50), num_device_wires=6
        )
        assert shapes[0] == jax.core.ShapedArray(
            (4,), jax.numpy.float64 if x64_mode else jax.numpy.float32
        )

        with pytest.raises(NotImplementedError):
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1, 2)

        jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
class TestSample:

    @pytest.mark.parametrize("wires, dim1_len", [([0, 1, 2], 3), ([], 4)])
    def test_wires(self, wires, dim1_len, x64_mode):
        """Tests capturing samples on wires."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f(*inner_wires):
            return qml.sample(wires=inner_wires)

        jaxpr = jax.make_jaxpr(f)(*wires)

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == SampleMP._wires_primitive
        assert [x.aval for x in jaxpr.eqns[0].invars] == jaxpr.in_avals
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == len(wires)
        assert mp._abstract_eval == SampleMP._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, shots=qml.measurements.Shots(50), num_device_wires=4
        )
        assert shapes[0] == jax.core.ShapedArray(
            (50, dim1_len), jax.numpy.int64 if x64_mode else jax.numpy.int32
        )

        with pytest.raises(ValueError, match="finite shots are required"):
            jaxpr.out_avals[0].abstract_eval(shots=None, num_device_wires=4)

        jax.config.update("jax_enable_x64", initial_mode)

    def test_eigvals(self, x64_mode):
        """Test capturing samples with eigenvalues."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f(eigs):
            return SampleMP(eigvals=eigs, wires=qml.wires.Wires((0, 1)))

        eigvals = np.array([-1.0, -0.5, 0.5, 1.0])
        jaxpr = jax.make_jaxpr(f)(eigvals)

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == SampleMP._wires_primitive
        assert jaxpr.eqns[0].params == {"has_eigvals": True}
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == 2
        assert mp._abstract_eval == SampleMP._abstract_eval

        shapes = _get_shapes_for(*jaxpr.out_avals, shots=qml.measurements.Shots(50))
        assert shapes[0] == jax.core.ShapedArray(
            (50,), jax.numpy.float64 if x64_mode else jax.numpy.float32
        )

        jax.config.update("jax_enable_x64", initial_mode)

    def test_multiple_mcms(self, x64_mode):
        """Test sampling from multiple mcms."""

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)

        def f():
            return qml.sample(op=[1, 2])

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == SampleMP._mcm_primitive
        out = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(out, AbstractMeasurement)
        assert out.n_wires == 2
        assert out._abstract_eval == SampleMP._abstract_eval

        shapes = _get_shapes_for(*jaxpr.out_avals, shots=qml.measurements.Shots(50))
        assert shapes[0] == jax.core.ShapedArray(
            (50, 2), jax.numpy.int64 if x64_mode else jax.numpy.int32
        )

        with pytest.raises(NotImplementedError):
            f()

        jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
def test_shadow_expval(x64_mode):
    """Test that the shadow expval of an observable can be captured."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    def f():
        return qml.shadow_expval(qml.X(0), seed=887, k=4)

    jaxpr = jax.make_jaxpr(f)()

    assert len(jaxpr.eqns) == 2
    assert jaxpr.eqns[0].primitive == qml.X._primitive

    assert jaxpr.eqns[1].primitive == ShadowExpvalMP._obs_primitive
    assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars
    assert jaxpr.eqns[1].params == {"seed": 887, "k": 4}

    am = jaxpr.eqns[1].outvars[0].aval
    assert isinstance(am, AbstractMeasurement)
    assert am.n_wires is None
    assert am._abstract_eval == ShadowExpvalMP._abstract_eval

    shapes = _get_shapes_for(*jaxpr.out_avals, num_device_wires=0, shots=qml.measurements.Shots(50))
    assert shapes[0] == jax.core.ShapedArray(
        (), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
@pytest.mark.parametrize("mtype, kwargs", [(VnEntropyMP, {"log_base": 2}), (PurityMP, {})])
def test_qinfo_measurements(mtype, kwargs, x64_mode):
    """Test the capture of a vn entropy and purity measurement."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    def f(w1, w2):
        return mtype(wires=qml.wires.Wires([w1, w2]), **kwargs)

    jaxpr = jax.make_jaxpr(f)(1, 2)
    assert len(jaxpr.eqns) == 1

    assert jaxpr.eqns[0].primitive == mtype._wires_primitive
    assert jaxpr.eqns[0].params == kwargs
    assert len(jaxpr.eqns[0].invars) == 2
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 2
    assert mp._abstract_eval == mtype._abstract_eval

    shapes = _get_shapes_for(
        *jaxpr.out_avals, num_device_wires=4, shots=qml.measurements.Shots(None)
    )
    assert shapes[0] == jax.core.ShapedArray(
        (), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
def test_MutualInfo(x64_mode):
    """Test the capture of a vn entropy and purity measurement."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)

    def f(w1, w2):
        return qml.mutual_info(wires0=[w1, 1], wires1=[w2, 3], log_base=2)

    jaxpr = jax.make_jaxpr(f)(0, 2)
    assert len(jaxpr.eqns) == 1

    assert jaxpr.eqns[0].primitive == MutualInfoMP._wires_primitive
    assert jaxpr.eqns[0].params == {"log_base": 2, "n_wires0": 2}
    assert len(jaxpr.eqns[0].invars) == 4
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp._abstract_eval == MutualInfoMP._abstract_eval

    shapes = _get_shapes_for(
        *jaxpr.out_avals, num_device_wires=4, shots=qml.measurements.Shots(None)
    )
    assert shapes[0] == jax.core.ShapedArray(
        (), jax.numpy.float64 if x64_mode else jax.numpy.float32
    )

    jax.config.update("jax_enable_x64", initial_mode)


def test_ClassicalShadow():
    """Test that the classical shadow measurement can be captured."""

    def f():
        return qml.classical_shadow(wires=(0, 1, 2), seed=95)

    jaxpr = jax.make_jaxpr(f)()

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 1

    assert jaxpr.eqns[0].primitive == ClassicalShadowMP._wires_primitive
    assert jaxpr.eqns[0].params == {"seed": 95}
    assert len(jaxpr.eqns[0].invars) == 3
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 3
    assert mp._abstract_eval == ClassicalShadowMP._abstract_eval

    shapes = _get_shapes_for(*jaxpr.out_avals, num_device_wires=4, shots=qml.measurements.Shots(50))
    assert shapes[0] == jax.core.ShapedArray((2, 50, 3), np.int8)
