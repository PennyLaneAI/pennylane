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
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    DensityMatrixMP,
    ExpectationMP,
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

from pennylane.capture.primitives import (  # pylint: disable=wrong-import-position
    AbstractMeasurement,
)

pytestmark = [pytest.mark.jax, pytest.mark.capture]


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


class TestCounts:

    def test_counts_no_implementation(self):
        """Test that counts can't be measured and raises a NotImplementedError."""

        with pytest.raises(
            NotImplementedError,
            match=r"Counts has no execution implementation with program capture.",
        ):
            qml.counts()

    def test_warning_about_all_outcomes(self):
        """Test a warning is raised about all_outcomes=False"""

        def f():
            return qml.counts(all_outcomes=False)

        with pytest.warns(UserWarning, match="all_outcomes=True"):
            jax.make_jaxpr(f)()

    def test_counts_capture_jaxpr(self):
        """Test that counts can be captured into jaxpr."""

        def f():
            return qml.counts(wires=(0, 1), all_outcomes=True)

        jaxpr = jax.make_jaxpr(f)()
        jaxpr = jaxpr.jaxpr

        assert len(jaxpr.outvars) == 2

        assert jaxpr.eqns[0].primitive == CountsMP._wires_primitive
        assert len(jaxpr.eqns[0].invars) == 2

        assert isinstance(jaxpr.outvars[0].aval, AbstractMeasurement)
        keys_shape = jaxpr.outvars[0].aval.abstract_eval(num_device_wires=0, shots=50)
        assert keys_shape[0] == (2**2,)
        assert keys_shape[1] == int

        with pytest.raises(ValueError, match="finite shots are required"):
            jaxpr.outvars[0].aval.abstract_eval(num_device_wires=0, shots=None)

        assert isinstance(jaxpr.outvars[1].aval, AbstractMeasurement)
        keys_shape = jaxpr.outvars[1].aval.abstract_eval(num_device_wires=0, shots=50)
        assert keys_shape[0] == (2**2,)
        assert keys_shape[1] == int

        with pytest.raises(ValueError, match="finite shots are required"):
            jaxpr.outvars[1].aval.abstract_eval(num_device_wires=0, shots=None)

    def test_counts_capture_jaxpr_all_wires(self):
        """Test that counts can be captured into jaxpr."""

        def f():
            return qml.counts(all_outcomes=True)

        jaxpr = jax.make_jaxpr(f)()
        jaxpr = jaxpr.jaxpr

        assert len(jaxpr.outvars) == 2

        assert jaxpr.eqns[0].primitive == CountsMP._wires_primitive
        assert len(jaxpr.eqns[0].invars) == 0

        assert isinstance(jaxpr.outvars[0].aval, AbstractMeasurement)
        keys_shape = jaxpr.outvars[0].aval.abstract_eval(num_device_wires=3, shots=50)
        assert keys_shape[0] == (2**3,)
        assert keys_shape[1] == int

        with pytest.raises(ValueError, match="finite shots are required"):
            jaxpr.outvars[0].aval.abstract_eval(num_device_wires=3, shots=None)

        assert isinstance(jaxpr.outvars[1].aval, AbstractMeasurement)
        keys_shape = jaxpr.outvars[1].aval.abstract_eval(num_device_wires=3, shots=50)
        assert keys_shape[0] == (2**3,)
        assert keys_shape[1] == int

        with pytest.raises(ValueError, match="finite shots are required"):
            jaxpr.outvars[1].aval.abstract_eval(num_device_wires=3, shots=None)

    def test_qnode_integration(self):
        """Test that counts can integrate with capturing a qnode."""

        def w():
            @qml.qnode(qml.device("default.qubit", wires=2), shots=10)
            def c():
                return qml.counts(all_outcomes=True), qml.sample()

            r = c()
            assert isinstance(r, tuple)
            assert len(r) == 2
            assert isinstance(r[0], tuple)
            assert len(r[0]) == 2
            for i in (0, 1):
                assert r[0][i].shape == (4,)
                assert r[0][i].dtype == jax.numpy.int64

            assert r[1].shape == (10, 2)
            assert r[1].dtype == jax.numpy.int64

            return r

        jaxpr = jax.make_jaxpr(w)().jaxpr
        assert len(jaxpr.outvars) == 3


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
]


@pytest.mark.parametrize("func", creation_funcs)
def test_capture_and_eval(func):
    """Test that captured jaxpr can be evaluated to restore the initial measurement."""

    mp = func()

    jaxpr = jax.make_jaxpr(func)()
    out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0]

    qml.assert_equal(mp, out)


@pytest.mark.parametrize("state_wires, shape", [(None, 16), (qml.wires.Wires((0, 1, 2, 3, 4)), 32)])
def test_state(state_wires, shape):
    """Test the capture of a state measurement."""

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
    t = jax.numpy.complex128 if jax.config.jax_enable_x64 else jax.numpy.complex64
    assert shapes == jax.core.ShapedArray((shape,), t)


@pytest.mark.parametrize("wires, shape", [([0, 1], (4, 4)), ([], (16, 16))])
def test_density_matrix(wires, shape):
    """Test the capture of a density matrix."""

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
    t = jax.numpy.complex128 if jax.config.jax_enable_x64 else jax.numpy.complex64
    assert shapes[0] == jax.core.ShapedArray(shape, t)


@pytest.mark.parametrize("m_type", (ExpectationMP, VarianceMP))
class TestExpvalVar:
    """Tests for capturing an expectation value or variance."""

    def test_capture_obs(self, m_type):
        """Test that the expectation value of an observable can be captured."""

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
        t = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        assert shapes == jax.core.ShapedArray((), t)

    def test_capture_eigvals_wires(self, m_type):
        """Test that we can capture an expectation value of eigvals+wires."""

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
        t = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        assert shapes == jax.core.ShapedArray((), t)

    def test_simple_single_mcm(self, m_type):
        """Test that we can take the expectation value of a mid circuit measurement."""

        def f():
            m = qml.measure(0)
            return m_type(obs=m)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 2

        assert jaxpr.eqns[1].primitive == m_type._mcm_primitive
        aval1 = jaxpr.eqns[1].outvars[0].aval
        assert isinstance(aval1, AbstractMeasurement)
        assert aval1.n_wires == 1
        assert aval1._abstract_eval == m_type._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, num_device_wires=0, shots=qml.measurements.Shots(50)
        )[0]
        t = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        assert shapes == jax.core.ShapedArray((), t)


class TestProbs:

    @pytest.mark.parametrize(
        "wires, shape", [([0, 1, 2], 8), ([], 16), (jax.numpy.array([0, 1, 2]), 8)]
    )
    def test_wires(self, wires, shape):
        """Tests capturing probabilities on wires."""

        def f():
            return qml.probs(wires=wires)

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[0].primitive == ProbabilityMP._wires_primitive
        assert [x.val for x in jaxpr.eqns[0].invars] == list(wires)
        mp = jaxpr.eqns[0].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert mp.n_wires == len(wires)
        assert mp._abstract_eval == ProbabilityMP._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, shots=qml.measurements.Shots(50), num_device_wires=4
        )[0]
        assert shapes == jax.core.ShapedArray(
            (shape,), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        )

    def test_eigvals(self):
        """Test capturing probabilities with eigenvalues."""

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
            (4,), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        )

    def test_multiple_mcms(self):
        """Test measuring probabilities of multiple mcms."""

        def f():
            m1 = qml.measure(0)
            m2 = qml.measure(0)
            return qml.probs(op=[m1, m2])

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[2].primitive == ProbabilityMP._mcm_primitive
        out = jaxpr.eqns[2].outvars[0].aval
        assert isinstance(out, AbstractMeasurement)
        assert out.n_wires == 2
        assert out._abstract_eval == ProbabilityMP._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, shots=qml.measurements.Shots(50), num_device_wires=6
        )
        assert shapes[0] == jax.core.ShapedArray(
            (4,), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        )


class TestSample:

    @pytest.mark.parametrize(
        "wires, dim1_len",
        [
            ([0, 1, 2], 3),
            ([], 4),
            (1, 1),
            (jax.numpy.array([0, 1, 2]), 3),
            (np.array([0, 1, 2]), 3),
        ],
    )
    def test_wires(self, wires, dim1_len):
        """Tests capturing samples on wires."""
        if isinstance(wires, list):

            def f(*inner_wires):
                return qml.sample(wires=inner_wires)

            jaxpr = jax.make_jaxpr(f)(*wires)
        else:

            def f(inner_wire):
                return qml.sample(wires=inner_wire)

            jaxpr = jax.make_jaxpr(f)(wires)

        if not isinstance(wires, (jax.numpy.ndarray, np.ndarray)):
            assert len(jaxpr.eqns) == 1

        assert jaxpr.eqns[-1].primitive == SampleMP._wires_primitive
        assert [x.aval for x in jaxpr.eqns[0].invars] == jaxpr.in_avals
        mp = jaxpr.eqns[-1].outvars[0].aval
        assert isinstance(mp, AbstractMeasurement)
        assert (
            mp.n_wires == len(wires)
            if isinstance(wires, (list, jax.numpy.ndarray, np.ndarray))
            else 1
        )
        assert mp._abstract_eval == SampleMP._abstract_eval

        shapes = _get_shapes_for(
            *jaxpr.out_avals, shots=qml.measurements.Shots(50), num_device_wires=4
        )
        assert len(shapes) == 1
        shape = (50, dim1_len)
        assert shapes[0] == jax.core.ShapedArray(
            shape, jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        )

        with pytest.raises(ValueError, match="finite shots are required"):
            jaxpr.out_avals[0].abstract_eval(shots=None, num_device_wires=4)

    def test_eigvals(self):
        """Test capturing samples with eigenvalues."""

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
            (50,), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        )

    def test_multiple_mcms(self):
        """Test sampling from multiple mcms."""

        def f():
            m1 = qml.measure(0)
            m2 = qml.measure(0)
            return qml.sample(op=[m1, m2])

        jaxpr = jax.make_jaxpr(f)()

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[2].primitive == SampleMP._mcm_primitive
        out = jaxpr.eqns[2].outvars[0].aval
        assert isinstance(out, AbstractMeasurement)
        assert out.n_wires == 2
        assert out._abstract_eval == SampleMP._abstract_eval

        shapes = _get_shapes_for(*jaxpr.out_avals, shots=qml.measurements.Shots(50))
        assert shapes[0] == jax.core.ShapedArray(
            (50, 2), jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        )


def test_shadow_expval(seed):
    """Test that the shadow expval of an observable can be captured."""

    def f():
        return qml.shadow_expval(qml.X(0), seed=seed, k=4)

    jaxpr = jax.make_jaxpr(f)()

    assert len(jaxpr.eqns) == 2
    assert jaxpr.eqns[0].primitive == qml.X._primitive

    assert jaxpr.eqns[1].primitive == ShadowExpvalMP._obs_primitive
    assert jaxpr.eqns[0].outvars == jaxpr.eqns[1].invars
    assert jaxpr.eqns[1].params == {"seed": seed, "k": 4}

    am = jaxpr.eqns[1].outvars[0].aval
    assert isinstance(am, AbstractMeasurement)
    assert am.n_wires is None
    assert am._abstract_eval == ShadowExpvalMP._abstract_eval

    shapes = _get_shapes_for(*jaxpr.out_avals, num_device_wires=0, shots=qml.measurements.Shots(50))
    assert shapes[0] == jax.core.ShapedArray(
        (), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
    )


@pytest.mark.parametrize("mtype, kwargs", [(VnEntropyMP, {"log_base": 2}), (PurityMP, {})])
def test_vn_entropy_purity(mtype, kwargs):
    """Test the capture of a vn entropy and purity measurement."""

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
        (), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
    )


def test_mutual_info():
    """Test the capture of a mutual info and vn entanglement entropy measurement."""

    def f(w1, w2):
        return MutualInfoMP(wires=(qml.wires.Wires([w1, 1]), qml.wires.Wires([w2, 3])), log_base=2)

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
        (), jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
    )


def test_ClassicalShadow(seed):
    """Test that the classical shadow measurement can be captured."""

    def f():
        return qml.classical_shadow(wires=(0, 1, 2), seed=seed)

    jaxpr = jax.make_jaxpr(f)()

    jaxpr = jax.make_jaxpr(f)()
    assert len(jaxpr.eqns) == 1

    assert jaxpr.eqns[0].primitive == ClassicalShadowMP._wires_primitive
    assert jaxpr.eqns[0].params == {"seed": seed}
    assert len(jaxpr.eqns[0].invars) == 3
    mp = jaxpr.eqns[0].outvars[0].aval
    assert isinstance(mp, AbstractMeasurement)
    assert mp.n_wires == 3
    assert mp._abstract_eval == ClassicalShadowMP._abstract_eval

    shapes = _get_shapes_for(*jaxpr.out_avals, num_device_wires=4, shots=qml.measurements.Shots(50))
    assert shapes[0] == jax.core.ShapedArray((2, 50, 3), np.int8)
