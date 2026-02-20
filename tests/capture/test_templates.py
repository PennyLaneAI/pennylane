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
Integration tests for the capture of PennyLane templates into plxpr.
"""
import inspect
from itertools import combinations

# pylint: disable=protected-access
from typing import Any

import numpy as np
import pytest

import pennylane as qml
from pennylane import math

jax = pytest.importorskip("jax")
jnp = jax.numpy

pytestmark = [pytest.mark.jax, pytest.mark.capture]
original_op_bind_code = qml.operation.Operator._primitive_bind_call.__code__


def normalize_for_comparison(obj):
    """Normalize objects for comparison by converting tuples to lists recursively.

    In JAX 0.7.0, _make_hashable converts lists to tuples for hashability.
    This function reverses that for test comparisons.
    """
    # Don't normalize callables (functions, operators, etc.)
    if callable(obj):
        return obj

    # Recursively normalize dictionaries
    if isinstance(obj, dict):
        return {k: normalize_for_comparison(v) for k, v in obj.items()}

    # Convert tuples and lists to lists with normalized contents
    if isinstance(obj, (tuple, list)):
        return [normalize_for_comparison(item) for item in obj]

    return obj


unmodified_templates_cases = [
    (qml.AmplitudeEmbedding, (jnp.array([1.0, 0.0]), 2), {}),
    (qml.AmplitudeEmbedding, (jnp.eye(4)[2], [2, 3]), {"normalize": False}),
    (qml.AmplitudeEmbedding, (jnp.array([0.3, 0.1, 0.2]),), {"pad_with": 1.2, "wires": [0, 3]}),
    (qml.AngleEmbedding, (jnp.array([1.0, 0.0]), [2, 3]), {}),
    (qml.AngleEmbedding, (jnp.array([0.4]), [0]), {"rotation": "X"}),
    (qml.AngleEmbedding, (jnp.array([0.3, 0.1, 0.2]),), {"rotation": "Z", "wires": [0, 2, 3]}),
    (qml.BasisEmbedding, (jnp.array([1, 0]), [2, 3]), {}),
    pytest.param(
        qml.BasisEmbedding,
        (),
        {"features": jnp.array([1, 0]), "wires": [2, 3]},
        marks=pytest.mark.xfail(reason="arrays should never have been in the metadata [sc-104808]"),
    ),
    (qml.BasisEmbedding, (6, [0, 5, 2]), {}),
    (qml.BasisEmbedding, (jnp.array([1, 0, 1]),), {"wires": [0, 2, 3]}),
    (qml.IQPEmbedding, (jnp.array([2.3, 0.1]), [2, 0]), {}),
    (qml.IQPEmbedding, (jnp.array([0.4, 0.2, 0.1]), [2, 1, 0]), {"pattern": [[2, 0], [1, 0]]}),
    (qml.IQPEmbedding, (jnp.array([0.4, 0.1]), [0, 10]), {"n_repeats": 3, "pattern": None}),
    (qml.QAOAEmbedding, (jnp.array([1.0, 0.0]), jnp.ones((3, 3)), [2, 3]), {}),
    (qml.QAOAEmbedding, (jnp.array([0.4]), jnp.ones((2, 1)), [0]), {"local_field": "X"}),
    (
        qml.QAOAEmbedding,
        (jnp.array([0.3, 0.1, 0.2]), jnp.zeros((2, 6))),
        {"local_field": "Z", "wires": [0, 2, 3]},
    ),
    (qml.BasicEntanglerLayers, (jnp.ones((5, 2)), [2, 3]), {}),
    (qml.BasicEntanglerLayers, (jnp.ones((2, 1)), [0]), {"rotation": "X"}),
    (
        qml.BasicEntanglerLayers,
        (jnp.array([[0.3, 0.1, 0.2]]),),
        {"rotation": "Z", "wires": [0, 2, 3]},
    ),
    # Need to fix GateFabric positional args: Currently have to pass init_state as kwarg if we want to pass wires as kwarg
    # https://github.com/PennyLaneAI/pennylane/issues/5521
    (qml.GateFabric, (jnp.ones((3, 1, 2)), [2, 3, 0, 1]), {"init_state": [0, 1, 1, 0]}),
    pytest.param(
        qml.GateFabric,
        (jnp.zeros((2, 3, 2)),),
        {"include_pi": False, "wires": list(range(8)), "init_state": jnp.ones(8)},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    # (qml.GateFabric, (jnp.zeros((2, 3, 2)), jnp.ones(8)), {"include_pi": False, "wires": list(range(8))}), # Can't even init
    # (qml.GateFabric, (jnp.ones((5, 2, 2)), list(range(6)), jnp.array([0, 0, 1, 1, 0, 1])), {"include_pi": True}), # Can't trace
    # https://github.com/PennyLaneAI/pennylane/issues/5522
    # (qml.ParticleConservingU1, (jnp.ones((3, 1, 2)), [2, 3]), {}),
    (qml.ParticleConservingU1, (jnp.ones((3, 1, 2)), [2, 3]), {"init_state": [0, 1]}),
    pytest.param(
        qml.ParticleConservingU1,
        (jnp.zeros((5, 3, 2)),),
        {"wires": [0, 1, 2, 3], "init_state": jnp.ones(4)},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    # https://github.com/PennyLaneAI/pennylane/issues/5522
    # (qml.ParticleConservingU2, (jnp.ones((3, 3)), [2, 3]), {}),
    (qml.ParticleConservingU2, (jnp.ones((3, 3)), [2, 3]), {"init_state": [0, 1]}),
    pytest.param(
        qml.ParticleConservingU2,
        (jnp.zeros((5, 7)),),
        {"wires": [0, 1, 2, 3], "init_state": jnp.ones(4)},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.RandomLayers, (jnp.ones((3, 3)), [2, 3]), {}),
    (qml.RandomLayers, (jnp.ones((3, 3)),), {"wires": [3, 2, 1], "ratio_imprim": 0.5}),
    pytest.param(
        qml.RandomLayers,
        (),
        {"weights": jnp.ones((3, 3)), "wires": [3, 2, 1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.RandomLayers, (jnp.ones((3, 3)),), {"wires": [3, 2, 1], "rotations": (qml.RX, qml.RZ)}),
    (qml.RandomLayers, (jnp.ones((3, 3)), [0, 1]), {"rotations": (qml.RX, qml.RZ), "seed": 41}),
    (qml.SimplifiedTwoDesign, (jnp.ones(2), jnp.zeros((3, 1, 2)), [2, 3]), {}),
    (qml.SimplifiedTwoDesign, (jnp.ones(3), jnp.zeros((3, 2, 2))), {"wires": [0, 1, 2]}),
    pytest.param(
        qml.SimplifiedTwoDesign,
        (jnp.ones(2),),
        {"weights": jnp.zeros((3, 1, 2)), "wires": [0, 2]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    pytest.param(
        qml.SimplifiedTwoDesign,
        (),
        {"initial_layer_weights": jnp.ones(2), "weights": jnp.zeros((3, 1, 2)), "wires": [0, 2]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.StronglyEntanglingLayers, (jnp.ones((3, 2, 3)), [2, 3]), {"ranges": [1, 1, 1]}),
    (
        qml.StronglyEntanglingLayers,
        (jnp.ones((1, 3, 3)),),
        {"wires": [3, 2, 1], "imprimitive": qml.CZ},
    ),
    pytest.param(
        qml.StronglyEntanglingLayers,
        (),
        {"weights": jnp.ones((3, 3, 3)), "wires": [3, 2, 1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.ArbitraryStatePreparation, (jnp.ones(6), [2, 3]), {}),
    (qml.ArbitraryStatePreparation, (jnp.zeros(14),), {"wires": [3, 2, 0]}),
    pytest.param(
        qml.ArbitraryStatePreparation,
        (),
        {"weights": jnp.ones(2), "wires": [1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.CosineWindow, ([2, 3],), {}),
    (qml.CosineWindow, (), {"wires": [2, 0, 1]}),
    (qml.MottonenStatePreparation, (jnp.ones(4) / 2, [2, 3]), {}),
    (
        qml.MottonenStatePreparation,
        (jnp.ones(8) / jnp.sqrt(8),),
        {"wires": [3, 2, 0]},
    ),
    pytest.param(
        qml.MottonenStatePreparation,
        (),
        {"state_vector": jnp.array([1.0, 0.0]), "wires": [1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.AQFT, (1, [0, 1, 2]), {}),
    (qml.AQFT, (2,), {"wires": [0, 1, 2, 3]}),
    (qml.AQFT, (), {"order": 2, "wires": [0, 2, 3, 1]}),
    (qml.QFT, ([0, 1],), {}),
    (qml.QFT, (), {"wires": [0, 1]}),
    (qml.ArbitraryUnitary, (jnp.ones(15), [2, 3]), {}),
    (qml.ArbitraryUnitary, (jnp.zeros(15),), {"wires": [3, 2]}),
    pytest.param(
        qml.ArbitraryUnitary,
        (),
        {"weights": jnp.ones(3), "wires": [1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.FABLE, (jnp.eye(4), [2, 3, 0, 1, 5]), {}),
    (qml.FABLE, (jnp.ones((4, 4)),), {"wires": [0, 3, 2, 1, 9]}),
    pytest.param(
        qml.FABLE,
        (),
        {"input_matrix": jnp.array([[1, 1], [1, -1]]) / np.sqrt(2), "wires": [1, 10, 17]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qml.FermionicSingleExcitation, (0.421,), {"wires": [0, 3, 2]}),
    (qml.FlipSign, (7,), {"wires": [0, 3, 2]}),
    (qml.FlipSign, (np.array([1, 0, 0]), [0, 1, 2]), {}),
    (
        qml.kUpCCGSD,
        (jnp.ones((1, 6)), [0, 1, 2, 3]),
        {"k": 1, "delta_sz": 0, "init_state": [1, 1, 0, 0]},
    ),
    (qml.Permute, (np.array([1, 2, 0]), [0, 1, 2]), {}),
    (qml.Permute, (np.array([1, 2, 0]),), {"wires": [0, 1, 2]}),
    (
        qml.TwoLocalSwapNetwork,
        ([0, 1, 2, 3, 4],),
        {"acquaintances": lambda index, wires, param=None: qml.CNOT(index)},
    ),
    (qml.GroverOperator, (), {"wires": [0, 1]}),
    (qml.GroverOperator, ([0, 1],), {}),
    (
        qml.UCCSD,
        (jnp.ones(3), [2, 3, 0, 1]),
        {"s_wires": [[0], [1]], "d_wires": [[[2], [3]]], "init_state": [0, 1, 1, 0]},
    ),
    (qml.TemporaryAND, (), ({"wires": [0, 1, 2], "control_values": [0, 1]})),
    (qml.TemporaryAND, ([0, 1, 2],), ({"control_values": [0, 1]})),
]


@pytest.mark.parametrize("template, args, kwargs", unmodified_templates_cases)
def test_unmodified_templates(template, args, kwargs):
    """Test that templates with unmodified primitive binds are captured as expected."""

    # Make sure the input data is valid
    template(*args, **kwargs)

    # Make sure the template actually is not modified in its primitive binding function
    assert template._primitive_bind_call.__code__ == original_op_bind_code

    def fn(*args):
        template(*args, **kwargs)

    jaxpr = jax.make_jaxpr(fn)(*args)

    # Check basic structure of jaxpr: single equation with template primitive
    assert len(jaxpr.eqns) == 1
    eqn = jaxpr.eqns[0]
    assert eqn.primitive == template._primitive

    # Check that all arguments are passed correctly, taking wires parsing into account
    # Also, store wires for later
    if "wires" in kwargs:
        # If wires are in kwargs, they are not invars to the jaxpr
        num_invars_wo_wires = len(eqn.invars) - len(kwargs["wires"])
        assert eqn.invars[:num_invars_wo_wires] == jaxpr.jaxpr.invars
        wires = kwargs.pop("wires")
    else:
        # If wires are in args, they are also invars to the jaxpr
        assert eqn.invars == jaxpr.jaxpr.invars
        wires = args[-1]

    # Check outvars; there should only be the DropVar returned by the template
    assert len(eqn.outvars) == 1
    assert isinstance(eqn.outvars[0], jax.core.DropVar)

    # Check that `n_wires` is inferred correctly
    if isinstance(wires, int):
        wires = (wires,)
    assert eqn.params.pop("n_wires") == len(wires)
    # Check that remaining kwargs are passed properly to the eqn
    # JAX 0.7.0 converts lists to tuples for hashability, so normalize both sides
    assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)


# Only add a template to the following list if you manually added a test for it to
# TestModifiedTemplates below.
tested_modified_templates = [
    qml.TrotterProduct,
    qml.AllSinglesDoubles,
    qml.AmplitudeAmplification,
    qml.ApproxTimeEvolution,
    qml.BasisRotation,
    qml.BBQRAM,
    qml.CommutingEvolution,
    qml.ControlledSequence,
    qml.FermionicDoubleExcitation,
    qml.HilbertSchmidt,
    qml.HybridQRAM,
    qml.IQP,
    qml.LocalHilbertSchmidt,
    qml.QDrift,
    qml.QSVT,
    qml.QuantumMonteCarlo,
    qml.QuantumPhaseEstimation,
    qml.Qubitization,
    qml.Reflection,
    qml.Select,
    qml.SelectOnlyQRAM,
    qml.MERA,
    qml.MPS,
    qml.TTN,
    qml.QROM,
    qml.PhaseAdder,
    qml.Adder,
    qml.SemiAdder,
    qml.Multiplier,
    qml.OutMultiplier,
    qml.OutAdder,
    qml.ModExp,
    qml.OutPoly,
    qml.Superposition,
    qml.MPSPrep,
    qml.GQSP,
    qml.QROMStatePreparation,
    qml.MultiplexerStatePreparation,
    qml.SelectPauliRot,
]


# pylint: disable=too-many-public-methods
class TestModifiedTemplates:
    """Test that templates with custom primitive binds are captured as expected."""

    @pytest.mark.parametrize(
        "template, kwargs",
        [
            (qml.TrotterProduct, {"order": 2}),
            (qml.ApproxTimeEvolution, {"n": 2}),
            (qml.CommutingEvolution, {"frequencies": (1.2, 2)}),
            (qml.QDrift, {"n": 2, "seed": 10}),
        ],
    )
    def test_evolution_ops(self, template, kwargs):
        """Test the primitive bind call of Hamiltonian time evolution templates."""

        coeffs = [0.25, 0.75]

        def qfunc(coeffs):
            ops = [qml.X(0), qml.Z(0)]
            H = qml.dot(coeffs, ops)
            template(H, 2.4, **kwargs)

        # Validate inputs
        qfunc(coeffs)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(coeffs)

        assert len(jaxpr.eqns) == 6

        # due to flattening and unflattening H
        assert jaxpr.eqns[0].primitive == qml.X._primitive
        assert jaxpr.eqns[1].primitive == qml.Z._primitive
        assert jaxpr.eqns[2].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[3].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[4].primitive == qml.ops.Sum._primitive

        eqn = jaxpr.eqns[5]
        assert eqn.primitive == template._primitive
        assert eqn.invars[0] == jaxpr.eqns[4].outvars[0]  # the sum op
        assert eqn.invars[1].val == 2.4

        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, coeffs[0], coeffs[1])

        assert len(q) == 1
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)
        assert q.queue[0] == template(H, time=2.4, **kwargs)

    def test_amplitude_amplification(self):
        """Test the primitive bind call of AmplitudeAmplification."""

        U = qml.Hadamard(0)
        O = qml.FlipSign(1, 0)
        iters = 3

        kwargs = {"iters": iters, "fixed_point": False, "p_min": 0.4}

        def qfunc(U, O):
            qml.AmplitudeAmplification(U, O, **kwargs)

        # Validate inputs
        qfunc(U, O)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(U, O)

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[1].primitive == qml.FlipSign._primitive

        eqn = jaxpr.eqns[2]
        assert eqn.primitive == qml.AmplitudeAmplification._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]  # Hadamard
        assert eqn.invars[1] == jaxpr.eqns[1].outvars[0]  # FlipSign
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert q.queue[0] == qml.AmplitudeAmplification(U, O, **kwargs)

    def test_basis_rotation(self):
        """Test the primitive bind call of BasisRotation."""

        mat = np.eye(4)
        wires = [0, 5]

        def qfunc(wires, mat):
            qml.BasisRotation(wires, mat, check=True)

        # Validate inputs
        qfunc(wires, mat)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(wires, mat)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.BasisRotation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params["check"] is True
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *wires, mat)

        assert len(q) == 1
        assert q.queue[0] == qml.BasisRotation(wires=wires, unitary_matrix=mat, check=True)

    def test_controlled_sequence(self):
        """Test the primitive bind call of ControlledSequence."""

        assert (
            qml.ControlledSequence._primitive_bind_call.__code__
            == qml.ops.op_math.SymbolicOp._primitive_bind_call.__code__
        )

        base = qml.RX(0.5, 0)
        control = [1, 5]

        def fn(base):
            qml.ControlledSequence(base, control=control)

        # Validate inputs
        fn(base)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(fn)(base)

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qml.RX._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qml.ControlledSequence._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars
        # JAX 0.7.0 converts lists to tuples for hashability
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(
            {"control": control}
        )
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

        assert len(q) == 1  # One for each control
        assert q.queue[0] == qml.ControlledSequence(base, control)

    def test_fermionic_double_excitation(self):
        """Test the primitive bind call of FermionicDoubleExcitation."""

        weight = 0.251

        kwargs = {"wires1": [0, 6], "wires2": [2, 3]}

        def qfunc(weight):
            qml.FermionicDoubleExcitation(weight, **kwargs)

        # Validate inputs
        qfunc(weight)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(weight)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.FermionicDoubleExcitation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, weight)

        assert len(q) == 1
        assert q.queue[0] == qml.FermionicDoubleExcitation(weight, **kwargs)

    @pytest.mark.parametrize("template", [qml.HilbertSchmidt, qml.LocalHilbertSchmidt])
    def test_hilbert_schmidt(self, template):
        """Test the primitive bind call of HilbertSchmidt and LocalHilbertSchmidt."""

        def qfunc(v_params):
            U = qml.Hadamard(0)
            V = qml.RZ(v_params[0], wires=1)
            template(V, U)

        v_params = jnp.array([0.1])
        # Validate inputs
        qfunc(v_params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(v_params)

        assert len(jaxpr.eqns) == 5
        assert jaxpr.eqns[0].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[-2].primitive == qml.RZ._primitive

        eqn = jaxpr.eqns[-1]
        assert eqn.primitive == template._primitive
        assert eqn.params == {"num_v_ops": 1}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, v_params)

        assert len(q) == 1

        U = qml.Hadamard(0)
        V = qml.RZ(v_params[0], wires=1)
        assert qml.equal(q.queue[0], template(V, U)) is True

    @pytest.mark.parametrize("template", [qml.HilbertSchmidt, qml.LocalHilbertSchmidt])
    def test_hilbert_schmidt_multiple_ops(self, template):
        """Test the primitive bind call of HilbertSchmidt and LocalHilbertSchmidt with multiple ops."""

        def qfunc(v_params):
            U = [qml.Hadamard(0), qml.Hadamard(1)]
            V = [qml.RZ(v_params[0], wires=2), qml.RX(v_params[1], wires=3)]
            template(V, U)

        v_params = jnp.array([0.1, 0.2])
        # Validate inputs
        qfunc(v_params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(v_params)

        assert len(jaxpr.eqns) == 9
        assert jaxpr.eqns[0].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[-5].primitive == qml.RZ._primitive
        assert jaxpr.eqns[-2].primitive == qml.RX._primitive

        eqn = jaxpr.eqns[-1]
        assert eqn.primitive == template._primitive
        assert eqn.params == {"num_v_ops": 2}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, v_params)

        assert len(q) == 1

        U = [qml.Hadamard(0), qml.Hadamard(1)]
        V = [qml.RZ(v_params[0], wires=2), qml.RX(v_params[1], wires=3)]
        assert qml.equal(q.queue[0], template(V, U)) is True

    def test_iqp(self):
        """Test the primitive bind call of IQP."""

        pattern = []
        for weight in math.arange(1, 2):
            for gate in combinations(math.arange(4), weight):
                pattern.append(tuple(tuple(gate)))
        pattern = tuple(pattern)

        kwargs = {
            "num_wires": 4,
            "weights": tuple(math.random.uniform(0, 2 * np.pi, 4)),
            "pattern": pattern,
            "spin_sym": True,
        }

        def qfunc():
            qml.IQP(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.IQP._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.IQP(**kwargs))

    @pytest.mark.parametrize("template", [qml.MERA, qml.MPS, qml.TTN])
    def test_tensor_networks(self, template):
        """Test the primitive bind call of MERA, MPS, and TTN."""

        def block(weights, wires):
            return [
                qml.CNOT(wires),
                qml.RY(weights[0], wires[0]),
                qml.RY(weights[1], wires[1]),
            ]

        wires = list(range(4))
        n_block_wires = 2
        n_blocks = template.get_n_blocks(wires, n_block_wires)

        kwargs = {
            "wires": wires,
            "n_block_wires": n_block_wires,
            "block": block,
            "n_params_block": 2,
            "template_weights": [[0.1, -0.3]] * n_blocks,
        }

        def qfunc():
            template(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == template._primitive

        expected_params = {
            "n_block_wires": n_block_wires,
            "block": block,
            "n_params_block": 2,
            "template_weights": kwargs["template_weights"],
            "n_wires": 4,
        }
        if template is qml.MPS:
            expected_params["offset"] = None
        # JAX 0.7.0 converts lists to tuples for hashability
        actual_params = {k: v for k, v in eqn.params.items() if k in expected_params}
        assert normalize_for_comparison(actual_params) == normalize_for_comparison(expected_params)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], template(**kwargs))

    def test_qsvt(self):
        """Test the primitive bind call of QSVT."""

        def qfunc(A):
            block_encode = qml.BlockEncode(A, wires=[0, 1])
            shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
            qml.QSVT(block_encode, projectors=shifts)

        A = np.array([[0.1]])
        # Validate inputs
        qfunc(A)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(A)

        assert len(jaxpr.eqns) == 5

        assert jaxpr.eqns[0].primitive == qml.BlockEncode._primitive

        eqn = jaxpr.eqns[-1]
        assert eqn.primitive == qml.QSVT._primitive
        for i in range(4):
            assert eqn.invars[i] == jaxpr.eqns[i].outvars[0]
        assert eqn.params == {}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, A)

        assert len(q) == 1
        block_encode = qml.BlockEncode(A, wires=[0, 1])
        shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
        assert q.queue[0] == qml.QSVT(block_encode, shifts)

    def test_mps_prep(self):
        """Test the primitive bind call of MPSPrep."""

        mps = [
            np.array([[0.0, 0.107], [0.994, 0.0]]),
            np.array(
                [
                    [[0.0, 0.0, 0.0, -0.0], [1.0, 0.0, 0.0, -0.0]],
                    [[0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 0.0, -0.0]],
                ]
            ),
            np.array(
                [
                    [[-1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 0.0]],
                ]
            ),
            np.array([[-1.0, -0.0], [-0.0, -1.0]]),
        ]
        wires = [0, 1, 2]

        def qfunc(mps):
            qml.MPSPrep(mps=mps, wires=wires)

        # Validate inputs
        qfunc(mps)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(mps)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.MPSPrep._primitive
        assert eqn.invars[:4] == jaxpr.jaxpr.invars
        assert [invar.val for invar in eqn.invars[4:]] == [0, 1, 2]
        expected_params = {
            "n_wires": 3,
            "work_wires": None,
            "right_canonicalize": False,
        }
        actual_params = {k: v for k, v in eqn.params.items() if k in expected_params}
        assert actual_params == expected_params
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *mps)

        assert len(q) == 1
        assert q.queue[0] == qml.MPSPrep(mps=mps, wires=wires)

    def test_all_singles_doubles(self):
        arguments = (
            jnp.array([-2.8, 0.5]),
            jnp.array([1, 2, 3, 4]),
            jnp.array([1, 1, 0, 0]),
        )
        keyword_args = (
            jnp.array([[0, 2]]),
            jnp.array([[0, 1, 2, 3]]),
        )
        params = (*arguments, *keyword_args)

        def qfunc(weights, wires, hf_state, singles, doubles):
            qml.AllSinglesDoubles(weights, wires, hf_state, singles=singles, doubles=doubles)

        # Validate inputs
        qfunc(*params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(*params)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

        assert len(q) == 1
        assert q.queue[0] == qml.AllSinglesDoubles(*params)

    def test_quantum_monte_carlo(self):
        """Test the primitive bind call of QuantumMonteCarlo."""

        # This test follows the docstring example

        from scipy.stats import norm

        m = 5
        M = 2**m
        n = 10

        xs = np.linspace(-np.pi, np.pi, M)
        probs = np.array([norm().pdf(x) for x in xs])
        probs /= np.sum(probs)

        def func(i):
            return np.sin(xs[i]) ** 2

        target_wires = range(m + 1)
        estimation_wires = range(m + 1, n + m + 1)

        kwargs = {"func": func, "num_target_wires": 6}

        def qfunc(probs, target_wires, estimation_wires):
            qml.QuantumMonteCarlo(probs, func, target_wires, estimation_wires)

        # Validate inputs
        qfunc(probs, target_wires, estimation_wires)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(probs, list(target_wires), list(estimation_wires))

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.QuantumMonteCarlo._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        relevant_keys = {"func", "num_target_wires"}
        actual_params = {k: v for k, v in eqn.params.items() if k in relevant_keys}
        assert normalize_for_comparison(actual_params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, probs, *target_wires, *estimation_wires)

        assert len(q) == 1
        assert q.queue[0] == qml.QuantumMonteCarlo(probs, func, target_wires, estimation_wires)

    def test_qubitization(self):
        """Test the primitive bind call of Qubitization."""

        hamiltonian = qml.dot([0.5, 1.2, -0.84], [qml.X(2), qml.Hadamard(3), qml.Z(2) @ qml.Y(3)])
        kwargs = {"hamiltonian": hamiltonian, "control": [0, 1]}

        def qfunc():
            qml.Qubitization(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.Qubitization._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Qubitization(**kwargs))

    def test_bbqram(self):
        """Test the primitve bind call of BBQRAM."""

        kwargs = {
            "data": ((0, 1, 0), (1, 1, 1), (1, 1, 0), (0, 0, 0)),
            "control_wires": (0, 1),
            "target_wires": (2, 3, 4),
            "work_wires": tuple([5] + [6, 7, 8] + [12, 13, 14] + [9, 10, 11]),
        }

        def qfunc():
            qml.BBQRAM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.BBQRAM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.BBQRAM(**kwargs))

    def test_select_only_qram(self):
        """Test the primitve bind call of SelectOnlyQRAM."""

        kwargs = {
            "data": (
                (0, 1, 0),
                (1, 1, 1),
                (1, 1, 0),
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 1),
                (1, 1, 0),
                (0, 0, 0),
            ),
            "control_wires": (0, 1),
            "target_wires": (2, 3, 4),
            "select_wires": (12),
            "select_value": 1,
        }

        def qfunc():
            qml.SelectOnlyQRAM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.SelectOnlyQRAM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.SelectOnlyQRAM(**kwargs))

    def test_hybrid_qram(self):
        """Test the primitve bind call of HybridQRAM."""

        kwargs = {
            "data": ((0, 1, 0), (1, 1, 1), (1, 1, 0), (0, 0, 0)),
            "control_wires": (0, 1),
            "target_wires": (2, 3, 4),
            "work_wires": tuple([5, 6, 7, 8, 12, 13, 14, 15, 9, 10, 11]),
            "k": 0,
        }

        def qfunc():
            qml.HybridQRAM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.HybridQRAM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.HybridQRAM(**kwargs))

    def test_qrom(self):
        """Test the primitive bind call of QROM."""

        kwargs = {
            "data": ((0,), (1,)),
            "control_wires": [0],
            "target_wires": [1],
            "work_wires": None,
        }

        def qfunc():
            qml.QROM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.QROM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.QROM(**kwargs))

    @pytest.mark.xfail(reason="QROMStatePreparation uses array in metadata, [sc-104808]")
    def test_qrom_state_prep(self):
        """Test the primitive bind call of QROMStatePreparation."""

        kwargs = {
            "state_vector": np.array([1 / 2, -1 / 2, 1 / 2, 1j / 2]),
            "precision_wires": [0, 1, 2, 3],
            "work_wires": [4, 5, 6, 7],
            "wires": [8, 9],
        }

        def qfunc():
            qml.QROMStatePreparation(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.QROMStatePreparation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.QROMStatePreparation(**kwargs))

    def test_multiplexer_state_prep(self):
        """Test the primitive bind call of MultiplexerStatePreparation."""

        state_vector = np.array([1 / 2, -1 / 2, 1 / 2, 1j / 2])
        kwargs = {
            "wires": (8, 9),
        }

        def qfunc(state_vector):
            qml.MultiplexerStatePreparation(state_vector, **kwargs)

        qfunc(state_vector)
        jaxpr = jax.make_jaxpr(qfunc)(state_vector)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.MultiplexerStatePreparation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, state_vector)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.MultiplexerStatePreparation(state_vector, **kwargs))

    def test_multiplexed_rotation(self):
        """Test the primitive bind call of SelectPauliRot."""

        angles = np.arange(1, 9)
        kwargs = {
            "control_wires": [0, 1, 2],
            "target_wire": 3,
            "rot_axis": "X",
        }

        def qfunc(angles):
            qml.SelectPauliRot(angles, **kwargs)

        # Validate inputs
        qfunc(angles)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(angles)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.SelectPauliRot._primitive
        assert eqn.invars[:1] == jaxpr.jaxpr.invars
        assert [invar.val for invar in eqn.invars[1:]] == [0, 1, 2, 3]
        assert eqn.params == {"n_wires": 4, "rot_axis": "X"}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, angles)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.SelectPauliRot(angles, **kwargs))

    def test_phase_adder(self):
        """Test the primitive bind call of PhaseAdder."""

        kwargs = {
            "k": 3,
            "x_wires": [0, 1],
            "mod": None,
            "work_wire": None,
        }

        def qfunc():
            qml.PhaseAdder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.PhaseAdder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.PhaseAdder(**kwargs))

    def test_adder(self):
        """Test the primitive bind call of Adder."""

        kwargs = {
            "k": 3,
            "x_wires": [0, 1],
            "mod": None,
            "work_wires": None,
        }

        def qfunc():
            qml.Adder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.Adder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Adder(**kwargs))

    def test_semiadder(self):
        """Test the primitive bind call of SemiAdder."""

        kwargs = {
            "x_wires": [0, 1, 2],
            "y_wires": [3, 4, 5],
            "work_wires": [6, 7],
        }

        def qfunc():
            qml.SemiAdder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.SemiAdder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.SemiAdder(**kwargs))

    def test_multiplier(self):
        """Test the primitive bind call of Multiplier."""

        kwargs = {
            "k": 3,
            "x_wires": [0, 1],
            "mod": None,
            "work_wires": [2, 3],
        }

        def qfunc():
            qml.Multiplier(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.Multiplier._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Multiplier(**kwargs))

    def test_out_multiplier(self):
        """Test the primitive bind call of OutMultiplier."""

        kwargs = {
            "x_wires": [0, 1],
            "y_wires": [2, 3],
            "output_wires": [4, 5],
            "mod": None,
            "work_wires": None,
        }

        def qfunc():
            qml.OutMultiplier(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.OutMultiplier._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.OutMultiplier(**kwargs))

    def test_out_adder(self):
        """Test the primitive bind call of OutAdder."""

        kwargs = {
            "x_wires": [0, 1],
            "y_wires": [2, 3],
            "output_wires": [4, 5],
            "mod": None,
            "work_wires": None,
        }

        def qfunc():
            qml.OutAdder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.OutAdder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.OutAdder(**kwargs))

    def test_mod_exp(self):
        """Test the primitive bind call of ModExp."""

        kwargs = {
            "x_wires": [0, 1],
            "output_wires": [4, 5],
            "base": 3,
            "mod": None,
            "work_wires": [2, 3],
        }

        def qfunc():
            qml.ModExp(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.ModExp._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.ModExp(**kwargs))

    def test_out_poly(self):
        """Test the primitive bind call of OutPoly."""

        def func(x, y):
            return x**2 + y

        kwargs = {
            "polynomial_function": func,
            "input_registers": [[0, 1], [2, 3]],
            "output_wires": [4, 5],
            "mod": 3,
            "work_wires": [6, 7],
        }

        def qfunc():
            qml.OutPoly(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.OutPoly._primitive
        assert eqn.invars == jaxpr.jaxpr.invars

        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)

        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.OutPoly(**kwargs))

    def test_gqsp(self):
        """Test the primitive bind call of GQSP."""

        def qfunc(unitary, angles):
            qml.GQSP(unitary, angles, control=0)

        angles = np.ones([3, 3])
        unitary = qml.RX(1, wires=1)
        # Validate inputs
        qfunc(unitary, angles)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(unitary, angles)

        assert len(jaxpr.eqns) == 2

        rx_eqn = jaxpr.eqns[0]
        assert rx_eqn.primitive == qml.RX._primitive
        gqps_eqn = jaxpr.eqns[1]
        assert gqps_eqn.primitive == qml.GQSP._primitive
        assert gqps_eqn.invars[0] == rx_eqn.outvars[0]
        assert gqps_eqn.invars[1] == jaxpr.jaxpr.invars[1]
        assert gqps_eqn.invars[2].val == 0  # Control wire
        assert gqps_eqn.params["n_wires"] == 1
        assert len(gqps_eqn.outvars) == 1
        assert isinstance(gqps_eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, unitary.data, angles)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.GQSP(unitary, angles, control=0))

    def test_reflection(self):
        """Test the primitive bind call of Reflection."""

        op = qml.RX(np.pi / 4, 0) @ qml.Hadamard(1)
        reflection_wires = [0]
        alpha = np.pi / 2

        def qfunc(op, alpha):
            qml.Reflection(op, alpha, reflection_wires=reflection_wires)

        # Validate inputs
        qfunc(op, alpha)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(op, alpha)

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.ops.op_math.Prod._primitive

        eqn = jaxpr.eqns[3]
        assert eqn.primitive == qml.Reflection._primitive
        # Input operator and reflection/estimation wires are invars to template
        assert eqn.invars[:1] == jaxpr.eqns[2].outvars
        assert eqn.invars[1] == jaxpr.jaxpr.invars[1]
        assert [invar.val for invar in eqn.invars[2:]] == reflection_wires
        assert eqn.params == {"n_wires": 1}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, np.pi / 4, np.pi / 2)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Reflection(op, alpha, reflection_wires=reflection_wires))

    def test_quantum_phase_estimation(self):
        """Test the primitive bind call of QuantumPhaseEstimation."""

        kwargs = {"estimation_wires": [2, 3]}
        op = qml.RX(np.pi / 2, 0) @ qml.Hadamard(1)

        def qfunc(op):
            qml.QuantumPhaseEstimation(op, **kwargs)

        # Validate inputs
        qfunc(op)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(op)

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.ops.op_math.Prod._primitive

        eqn = jaxpr.eqns[3]
        assert eqn.primitive == qml.QuantumPhaseEstimation._primitive
        assert eqn.invars == jaxpr.eqns[2].outvars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, np.pi / 2)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.QuantumPhaseEstimation(op, **kwargs))

    def test_select(self):
        """Test the primitive bind call of Select."""

        ops = [qml.X(2), qml.RX(0.2, 3), qml.Y(2), qml.Z(3)]
        kwargs = {"control": [0, 1]}

        def qfunc(ops):
            qml.Select(ops, **kwargs)

        # Validate inputs
        qfunc(ops)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(ops)

        assert len(jaxpr.eqns) == 5  # four operator-creating eqns and one for Select itself

        op_eqns = jaxpr.eqns[:4]
        op_types = [qml.X, qml.RX, qml.Y, qml.Z]
        assert all(
            eqn.primitive == op_type._primitive
            for eqn, op_type in zip(op_eqns, op_types, strict=True)
        )
        eqn = jaxpr.eqns[4]
        assert eqn.primitive == qml.Select._primitive
        assert eqn.invars[:-2] == [eqn.outvars[0] for eqn in op_eqns]
        assert [invar.val for invar in eqn.invars[-2:]] == [0, 1]
        assert eqn.params == {"n_wires": 2}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            # Need to pass in angle for RX as argument to jaxpr
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.2)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Select(ops, **kwargs))

    def test_superposition(self):
        """Test the primitive bind call of Superposition."""

        coeffs = [0.5, 0.5, -0.5, -0.5]
        bases = [[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]
        kwargs = {"coeffs": coeffs, "bases": bases, "wires": [0, 1, 2, 3], "work_wire": 4}

        def qfunc():
            qml.Superposition(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.Superposition._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qml.assert_equal(q.queue[0], qml.Superposition(**kwargs))


def filter_fn(member: Any) -> bool:
    """Determine whether a member of a module is a class and genuinely belongs to
    qml.templates."""
    return inspect.isclass(member) and member.__module__.startswith("pennylane.templates")


_, all_templates = zip(*inspect.getmembers(qml.templates, filter_fn))

unmodified_templates = [template for template, *_ in unmodified_templates_cases]
unsupported_templates = [
    qml.CVNeuralNetLayers,
    qml.DisplacementEmbedding,
    qml.Interferometer,
    qml.PrepSelPrep,
    qml.QutritBasisStatePreparation,
    qml.SqueezingEmbedding,
    qml.TrotterizedQfunc,  # TODO: add support in follow up PR
    qml.templates.SubroutineOp,
    qml.templates.Subroutine,
]
modified_templates = [
    t for t in all_templates if t not in unmodified_templates + unsupported_templates
]


@pytest.mark.parametrize("template", modified_templates)
def test_templates_are_modified(template):
    """Test that all templates that are not listed as unmodified in the test cases above
    actually have their _primitive_bind_call modified."""
    # Make sure the template actually is modified in its primitive binding function
    if template == qml.templates.SubroutineOp:
        return
    assert template._primitive_bind_call.__code__ != original_op_bind_code


def test_all_modified_templates_are_tested():
    """Test that all templates in `modified_templates` (automatically computed and
    validated above) also are in `tested_modified_templates` (manually created and
    expected to resemble all tested templates."""
    assert set(modified_templates) == set(tested_modified_templates)
