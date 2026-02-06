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

import pennylane as qp
from pennylane import math

jax = pytest.importorskip("jax")
jnp = jax.numpy

pytestmark = [pytest.mark.jax, pytest.mark.capture]
original_op_bind_code = qp.operation.Operator._primitive_bind_call.__code__


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
    (qp.AmplitudeEmbedding, (jnp.array([1.0, 0.0]), 2), {}),
    (qp.AmplitudeEmbedding, (jnp.eye(4)[2], [2, 3]), {"normalize": False}),
    (qp.AmplitudeEmbedding, (jnp.array([0.3, 0.1, 0.2]),), {"pad_with": 1.2, "wires": [0, 3]}),
    (qp.AngleEmbedding, (jnp.array([1.0, 0.0]), [2, 3]), {}),
    (qp.AngleEmbedding, (jnp.array([0.4]), [0]), {"rotation": "X"}),
    (qp.AngleEmbedding, (jnp.array([0.3, 0.1, 0.2]),), {"rotation": "Z", "wires": [0, 2, 3]}),
    (qp.BasisEmbedding, (jnp.array([1, 0]), [2, 3]), {}),
    pytest.param(
        qp.BasisEmbedding,
        (),
        {"features": jnp.array([1, 0]), "wires": [2, 3]},
        marks=pytest.mark.xfail(reason="arrays should never have been in the metadata [sc-104808]"),
    ),
    (qp.BasisEmbedding, (6, [0, 5, 2]), {"id": "my_id"}),
    (qp.BasisEmbedding, (jnp.array([1, 0, 1]),), {"wires": [0, 2, 3]}),
    (qp.IQPEmbedding, (jnp.array([2.3, 0.1]), [2, 0]), {}),
    (qp.IQPEmbedding, (jnp.array([0.4, 0.2, 0.1]), [2, 1, 0]), {"pattern": [[2, 0], [1, 0]]}),
    (qp.IQPEmbedding, (jnp.array([0.4, 0.1]), [0, 10]), {"n_repeats": 3, "pattern": None}),
    (qp.QAOAEmbedding, (jnp.array([1.0, 0.0]), jnp.ones((3, 3)), [2, 3]), {}),
    (qp.QAOAEmbedding, (jnp.array([0.4]), jnp.ones((2, 1)), [0]), {"local_field": "X"}),
    (
        qp.QAOAEmbedding,
        (jnp.array([0.3, 0.1, 0.2]), jnp.zeros((2, 6))),
        {"local_field": "Z", "wires": [0, 2, 3]},
    ),
    (qp.BasicEntanglerLayers, (jnp.ones((5, 2)), [2, 3]), {}),
    (qp.BasicEntanglerLayers, (jnp.ones((2, 1)), [0]), {"rotation": "X", "id": "my_id"}),
    (
        qp.BasicEntanglerLayers,
        (jnp.array([[0.3, 0.1, 0.2]]),),
        {"rotation": "Z", "wires": [0, 2, 3]},
    ),
    # Need to fix GateFabric positional args: Currently have to pass init_state as kwarg if we want to pass wires as kwarg
    # https://github.com/PennyLaneAI/pennylane/issues/5521
    (qp.GateFabric, (jnp.ones((3, 1, 2)), [2, 3, 0, 1]), {"init_state": [0, 1, 1, 0]}),
    pytest.param(
        qp.GateFabric,
        (jnp.zeros((2, 3, 2)),),
        {"include_pi": False, "wires": list(range(8)), "init_state": jnp.ones(8)},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    # (qp.GateFabric, (jnp.zeros((2, 3, 2)), jnp.ones(8)), {"include_pi": False, "wires": list(range(8))}), # Can't even init
    # (qp.GateFabric, (jnp.ones((5, 2, 2)), list(range(6)), jnp.array([0, 0, 1, 1, 0, 1])), {"include_pi": True, "id": "my_id"}), # Can't trace
    # https://github.com/PennyLaneAI/pennylane/issues/5522
    # (qp.ParticleConservingU1, (jnp.ones((3, 1, 2)), [2, 3]), {}),
    (qp.ParticleConservingU1, (jnp.ones((3, 1, 2)), [2, 3]), {"init_state": [0, 1]}),
    pytest.param(
        qp.ParticleConservingU1,
        (jnp.zeros((5, 3, 2)),),
        {"wires": [0, 1, 2, 3], "init_state": jnp.ones(4)},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    # https://github.com/PennyLaneAI/pennylane/issues/5522
    # (qp.ParticleConservingU2, (jnp.ones((3, 3)), [2, 3]), {}),
    (qp.ParticleConservingU2, (jnp.ones((3, 3)), [2, 3]), {"init_state": [0, 1]}),
    pytest.param(
        qp.ParticleConservingU2,
        (jnp.zeros((5, 7)),),
        {"wires": [0, 1, 2, 3], "init_state": jnp.ones(4)},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.RandomLayers, (jnp.ones((3, 3)), [2, 3]), {}),
    (qp.RandomLayers, (jnp.ones((3, 3)),), {"wires": [3, 2, 1], "ratio_imprim": 0.5}),
    pytest.param(
        qp.RandomLayers,
        (),
        {"weights": jnp.ones((3, 3)), "wires": [3, 2, 1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.RandomLayers, (jnp.ones((3, 3)),), {"wires": [3, 2, 1], "rotations": (qp.RX, qp.RZ)}),
    (qp.RandomLayers, (jnp.ones((3, 3)), [0, 1]), {"rotations": (qp.RX, qp.RZ), "seed": 41}),
    (qp.SimplifiedTwoDesign, (jnp.ones(2), jnp.zeros((3, 1, 2)), [2, 3]), {}),
    (qp.SimplifiedTwoDesign, (jnp.ones(3), jnp.zeros((3, 2, 2))), {"wires": [0, 1, 2]}),
    pytest.param(
        qp.SimplifiedTwoDesign,
        (jnp.ones(2),),
        {"weights": jnp.zeros((3, 1, 2)), "wires": [0, 2]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    pytest.param(
        qp.SimplifiedTwoDesign,
        (),
        {"initial_layer_weights": jnp.ones(2), "weights": jnp.zeros((3, 1, 2)), "wires": [0, 2]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.StronglyEntanglingLayers, (jnp.ones((3, 2, 3)), [2, 3]), {"ranges": [1, 1, 1]}),
    (
        qp.StronglyEntanglingLayers,
        (jnp.ones((1, 3, 3)),),
        {"wires": [3, 2, 1], "imprimitive": qp.CZ},
    ),
    pytest.param(
        qp.StronglyEntanglingLayers,
        (),
        {"weights": jnp.ones((3, 3, 3)), "wires": [3, 2, 1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.ArbitraryStatePreparation, (jnp.ones(6), [2, 3]), {}),
    (qp.ArbitraryStatePreparation, (jnp.zeros(14),), {"wires": [3, 2, 0]}),
    pytest.param(
        qp.ArbitraryStatePreparation,
        (),
        {"weights": jnp.ones(2), "wires": [1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.CosineWindow, ([2, 3],), {}),
    (qp.CosineWindow, (), {"wires": [2, 0, 1]}),
    (qp.MottonenStatePreparation, (jnp.ones(4) / 2, [2, 3]), {}),
    (
        qp.MottonenStatePreparation,
        (jnp.ones(8) / jnp.sqrt(8),),
        {"wires": [3, 2, 0], "id": "your_id"},
    ),
    pytest.param(
        qp.MottonenStatePreparation,
        (),
        {"state_vector": jnp.array([1.0, 0.0]), "wires": [1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.AQFT, (1, [0, 1, 2]), {}),
    (qp.AQFT, (2,), {"wires": [0, 1, 2, 3]}),
    (qp.AQFT, (), {"order": 2, "wires": [0, 2, 3, 1]}),
    (qp.QFT, ([0, 1],), {}),
    (qp.QFT, (), {"wires": [0, 1]}),
    (qp.ArbitraryUnitary, (jnp.ones(15), [2, 3]), {}),
    (qp.ArbitraryUnitary, (jnp.zeros(15),), {"wires": [3, 2]}),
    pytest.param(
        qp.ArbitraryUnitary,
        (),
        {"weights": jnp.ones(3), "wires": [1]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.FABLE, (jnp.eye(4), [2, 3, 0, 1, 5]), {}),
    (qp.FABLE, (jnp.ones((4, 4)),), {"wires": [0, 3, 2, 1, 9]}),
    pytest.param(
        qp.FABLE,
        (),
        {"input_matrix": jnp.array([[1, 1], [1, -1]]) / np.sqrt(2), "wires": [1, 10, 17]},
        marks=pytest.mark.xfail(
            reason="arrays should never have been in the metadata, [sc-104808]"
        ),
    ),
    (qp.FermionicSingleExcitation, (0.421,), {"wires": [0, 3, 2]}),
    (qp.FlipSign, (7,), {"wires": [0, 3, 2]}),
    (qp.FlipSign, (np.array([1, 0, 0]), [0, 1, 2]), {}),
    (
        qp.kUpCCGSD,
        (jnp.ones((1, 6)), [0, 1, 2, 3]),
        {"k": 1, "delta_sz": 0, "init_state": [1, 1, 0, 0]},
    ),
    (qp.Permute, (np.array([1, 2, 0]), [0, 1, 2]), {}),
    (qp.Permute, (np.array([1, 2, 0]),), {"wires": [0, 1, 2]}),
    (
        qp.TwoLocalSwapNetwork,
        ([0, 1, 2, 3, 4],),
        {"acquaintances": lambda index, wires, param=None: qp.CNOT(index)},
    ),
    (qp.GroverOperator, (), {"wires": [0, 1]}),
    (qp.GroverOperator, ([0, 1],), {}),
    (
        qp.UCCSD,
        (jnp.ones(3), [2, 3, 0, 1]),
        {"s_wires": [[0], [1]], "d_wires": [[[2], [3]]], "init_state": [0, 1, 1, 0]},
    ),
    (qp.TemporaryAND, (), ({"wires": [0, 1, 2], "control_values": [0, 1]})),
    (qp.TemporaryAND, ([0, 1, 2],), ({"control_values": [0, 1]})),
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
    qp.TrotterProduct,
    qp.AllSinglesDoubles,
    qp.AmplitudeAmplification,
    qp.ApproxTimeEvolution,
    qp.BasisRotation,
    qp.BBQRAM,
    qp.CommutingEvolution,
    qp.ControlledSequence,
    qp.FermionicDoubleExcitation,
    qp.HilbertSchmidt,
    qp.HybridQRAM,
    qp.IQP,
    qp.LocalHilbertSchmidt,
    qp.QDrift,
    qp.QSVT,
    qp.QuantumMonteCarlo,
    qp.QuantumPhaseEstimation,
    qp.Qubitization,
    qp.Reflection,
    qp.Select,
    qp.SelectOnlyQRAM,
    qp.MERA,
    qp.MPS,
    qp.TTN,
    qp.QROM,
    qp.PhaseAdder,
    qp.Adder,
    qp.SemiAdder,
    qp.Multiplier,
    qp.OutMultiplier,
    qp.OutAdder,
    qp.ModExp,
    qp.OutPoly,
    qp.Superposition,
    qp.MPSPrep,
    qp.GQSP,
    qp.QROMStatePreparation,
    qp.MultiplexerStatePreparation,
    qp.SelectPauliRot,
]


# pylint: disable=too-many-public-methods
class TestModifiedTemplates:
    """Test that templates with custom primitive binds are captured as expected."""

    @pytest.mark.parametrize(
        "template, kwargs",
        [
            (qp.TrotterProduct, {"order": 2}),
            (qp.ApproxTimeEvolution, {"n": 2}),
            (qp.CommutingEvolution, {"frequencies": (1.2, 2)}),
            (qp.QDrift, {"n": 2, "seed": 10}),
        ],
    )
    def test_evolution_ops(self, template, kwargs):
        """Test the primitive bind call of Hamiltonian time evolution templates."""

        coeffs = [0.25, 0.75]

        def qfunc(coeffs):
            ops = [qp.X(0), qp.Z(0)]
            H = qp.dot(coeffs, ops)
            template(H, 2.4, **kwargs)

        # Validate inputs
        qfunc(coeffs)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(coeffs)

        assert len(jaxpr.eqns) == 6

        # due to flattening and unflattening H
        assert jaxpr.eqns[0].primitive == qp.X._primitive
        assert jaxpr.eqns[1].primitive == qp.Z._primitive
        assert jaxpr.eqns[2].primitive == qp.ops.SProd._primitive
        assert jaxpr.eqns[3].primitive == qp.ops.SProd._primitive
        assert jaxpr.eqns[4].primitive == qp.ops.Sum._primitive

        eqn = jaxpr.eqns[5]
        assert eqn.primitive == template._primitive
        assert eqn.invars[0] == jaxpr.eqns[4].outvars[0]  # the sum op
        assert eqn.invars[1].val == 2.4

        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, coeffs[0], coeffs[1])

        assert len(q) == 1
        ops = [qp.X(0), qp.Z(0)]
        H = qp.dot(coeffs, ops)
        assert q.queue[0] == template(H, time=2.4, **kwargs)

    def test_amplitude_amplification(self):
        """Test the primitive bind call of AmplitudeAmplification."""

        U = qp.Hadamard(0)
        O = qp.FlipSign(1, 0)
        iters = 3

        kwargs = {"iters": iters, "fixed_point": False, "p_min": 0.4}

        def qfunc(U, O):
            qp.AmplitudeAmplification(U, O, **kwargs)

        # Validate inputs
        qfunc(U, O)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(U, O)

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[1].primitive == qp.FlipSign._primitive

        eqn = jaxpr.eqns[2]
        assert eqn.primitive == qp.AmplitudeAmplification._primitive
        assert eqn.invars[0] == jaxpr.eqns[0].outvars[0]  # Hadamard
        assert eqn.invars[1] == jaxpr.eqns[1].outvars[0]  # FlipSign
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert q.queue[0] == qp.AmplitudeAmplification(U, O, **kwargs)

    def test_basis_rotation(self):
        """Test the primitive bind call of BasisRotation."""

        mat = np.eye(4)
        wires = [0, 5]

        def qfunc(wires, mat):
            qp.BasisRotation(wires, mat, check=True)

        # Validate inputs
        qfunc(wires, mat)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(wires, mat)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.BasisRotation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == {"check": True, "id": None}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *wires, mat)

        assert len(q) == 1
        assert q.queue[0] == qp.BasisRotation(wires=wires, unitary_matrix=mat, check=True)

    def test_controlled_sequence(self):
        """Test the primitive bind call of ControlledSequence."""

        assert (
            qp.ControlledSequence._primitive_bind_call.__code__
            == qp.ops.op_math.SymbolicOp._primitive_bind_call.__code__
        )

        base = qp.RX(0.5, 0)
        control = [1, 5]

        def fn(base):
            qp.ControlledSequence(base, control=control)

        # Validate inputs
        fn(base)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(fn)(base)

        assert len(jaxpr.eqns) == 2
        assert jaxpr.eqns[0].primitive == qp.RX._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qp.ControlledSequence._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars
        # JAX 0.7.0 converts lists to tuples for hashability
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(
            {"control": control}
        )
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

        assert len(q) == 1  # One for each control
        assert q.queue[0] == qp.ControlledSequence(base, control)

    def test_fermionic_double_excitation(self):
        """Test the primitive bind call of FermionicDoubleExcitation."""

        weight = 0.251

        kwargs = {"wires1": [0, 6], "wires2": [2, 3]}

        def qfunc(weight):
            qp.FermionicDoubleExcitation(weight, **kwargs)

        # Validate inputs
        qfunc(weight)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(weight)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.FermionicDoubleExcitation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, weight)

        assert len(q) == 1
        assert q.queue[0] == qp.FermionicDoubleExcitation(weight, **kwargs)

    @pytest.mark.parametrize("template", [qp.HilbertSchmidt, qp.LocalHilbertSchmidt])
    def test_hilbert_schmidt(self, template):
        """Test the primitive bind call of HilbertSchmidt and LocalHilbertSchmidt."""

        def qfunc(v_params):
            U = qp.Hadamard(0)
            V = qp.RZ(v_params[0], wires=1)
            template(V, U)

        v_params = jnp.array([0.1])
        # Validate inputs
        qfunc(v_params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(v_params)

        assert len(jaxpr.eqns) == 5
        assert jaxpr.eqns[0].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[-2].primitive == qp.RZ._primitive

        eqn = jaxpr.eqns[-1]
        assert eqn.primitive == template._primitive
        assert eqn.params == {"num_v_ops": 1}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, v_params)

        assert len(q) == 1

        U = qp.Hadamard(0)
        V = qp.RZ(v_params[0], wires=1)
        assert qp.equal(q.queue[0], template(V, U)) is True

    @pytest.mark.parametrize("template", [qp.HilbertSchmidt, qp.LocalHilbertSchmidt])
    def test_hilbert_schmidt_multiple_ops(self, template):
        """Test the primitive bind call of HilbertSchmidt and LocalHilbertSchmidt with multiple ops."""

        def qfunc(v_params):
            U = [qp.Hadamard(0), qp.Hadamard(1)]
            V = [qp.RZ(v_params[0], wires=2), qp.RX(v_params[1], wires=3)]
            template(V, U)

        v_params = jnp.array([0.1, 0.2])
        # Validate inputs
        qfunc(v_params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(v_params)

        assert len(jaxpr.eqns) == 9
        assert jaxpr.eqns[0].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[-5].primitive == qp.RZ._primitive
        assert jaxpr.eqns[-2].primitive == qp.RX._primitive

        eqn = jaxpr.eqns[-1]
        assert eqn.primitive == template._primitive
        assert eqn.params == {"num_v_ops": 2}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, v_params)

        assert len(q) == 1

        U = [qp.Hadamard(0), qp.Hadamard(1)]
        V = [qp.RZ(v_params[0], wires=2), qp.RX(v_params[1], wires=3)]
        assert qp.equal(q.queue[0], template(V, U)) is True

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
            qp.IQP(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.IQP._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.IQP(**kwargs))

    @pytest.mark.parametrize("template", [qp.MERA, qp.MPS, qp.TTN])
    def test_tensor_networks(self, template):
        """Test the primitive bind call of MERA, MPS, and TTN."""

        def block(weights, wires):
            return [
                qp.CNOT(wires),
                qp.RY(weights[0], wires[0]),
                qp.RY(weights[1], wires[1]),
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
            "id": None,
            "n_wires": 4,
        }
        if template is qp.MPS:
            expected_params["offset"] = None
        # JAX 0.7.0 converts lists to tuples for hashability
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(expected_params)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], template(**kwargs))

    def test_qsvt(self):
        """Test the primitive bind call of QSVT."""

        def qfunc(A):
            block_encode = qp.BlockEncode(A, wires=[0, 1])
            shifts = [qp.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
            qp.QSVT(block_encode, projectors=shifts)

        A = np.array([[0.1]])
        # Validate inputs
        qfunc(A)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(A)

        assert len(jaxpr.eqns) == 5

        assert jaxpr.eqns[0].primitive == qp.BlockEncode._primitive

        eqn = jaxpr.eqns[-1]
        assert eqn.primitive == qp.QSVT._primitive
        for i in range(4):
            assert eqn.invars[i] == jaxpr.eqns[i].outvars[0]
        assert eqn.params == {}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, A)

        assert len(q) == 1
        block_encode = qp.BlockEncode(A, wires=[0, 1])
        shifts = [qp.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
        assert q.queue[0] == qp.QSVT(block_encode, shifts)

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
            qp.MPSPrep(mps=mps, wires=wires)

        # Validate inputs
        qfunc(mps)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(mps)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.MPSPrep._primitive
        assert eqn.invars[:4] == jaxpr.jaxpr.invars
        assert [invar.val for invar in eqn.invars[4:]] == [0, 1, 2]
        assert eqn.params == {
            "id": None,
            "n_wires": 3,
            "work_wires": None,
            "right_canonicalize": False,
        }
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *mps)

        assert len(q) == 1
        assert q.queue[0] == qp.MPSPrep(mps=mps, wires=wires)

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
            qp.AllSinglesDoubles(weights, wires, hf_state, singles=singles, doubles=doubles)

        # Validate inputs
        qfunc(*params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(*params)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *params)

        assert len(q) == 1
        assert q.queue[0] == qp.AllSinglesDoubles(*params)

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

        kwargs = {"func": func, "id": None, "num_target_wires": 6}

        def qfunc(probs, target_wires, estimation_wires):
            qp.QuantumMonteCarlo(probs, func, target_wires, estimation_wires)

        # Validate inputs
        qfunc(probs, target_wires, estimation_wires)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(probs, list(target_wires), list(estimation_wires))

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.QuantumMonteCarlo._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, probs, *target_wires, *estimation_wires)

        assert len(q) == 1
        assert q.queue[0] == qp.QuantumMonteCarlo(probs, func, target_wires, estimation_wires)

    def test_qubitization(self):
        """Test the primitive bind call of Qubitization."""

        hamiltonian = qp.dot([0.5, 1.2, -0.84], [qp.X(2), qp.Hadamard(3), qp.Z(2) @ qp.Y(3)])
        kwargs = {"hamiltonian": hamiltonian, "control": [0, 1]}

        def qfunc():
            qp.Qubitization(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.Qubitization._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Qubitization(**kwargs))

    def test_bbqram(self):
        """Test the primitve bind call of BBQRAM."""

        kwargs = {
            "bitstrings": ("010", "111", "110", "000"),
            "control_wires": (0, 1),
            "target_wires": (2, 3, 4),
            "work_wires": tuple([5] + [6, 7, 8] + [12, 13, 14] + [9, 10, 11]),
        }

        def qfunc():
            qp.BBQRAM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.BBQRAM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.BBQRAM(**kwargs))

    def test_select_only_qram(self):
        """Test the primitve bind call of SelectOnlyQRAM."""

        kwargs = {
            "bitstrings": ("010", "111", "110", "000", "010", "111", "110", "000"),
            "control_wires": (0, 1),
            "target_wires": (2, 3, 4),
            "select_wires": (12),
            "select_value": 1,
        }

        def qfunc():
            qp.SelectOnlyQRAM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.SelectOnlyQRAM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.SelectOnlyQRAM(**kwargs))

    def test_hybrid_qram(self):
        """Test the primitve bind call of HybridQRAM."""

        kwargs = {
            "bitstrings": ("010", "111", "110", "000"),
            "control_wires": (0, 1),
            "target_wires": (2, 3, 4),
            "work_wires": tuple([5, 6, 7, 8, 12, 13, 14, 15, 9, 10, 11]),
            "k": 0,
        }

        def qfunc():
            qp.HybridQRAM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.HybridQRAM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.HybridQRAM(**kwargs))

    def test_qrom(self):
        """Test the primitive bind call of QROM."""

        kwargs = {
            "bitstrings": ["0", "1"],
            "control_wires": [0],
            "target_wires": [1],
            "work_wires": None,
        }

        def qfunc():
            qp.QROM(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.QROM._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.QROM(**kwargs))

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
            qp.QROMStatePreparation(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.QROMStatePreparation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.QROMStatePreparation(**kwargs))

    def test_multiplexer_state_prep(self):
        """Test the primitive bind call of MultiplexerStatePreparation."""

        state_vector = np.array([1 / 2, -1 / 2, 1 / 2, 1j / 2])
        kwargs = {
            "wires": (8, 9),
        }

        def qfunc(state_vector):
            qp.MultiplexerStatePreparation(state_vector, **kwargs)

        qfunc(state_vector)
        jaxpr = jax.make_jaxpr(qfunc)(state_vector)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.MultiplexerStatePreparation._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, state_vector)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.MultiplexerStatePreparation(state_vector, **kwargs))

    def test_multiplexed_rotation(self):
        """Test the primitive bind call of SelectPauliRot."""

        angles = np.arange(1, 9)
        kwargs = {
            "control_wires": [0, 1, 2],
            "target_wire": 3,
            "rot_axis": "X",
        }

        def qfunc(angles):
            qp.SelectPauliRot(angles, **kwargs)

        # Validate inputs
        qfunc(angles)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(angles)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.SelectPauliRot._primitive
        assert eqn.invars[:1] == jaxpr.jaxpr.invars
        assert [invar.val for invar in eqn.invars[1:]] == [0, 1, 2, 3]
        assert eqn.params == {"n_wires": 4, "rot_axis": "X"}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, angles)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.SelectPauliRot(angles, **kwargs))

    def test_phase_adder(self):
        """Test the primitive bind call of PhaseAdder."""

        kwargs = {
            "k": 3,
            "x_wires": [0, 1],
            "mod": None,
            "work_wire": None,
        }

        def qfunc():
            qp.PhaseAdder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.PhaseAdder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.PhaseAdder(**kwargs))

    def test_adder(self):
        """Test the primitive bind call of Adder."""

        kwargs = {
            "k": 3,
            "x_wires": [0, 1],
            "mod": None,
            "work_wires": None,
        }

        def qfunc():
            qp.Adder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.Adder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Adder(**kwargs))

    def test_semiadder(self):
        """Test the primitive bind call of SemiAdder."""

        kwargs = {
            "x_wires": [0, 1, 2],
            "y_wires": [3, 4, 5],
            "work_wires": [6, 7],
        }

        def qfunc():
            qp.SemiAdder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.SemiAdder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.SemiAdder(**kwargs))

    def test_multiplier(self):
        """Test the primitive bind call of Multiplier."""

        kwargs = {
            "k": 3,
            "x_wires": [0, 1],
            "mod": None,
            "work_wires": [2, 3],
        }

        def qfunc():
            qp.Multiplier(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.Multiplier._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Multiplier(**kwargs))

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
            qp.OutMultiplier(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.OutMultiplier._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.OutMultiplier(**kwargs))

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
            qp.OutAdder(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.OutAdder._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.OutAdder(**kwargs))

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
            qp.ModExp(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.ModExp._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.ModExp(**kwargs))

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
            qp.OutPoly(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.OutPoly._primitive
        assert eqn.invars == jaxpr.jaxpr.invars

        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)

        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.OutPoly(**kwargs))

    def test_gqsp(self):
        """Test the primitive bind call of GQSP."""

        def qfunc(unitary, angles):
            qp.GQSP(unitary, angles, control=0)

        angles = np.ones([3, 3])
        unitary = qp.RX(1, wires=1)
        # Validate inputs
        qfunc(unitary, angles)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(unitary, angles)

        assert len(jaxpr.eqns) == 2

        rx_eqn = jaxpr.eqns[0]
        assert rx_eqn.primitive == qp.RX._primitive
        gqps_eqn = jaxpr.eqns[1]
        assert gqps_eqn.primitive == qp.GQSP._primitive
        assert gqps_eqn.invars[0] == rx_eqn.outvars[0]
        assert gqps_eqn.invars[1] == jaxpr.jaxpr.invars[1]
        assert gqps_eqn.invars[2].val == 0  # Control wire
        assert gqps_eqn.params == {"n_wires": 1, "id": None}
        assert len(gqps_eqn.outvars) == 1
        assert isinstance(gqps_eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, unitary.data, angles)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.GQSP(unitary, angles, control=0))

    def test_reflection(self):
        """Test the primitive bind call of Reflection."""

        op = qp.RX(np.pi / 4, 0) @ qp.Hadamard(1)
        reflection_wires = [0]
        alpha = np.pi / 2

        def qfunc(op, alpha):
            qp.Reflection(op, alpha, reflection_wires=reflection_wires)

        # Validate inputs
        qfunc(op, alpha)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(op, alpha)

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qp.ops.op_math.Prod._primitive

        eqn = jaxpr.eqns[3]
        assert eqn.primitive == qp.Reflection._primitive
        # Input operator and reflection/estimation wires are invars to template
        assert eqn.invars[:1] == jaxpr.eqns[2].outvars
        assert eqn.invars[1] == jaxpr.jaxpr.invars[1]
        assert [invar.val for invar in eqn.invars[2:]] == reflection_wires
        assert eqn.params == {"n_wires": 1}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, np.pi / 4, np.pi / 2)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Reflection(op, alpha, reflection_wires=reflection_wires))

    def test_quantum_phase_estimation(self):
        """Test the primitive bind call of QuantumPhaseEstimation."""

        kwargs = {"estimation_wires": [2, 3]}
        op = qp.RX(np.pi / 2, 0) @ qp.Hadamard(1)

        def qfunc(op):
            qp.QuantumPhaseEstimation(op, **kwargs)

        # Validate inputs
        qfunc(op)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(op)

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == qp.RX._primitive
        assert jaxpr.eqns[1].primitive == qp.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qp.ops.op_math.Prod._primitive

        eqn = jaxpr.eqns[3]
        assert eqn.primitive == qp.QuantumPhaseEstimation._primitive
        assert eqn.invars == jaxpr.eqns[2].outvars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, np.pi / 2)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.QuantumPhaseEstimation(op, **kwargs))

    def test_select(self):
        """Test the primitive bind call of Select."""

        ops = [qp.X(2), qp.RX(0.2, 3), qp.Y(2), qp.Z(3)]
        kwargs = {"control": [0, 1]}

        def qfunc(ops):
            qp.Select(ops, **kwargs)

        # Validate inputs
        qfunc(ops)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(ops)

        assert len(jaxpr.eqns) == 5  # four operator-creating eqns and one for Select itself

        op_eqns = jaxpr.eqns[:4]
        op_types = [qp.X, qp.RX, qp.Y, qp.Z]
        assert all(
            eqn.primitive == op_type._primitive
            for eqn, op_type in zip(op_eqns, op_types, strict=True)
        )
        eqn = jaxpr.eqns[4]
        assert eqn.primitive == qp.Select._primitive
        assert eqn.invars[:-2] == [eqn.outvars[0] for eqn in op_eqns]
        assert [invar.val for invar in eqn.invars[-2:]] == [0, 1]
        assert eqn.params == {"n_wires": 2}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            # Need to pass in angle for RX as argument to jaxpr
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.2)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Select(ops, **kwargs))

    def test_superposition(self):
        """Test the primitive bind call of Superposition."""

        coeffs = [0.5, 0.5, -0.5, -0.5]
        bases = [[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]
        kwargs = {"coeffs": coeffs, "bases": bases, "wires": [0, 1, 2, 3], "work_wire": 4}

        def qfunc():
            qp.Superposition(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qp.Superposition._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert normalize_for_comparison(eqn.params) == normalize_for_comparison(kwargs)
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qp.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        qp.assert_equal(q.queue[0], qp.Superposition(**kwargs))


def filter_fn(member: Any) -> bool:
    """Determine whether a member of a module is a class and genuinely belongs to
    qp.templates."""
    return inspect.isclass(member) and member.__module__.startswith("pennylane.templates")


_, all_templates = zip(*inspect.getmembers(qp.templates, filter_fn))

unmodified_templates = [template for template, *_ in unmodified_templates_cases]
unsupported_templates = [
    qp.CVNeuralNetLayers,
    qp.DisplacementEmbedding,
    qp.Interferometer,
    qp.PrepSelPrep,
    qp.QutritBasisStatePreparation,
    qp.SqueezingEmbedding,
    qp.TrotterizedQfunc,  # TODO: add support in follow up PR
]
modified_templates = [
    t for t in all_templates if t not in unmodified_templates + unsupported_templates
]


@pytest.mark.parametrize("template", modified_templates)
def test_templates_are_modified(template):
    """Test that all templates that are not listed as unmodified in the test cases above
    actually have their _primitive_bind_call modified."""
    # Make sure the template actually is modified in its primitive binding function
    assert template._primitive_bind_call.__code__ != original_op_bind_code


def test_all_modified_templates_are_tested():
    """Test that all templates in `modified_templates` (automatically computed and
    validated above) also are in `tested_modified_templates` (manually created and
    expected to resemble all tested templates."""
    assert set(modified_templates) == set(tested_modified_templates)
