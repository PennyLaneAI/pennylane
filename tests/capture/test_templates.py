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

# pylint: disable=protected-access
from typing import Any

import numpy as np
import pytest

import pennylane as qml
from pennylane.capture.primitives import _get_abstract_operator

jax = pytest.importorskip("jax")
jnp = jax.numpy

pytestmark = pytest.mark.jax

AbstractOperator = _get_abstract_operator()
original_op_bind_code = qml.operation.Operator._primitive_bind_call.__code__


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


unmodified_templates_cases = [
    (qml.AmplitudeEmbedding, (jnp.array([1.0, 0.0]), 2), {}),
    (qml.AmplitudeEmbedding, (jnp.eye(4)[2], [2, 3]), {"normalize": False}),
    (qml.AmplitudeEmbedding, (jnp.array([0.3, 0.1, 0.2]),), {"pad_with": 1.2, "wires": [0, 3]}),
    (qml.AngleEmbedding, (jnp.array([1.0, 0.0]), [2, 3]), {}),
    (qml.AngleEmbedding, (jnp.array([0.4]), [0]), {"rotation": "X"}),
    (qml.AngleEmbedding, (jnp.array([0.3, 0.1, 0.2]),), {"rotation": "Z", "wires": [0, 2, 3]}),
    (qml.BasisEmbedding, (jnp.array([1, 0]), [2, 3]), {}),
    (qml.BasisEmbedding, (), {"features": jnp.array([1, 0]), "wires": [2, 3]}),
    (qml.BasisEmbedding, (6, [0, 5, 2]), {"id": "my_id"}),
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
    (qml.BasicEntanglerLayers, (jnp.ones((2, 1)), [0]), {"rotation": "X", "id": "my_id"}),
    (
        qml.BasicEntanglerLayers,
        (jnp.array([[0.3, 0.1, 0.2]]),),
        {"rotation": "Z", "wires": [0, 2, 3]},
    ),
    # Need to fix GateFabric positional args: Currently have to pass init_state as kwarg if we want to pass wires as kwarg
    # https://github.com/PennyLaneAI/pennylane/issues/5521
    (qml.GateFabric, (jnp.ones((3, 1, 2)), [2, 3, 0, 1]), {"init_state": [0, 1, 1, 0]}),
    (
        qml.GateFabric,
        (jnp.zeros((2, 3, 2)),),
        {"include_pi": False, "wires": list(range(8)), "init_state": jnp.ones(8)},
    ),
    # (qml.GateFabric, (jnp.zeros((2, 3, 2)), jnp.ones(8)), {"include_pi": False, "wires": list(range(8))}), # Can't even init
    # (qml.GateFabric, (jnp.ones((5, 2, 2)), list(range(6)), jnp.array([0, 0, 1, 1, 0, 1])), {"include_pi": True, "id": "my_id"}), # Can't trace
    # https://github.com/PennyLaneAI/pennylane/issues/5522
    # (qml.ParticleConservingU1, (jnp.ones((3, 1, 2)), [2, 3]), {}),
    (qml.ParticleConservingU1, (jnp.ones((3, 1, 2)), [2, 3]), {"init_state": [0, 1]}),
    (
        qml.ParticleConservingU1,
        (jnp.zeros((5, 3, 2)),),
        {"wires": [0, 1, 2, 3], "init_state": jnp.ones(4)},
    ),
    # https://github.com/PennyLaneAI/pennylane/issues/5522
    # (qml.ParticleConservingU2, (jnp.ones((3, 3)), [2, 3]), {}),
    (qml.ParticleConservingU2, (jnp.ones((3, 3)), [2, 3]), {"init_state": [0, 1]}),
    (
        qml.ParticleConservingU2,
        (jnp.zeros((5, 7)),),
        {"wires": [0, 1, 2, 3], "init_state": jnp.ones(4)},
    ),
    (qml.RandomLayers, (jnp.ones((3, 3)), [2, 3]), {}),
    (qml.RandomLayers, (jnp.ones((3, 3)),), {"wires": [3, 2, 1], "ratio_imprim": 0.5}),
    (qml.RandomLayers, (), {"weights": jnp.ones((3, 3)), "wires": [3, 2, 1]}),
    (qml.RandomLayers, (jnp.ones((3, 3)),), {"wires": [3, 2, 1], "rotations": (qml.RX, qml.RZ)}),
    (qml.RandomLayers, (jnp.ones((3, 3)), [0, 1]), {"rotations": (qml.RX, qml.RZ), "seed": 41}),
    (qml.SimplifiedTwoDesign, (jnp.ones(2), jnp.zeros((3, 1, 2)), [2, 3]), {}),
    (qml.SimplifiedTwoDesign, (jnp.ones(3), jnp.zeros((3, 2, 2))), {"wires": [0, 1, 2]}),
    (qml.SimplifiedTwoDesign, (jnp.ones(2),), {"weights": jnp.zeros((3, 1, 2)), "wires": [0, 2]}),
    (
        qml.SimplifiedTwoDesign,
        (),
        {"initial_layer_weights": jnp.ones(2), "weights": jnp.zeros((3, 1, 2)), "wires": [0, 2]},
    ),
    (qml.StronglyEntanglingLayers, (jnp.ones((3, 2, 3)), [2, 3]), {"ranges": [1, 1, 1]}),
    (
        qml.StronglyEntanglingLayers,
        (jnp.ones((1, 3, 3)),),
        {"wires": [3, 2, 1], "imprimitive": qml.CZ},
    ),
    (qml.StronglyEntanglingLayers, (), {"weights": jnp.ones((3, 3, 3)), "wires": [3, 2, 1]}),
    (qml.ArbitraryStatePreparation, (jnp.ones(6), [2, 3]), {}),
    (qml.ArbitraryStatePreparation, (jnp.zeros(14),), {"wires": [3, 2, 0]}),
    (qml.ArbitraryStatePreparation, (), {"weights": jnp.ones(2), "wires": [1]}),
    (qml.BasisStatePreparation, (jnp.array([0, 1]), [2, 3]), {}),
    (qml.BasisStatePreparation, (jnp.ones(3),), {"wires": [3, 2, 0]}),
    (qml.BasisStatePreparation, (), {"basis_state": jnp.ones(1), "wires": [1]}),
    (qml.CosineWindow, ([2, 3],), {}),
    (qml.CosineWindow, (), {"wires": [2, 0, 1]}),
    (qml.MottonenStatePreparation, (jnp.ones(4) / 2, [2, 3]), {}),
    (
        qml.MottonenStatePreparation,
        (jnp.ones(8) / jnp.sqrt(8),),
        {"wires": [3, 2, 0], "id": "your_id"},
    ),
    (qml.MottonenStatePreparation, (), {"state_vector": jnp.array([1.0, 0.0]), "wires": [1]}),
    # Need to fix AllSinglesDoubles positional args: Currently have to pass hf_state as kwarg
    # if we want to pass wires as kwarg, see issue #5521
    (
        qml.AllSinglesDoubles,
        (jnp.ones(3), [2, 3, 0, 1]),
        {
            "singles": [[0, 2], [1, 3]],
            "doubles": [[2, 3, 0, 1]],
            "hf_state": np.array([0, 1, 1, 0]),
        },
    ),
    (
        qml.AllSinglesDoubles,
        (jnp.zeros(3),),
        {
            "singles": [[0, 2], [1, 3]],
            "doubles": [[2, 3, 0, 1]],
            "wires": list(range(8)),
            "hf_state": np.ones(8, dtype=int),
        },
    ),
    # (qml.AllSinglesDoubles, (jnp.ones(3), [2, 3, 0, 1], np.array([0, 1, 1, 0])), {"singles": [[0, 2], [1, 3]], "doubles": [[2,3,0,1]]}), # Can't trace
    (qml.AQFT, (1, [0, 1, 2]), {}),
    (qml.AQFT, (2,), {"wires": [0, 1, 2, 3]}),
    (qml.AQFT, (), {"order": 2, "wires": [0, 2, 3, 1]}),
    (qml.QFT, ([0, 1],), {}),
    (qml.QFT, (), {"wires": [0, 1]}),
    (qml.ArbitraryUnitary, (jnp.ones(15), [2, 3]), {}),
    (qml.ArbitraryUnitary, (jnp.zeros(15),), {"wires": [3, 2]}),
    (qml.ArbitraryUnitary, (), {"weights": jnp.ones(3), "wires": [1]}),
    (qml.FABLE, (jnp.eye(4), [2, 3, 0, 1, 5]), {}),
    (qml.FABLE, (jnp.ones((4, 4)),), {"wires": [0, 3, 2, 1, 9]}),
    (
        qml.FABLE,
        (),
        {"input_matrix": jnp.array([[1, 1], [1, -1]]) / np.sqrt(2), "wires": [1, 10, 17]},
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
    assert eqn.params == kwargs


# Only add a template to the following list if you manually added a test for it to
# TestModifiedTemplates below.
tested_modified_templates = [
    qml.TrotterProduct,
    qml.AmplitudeAmplification,
    qml.ApproxTimeEvolution,
    qml.BasisRotation,
    qml.CommutingEvolution,
    qml.ControlledSequence,
    qml.FermionicDoubleExcitation,
    qml.HilbertSchmidt,
    qml.LocalHilbertSchmidt,
    qml.QDrift,
    qml.QSVT,
    qml.QuantumMonteCarlo,
    qml.QuantumPhaseEstimation,
    qml.Qubitization,
    qml.Reflection,
    qml.Select,
    qml.MERA,
    qml.MPS,
    qml.TTN,
]


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
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        def qfunc(Hi):
            template(Hi, time=2.4, **kwargs)

        # Validate inputs
        qfunc(H)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(H)

        assert len(jaxpr.eqns) == 6

        # due to flattening and unflattening H
        assert jaxpr.eqns[0].primitive == qml.X._primitive
        assert jaxpr.eqns[1].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[2].primitive == qml.Z._primitive
        assert jaxpr.eqns[3].primitive == qml.ops.SProd._primitive
        assert jaxpr.eqns[4].primitive == qml.ops.Sum._primitive

        eqn = jaxpr.eqns[5]
        assert eqn.primitive == template._primitive
        assert eqn.invars == jaxpr.eqns[4].outvars  # the sum op

        assert eqn.params == kwargs | {"time": 2.4}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, coeffs[0], coeffs[1])

        assert len(q) == 1
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
        assert eqn.params == kwargs
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
        assert eqn.params == {"check": True, "id": None}
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
        assert eqn.params == {"control": control}
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
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, weight)

        assert len(q) == 1
        assert q.queue[0] == qml.FermionicDoubleExcitation(weight, **kwargs)

    @pytest.mark.parametrize("template", [qml.HilbertSchmidt, qml.LocalHilbertSchmidt])
    def test_hilbert_schmidt(self, template):
        """Test the primitive bind call of HilbertSchmidt and LocalHilbertSchmidt."""

        v_params = np.array([0.6])

        kwargs = {
            "u_tape": qml.tape.QuantumScript([qml.Hadamard(0)]),
            "v_function": lambda params: qml.RZ(params[0], wires=1),
            "v_wires": [1],
            "id": None,
        }

        def qfunc(v_params):
            template(v_params, **kwargs)

        # Validate inputs
        qfunc(v_params)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(v_params)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == template._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, v_params)

        assert len(q) == 1
        assert qml.equal(q.queue[0], template(v_params, **kwargs))

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
            "id": None,
            "n_wires": 4,
        }
        if template is qml.MPS:
            expected_params["offset"] = None
        assert eqn.params == expected_params
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert qml.equal(q.queue[0], template(**kwargs))

    def test_qsvt(self):
        """Test the primitive bind call of QSVT."""

        A = np.array([[0.1]])
        block_encode = qml.BlockEncode(A, wires=[0, 1])
        shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]

        def qfunc(block_encode):
            qml.QSVT(block_encode, projectors=shifts)

        # Validate inputs
        qfunc(block_encode)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(block_encode)

        assert len(jaxpr.eqns) == 2

        # due to flattening and unflattening BlockEncode
        assert jaxpr.eqns[0].primitive == qml.BlockEncode._primitive

        eqn = jaxpr.eqns[1]
        assert eqn.primitive == qml.QSVT._primitive
        assert eqn.invars == jaxpr.eqns[0].outvars
        assert eqn.params == {"projectors": shifts}
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, A)

        assert len(q) == 1
        assert q.queue[0] == qml.QSVT(block_encode, shifts)

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

        kwargs = {"func": func, "target_wires": target_wires, "estimation_wires": estimation_wires}

        def qfunc(probs):
            qml.QuantumMonteCarlo(probs, **kwargs)

        # Validate inputs
        qfunc(probs)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(probs)

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.QuantumMonteCarlo._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, probs)

        assert len(q) == 1
        assert q.queue[0] == qml.QuantumMonteCarlo(probs, **kwargs)

    @pytest.mark.usefixtures("new_opmath_only")
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
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert qml.equal(q.queue[0], qml.Qubitization(**kwargs))

    @pytest.mark.parametrize(
        "template, kwargs",
        [
            (qml.QuantumPhaseEstimation, {"estimation_wires": range(2, 4)}),
            (qml.Reflection, {"alpha": np.pi / 2, "reflection_wires": [0]}),
        ],
    )
    def test_quantum_phase_estimation_and_reflection(self, template, kwargs):
        """Test the primitive bind call of QuantumPhaseEstimation and Reflection."""

        op = qml.RX(np.pi / 2, 0) @ qml.Hadamard(1)

        def qfunc(op):
            template(op, **kwargs)

        # Validate inputs
        qfunc(op)

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)(op)

        assert len(jaxpr.eqns) == 4

        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == qml.Hadamard._primitive
        assert jaxpr.eqns[2].primitive == qml.ops.op_math.Prod._primitive

        eqn = jaxpr.eqns[3]
        assert eqn.primitive == template._primitive
        assert eqn.invars == jaxpr.eqns[2].outvars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, np.pi / 2)

        assert len(q) == 1
        assert qml.equal(q.queue[0], template(op, **kwargs))

    def test_select(self):
        """Test the primitive bind call of Select."""

        ops = [qml.X(2), qml.RX(0.2, 3), qml.Y(2), qml.Z(3)]
        kwargs = {"ops": ops, "control": [0, 1]}

        def qfunc():
            qml.Select(**kwargs)

        # Validate inputs
        qfunc()

        # Actually test primitive bind
        jaxpr = jax.make_jaxpr(qfunc)()

        assert len(jaxpr.eqns) == 1

        eqn = jaxpr.eqns[0]
        assert eqn.primitive == qml.Select._primitive
        assert eqn.invars == jaxpr.jaxpr.invars
        assert eqn.params == kwargs
        assert len(eqn.outvars) == 1
        assert isinstance(eqn.outvars[0], jax.core.DropVar)

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert len(q) == 1
        assert qml.equal(q.queue[0], qml.Select(**kwargs))


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
    qml.QutritBasisStatePreparation,
    qml.SqueezingEmbedding,
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
