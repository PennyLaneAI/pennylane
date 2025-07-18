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
Tests for the add_noise transform.
"""
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.noise.add_noise import add_noise
from pennylane.tape import QuantumScript

# pylint:disable = no-member


class TestAddNoise:
    """Tests for the add_noise transform using input tapes"""

    with qml.queuing.AnnotatedQueue() as q_tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    tape = QuantumScript.from_queue(q_tape)

    with qml.queuing.AnnotatedQueue() as q_tape_with_prep:
        qml.StatePrep([1, 0], wires=0)
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    tape_with_prep = QuantumScript.from_queue(q_tape_with_prep)

    # conditionals
    c0 = qml.noise.op_eq(qml.RX)
    c1 = qml.noise.op_in([qml.RY, qml.RZ])
    c2 = qml.noise.op_eq("StatePrep")

    # callables
    @staticmethod
    def n0(op, **kwargs):  # pylint: disable=unused-argument
        """Mapped callable for c0"""
        qml.RZ(op.parameters[0] * 0.05, op.wires)
        qml.apply(op)
        qml.RZ(-op.parameters[0] * 0.05, op.wires)

    n1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

    @staticmethod
    def n2(op, **kwargs):
        """Mapped callable for c2"""
        qml.ThermalRelaxationError(0.4, kwargs["t1"], 0.2, 0.6, op.wires)

    noise_model = qml.NoiseModel({c0: n0.__func__, c1: n1})
    noise_model_with_prep = noise_model + qml.NoiseModel({c2: n2.__func__}, t1=0.4)

    def test_noise_model_error(self):
        """Tests if a ValueError is raised when noise model is not given"""
        with pytest.raises(
            ValueError,
            match="Provided noise model object must define model_map and metatadata attributes",
        ):
            add_noise(self.tape, {})

    def test_noise_tape(self):
        """Test if the expected tape is returned with the transform"""
        [tape], _ = add_noise(self.tape, self.noise_model)

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RZ(0.045, wires=0)
            qml.RX(0.9, wires=0)
            qml.RZ(-0.045, wires=0)
            qml.RY(0.4, wires=1)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.RZ(0.03, wires=1)
            qml.RX(0.6, wires=1)
            qml.RZ(-0.03, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        tape_exp = QuantumScript.from_queue(q_tape_exp)

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"

        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)

    def test_noise_tape_with_state_prep(self):
        """Test if the expected tape is returned with the transform"""
        [tape], _ = add_noise(self.tape_with_prep, self.noise_model_with_prep)

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.StatePrep([1, 0], wires=0)
            qml.ThermalRelaxationError(0.4, 0.4, 0.2, 0.6, wires=0)
            qml.RZ(0.045, wires=0)
            qml.RX(0.9, wires=0)
            qml.RZ(-0.045, wires=0)
            qml.RY(0.4, wires=1)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.RZ(0.03, wires=1)
            qml.RX(0.6, wires=1)
            qml.RZ(-0.03, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        tape_exp = QuantumScript.from_queue(q_tape_exp)

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == "Prod"

        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)


class TestAddNoiseInterface:
    """Tests for the add_noise transform using input qnode and devices"""

    def test_add_noise_qnode(self):
        """Test that a QNode with add_noise decorator gives a different result."""
        dev = qml.device("default.mixed", wires=2)

        c, n = qml.noise.op_in([qml.RY, qml.RZ]), qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

        @partial(add_noise, noise_model=qml.NoiseModel({c: n}))
        @qml.qnode(dev)
        def f_noisy(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev)
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev)
        def g(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.AmplitudeDamping(0.4, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        args = [0.1, 0.2, 0.3, 0.4]

        assert not np.isclose(f_noisy(*args), f(*args))
        assert np.isclose(f_noisy(*args), g(*args))

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_add_noise_dev(self, dev_name):
        """Test if an device transformed by the add_noise transform does successfully add noise to
        subsequent circuit executions"""
        with qml.queuing.AnnotatedQueue() as q_in_tape:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.expval(qml.PauliZ(0))

        in_tape = QuantumScript.from_queue(q_in_tape)
        dev = qml.device(dev_name, wires=2)

        program = dev.preprocess_transforms()
        res_without_noise = qml.execute(
            [in_tape], dev, qml.gradients.param_shift, transform_program=program
        )

        c, n = qml.noise.op_in([qml.RX, qml.RY]), qml.noise.partial_wires(qml.PhaseShift, 0.4)
        new_dev = add_noise(dev, noise_model=qml.NoiseModel({c: n}))
        new_program = new_dev.preprocess_transforms()
        [tape], _ = new_program([in_tape])
        res_with_noise = qml.execute(
            [in_tape], new_dev, qml.gradients.param_shift, transform_program=new_program
        )

        with qml.queuing.AnnotatedQueue() as q_tape_exp:
            qml.RX(0.9, wires=0)
            qml.PhaseShift(0.4, wires=0)
            qml.RY(0.4, wires=1)
            qml.PhaseShift(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.PhaseShift(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.PhaseShift(0.4, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.expval(qml.PauliZ(0))

        tape_exp = QuantumScript.from_queue(q_tape_exp)
        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(
            np.allclose(o1.parameters, o2.parameters)
            for o1, o2 in zip(tape.operations, tape_exp.operations)
        )
        assert len(tape.measurements) == 2
        assert tape.observables[0].name == "Prod"

        assert tape.observables[0].wires.tolist() == [0, 1]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)
        assert tape.observables[1].name == "PauliZ"
        assert tape.observables[1].wires.tolist() == [0]
        assert isinstance(tape.measurements[1], qml.measurements.ExpectationMP)

        assert not np.allclose(res_without_noise, res_with_noise)

    def test_add_noise_template(self):
        """Test that noisy ops are inserted correctly into a decomposed template"""
        dev = qml.device("default.mixed", wires=2)

        c, n = qml.noise.op_in([qml.RX, qml.RY]), qml.noise.partial_wires(qml.PhaseDamping, 0.3)

        @partial(add_noise, noise_model=qml.NoiseModel({c: n}))
        @qml.qnode(dev)
        def f1(w1, w2):
            qml.SimplifiedTwoDesign(w1, w2, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def f2(w1, w2):
            qml.RY(w1[0], wires=0)
            qml.PhaseDamping(0.3, wires=0)
            qml.RY(w1[1], wires=1)
            qml.PhaseDamping(0.3, wires=1)
            qml.CZ(wires=[0, 1])
            qml.RY(w2[0][0][0], wires=0)
            qml.PhaseDamping(0.3, wires=0)
            qml.RY(w2[0][0][1], wires=1)
            qml.PhaseDamping(0.3, wires=1)
            return qml.expval(qml.PauliZ(0))

        w1 = np.random.random(2)
        w2 = np.random.random((1, 1, 2))

        assert np.allclose(f1(w1, w2), f2(w1, w2))

    # pylint: disable=unused-argument
    def test_add_noise_with_non_qwc_obs_and_mid_meas(self):
        """Test that the add_noise transform catches and reports errors from the enclosed function."""

        dev = qml.device("default.qubit", wires=5)

        fcond = qml.noise.wires_in([0, 1])

        def noise(op, **kwargs):
            qml.CNOT(wires=[1, 0])
            qml.CRX(kwargs["noise_param"], wires=[0, 1])

        @qml.qnode(dev)
        @partial(add_noise, noise_model=qml.NoiseModel({fcond: noise}, noise_param=0.3))
        def noisy_circuit(circuit_param):
            qml.RY(circuit_param, wires=0)
            qml.Hadamard(wires=0)
            qml.T(wires=0)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(~m0 & m1 == 0, qml.X)(wires=2)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def explicit_circuit(circuit_param):
            qml.RY(circuit_param, wires=0)
            noise(op=None, noise_param=0.3)
            qml.Hadamard(wires=0)
            noise(op=None, noise_param=0.3)
            qml.T(wires=0)
            noise(op=None, noise_param=0.3)
            m0 = qml.measure(0)
            noise(op=None, noise_param=0.3)
            m1 = qml.measure(1)
            noise(op=None, noise_param=0.3)
            qml.cond(~m0 & m1 == 0, qml.X)(wires=2)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

        assert np.allclose(noisy_circuit(0.4), explicit_circuit(0.4))

    # pylint:disable = cell-var-from-loop
    def test_add_noise_with_readout_errors(self):
        """Test that a add_noise works with readout errors."""
        dev = qml.device("default.mixed", wires=2)

        fc, fn = qml.noise.op_in([qml.RY, qml.RZ]), qml.noise.partial_wires(
            qml.AmplitudeDamping, 0.4
        )
        mc, mn = (qml.noise.meas_eq(qml.expval) | qml.noise.meas_eq(qml.var)) & qml.noise.wires_in(
            [0, 1]
        ), qml.noise.partial_wires(qml.PhaseFlip, 0.2)

        @partial(add_noise, noise_model=qml.NoiseModel({fc: fn}, {mc: mn}))
        @qml.qnode(dev)
        def f_noisy(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.probs(op=qml.Z(0) @ qml.Z(1)),
                qml.purity(wires=0),
                qml.var(qml.PauliZ(0) @ qml.PauliZ(1)),
            )

        args = [0.1, 0.2, 0.3, 0.4]

        results = []
        for mp in [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.var(qml.PauliZ(0) @ qml.PauliZ(1)),
        ]:

            @qml.qnode(dev)
            def f(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.AmplitudeDamping(0.4, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.AmplitudeDamping(0.4, wires=0)
                qml.RX(z, wires=1)
                qml.PhaseFlip(0.2, wires=0)
                qml.PhaseFlip(0.2, wires=1)
                return qml.apply(mp)

            results.append(f(*args))

        for mp in [qml.probs(op=qml.Z(0) @ qml.Z(1)), qml.purity(wires=0)]:

            @qml.qnode(dev)
            def g(w, x, y, z):
                qml.RX(w, wires=0)
                qml.RY(x, wires=1)
                qml.AmplitudeDamping(0.4, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(y, wires=0)
                qml.AmplitudeDamping(0.4, wires=0)
                qml.RX(z, wires=1)
                return qml.apply(mp)

            results.append(g(*args))

        noise_res = f_noisy(*args)
        assert qml.math.allclose(results[0], noise_res[0])
        assert qml.math.allclose(results[2], noise_res[1])
        assert qml.math.allclose(results[3], noise_res[2])
        assert qml.math.allclose(results[1], noise_res[3])


class TestAddNoiseLevels:
    """Tests for custom insertion of add_noise transform at correct level."""

    def test_level_none_deprecation(self):
        """Test that using level=None raises a deprecation warning."""
        dev = qml.device("default.mixed", wires=2)

        @qml.metric_tensor
        @qml.transforms.undo_swaps
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": np.pi / 4})
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        fcond = qml.noise.op_eq(qml.RX)
        fcall = qml.noise.partial_wires(qml.PhaseDamping, 0.4)
        noise_model = qml.NoiseModel({fcond: fcall})
        with pytest.warns(
            qml.exceptions.PennyLaneDeprecationWarning,
            match="`level=None` is deprecated",
        ):
            add_noise(f, noise_model=noise_model, level=None)

    @pytest.mark.parametrize(
        "level1, level2",
        [
            ("top", 0),
            (0, slice(0, 0)),
            ("user", 4),
            ("user", slice(0, 4)),
            (-1, slice(0, -1)),
            ("device", slice(0, None)),
        ],
    )
    def test_add_noise_level(self, level1, level2):
        """Test that add_noise can be inserted to correct level in the TransformProgram"""
        dev = qml.device("default.mixed", wires=2)

        @qml.metric_tensor
        @qml.transforms.undo_swaps
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": np.pi / 4})
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        fcond = qml.noise.op_eq(qml.RX)
        fcall = qml.noise.partial_wires(qml.PhaseDamping, 0.4)
        noise_model = qml.NoiseModel({fcond: fcall})

        noisy_qnode = add_noise(f, noise_model=noise_model, level=level1)

        transform_level1 = noisy_qnode.transform_program
        transform_level2 = qml.workflow.get_transform_program(f, level=level2)
        transform_level2.add_transform(add_noise, noise_model=noise_model, level=level1)

        assert len(transform_level1) == len(transform_level2) + bool(level1 == "user")
        for t1, t2 in zip(transform_level1, transform_level2):
            if t1.transform.__name__ == t2.transform.__name__ == "expand_fn":
                continue
            assert t1 == t2

    def test_add_noise_level_with_final(self):
        """Test that add_noise can be inserted in the TransformProgram with a final transform"""
        dev = qml.device("default.mixed", wires=2)

        @qml.metric_tensor
        @qml.transforms.undo_swaps
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": np.pi / 4})
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        fcond = qml.noise.op_eq(qml.RX)
        fcall = qml.noise.partial_wires(qml.PhaseDamping, 0.4)
        noise_model = qml.NoiseModel({fcond: fcall})

        noisy_qnode = add_noise(f, noise_model=noise_model)

        transform_level1 = qml.workflow.get_transform_program(f)
        transform_level2 = qml.workflow.get_transform_program(noisy_qnode)

        assert len(transform_level1) == len(transform_level2) - 1
        assert transform_level2[4].transform == add_noise.transform
        assert transform_level2[-1].transform == qml.metric_tensor.transform
