# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the SelectPauliRot template.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


def get_tape(angles, wires):
    """Auxiliary function to generate a tape with the operator given some angles"""

    return qml.tape.QuantumScript(
        [
            qml.SelectPauliRot(
                angles,
                control_wires=wires["control"],
                target_wire=wires["target"],
                rot_axis="Y",
            )
        ],
        [qml.state()],
    )


class TestSelectPauliRot:

    @pytest.mark.jax
    def test_standard_validity(self):
        """Check the operation using the assert_valid function."""

        wires = qml.registers({"control_wires": 3, "target_wire": 1})

        op = qml.SelectPauliRot(
            angles=qml.math.ones(8),
            control_wires=wires["control_wires"],
            target_wire=wires["target_wire"],
            rot_axis="X",
        )

        qml.ops.functions.assert_valid(op)

    @pytest.mark.parametrize(
        ("angles", "rot_axis", "target_wire", "msg_match"),
        [
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                "K",
                2,
                "'rot_axis' can only take the values 'X', 'Y' and 'Z'.",
            ),
            (
                np.array([1.0]),
                "Z",
                2,
                "Number of angles must",
            ),
            (
                np.array([1.0, 2.0, 3.0, 4.0]),
                "Z",
                [2, 3],
                "Only one target wire can",
            ),
        ],
    )
    def test_SelectPauliRot_error(self, angles, rot_axis, target_wire, msg_match):
        """Test that proper errors are raised for SelectPauliRot"""

        wires = qml.registers({"control_wires": 2})

        with pytest.raises(ValueError, match=msg_match):
            qml.SelectPauliRot(
                angles=angles,
                control_wires=wires["control_wires"],
                target_wire=target_wire,
                rot_axis=rot_axis,
            )

    @pytest.mark.parametrize(
        ("angles", "rot_axis"),
        [
            (np.array([1, 2, 3, 4, 5, 6, 7, 8]), "Z"),
            (np.array([12, 2, 3, 4, 5.4, 6, 7, 8]), "X"),
            (np.array([1, 2, -3, 4, 5, 26, 7, 8]), "Y"),
        ],
    )
    def test_correctness(self, angles, rot_axis):
        """Tests the correctness of the SelectPauliRot template.
        This is done by comparing the results with the naive Select(Rotation) decomposition
        """

        dev = qml.device("default.qubit", wires=4)

        gate = {"Z": qml.RZ, "Y": qml.RY, "X": qml.RX}

        # Check that applying the SelectPauliRot and adjoint(Select) to a state,
        # does not modify the state
        qs = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.SelectPauliRot(
                    angles, control_wires=[0, 1, 2], target_wire=3, rot_axis=rot_axis
                ),
                qml.Select([gate[rot_axis](-angle, 3) for angle in angles], control=[0, 1, 2]),
                qml.Hadamard(0),
                qml.Hadamard(1),
            ],
            [qml.state()],
        )

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert np.isclose(1.0, output[0])

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("axis", "XYZ")
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_decomposition(self, n, axis, batch_dim, seed):
        """Test that the correct gates are added in the decomposition"""

        np.random.seed(seed)
        shape = (2**n,) if batch_dim is None else (batch_dim, 2**n)
        x = np.random.random(shape)
        decomposition = qml.SelectPauliRot.compute_decomposition(
            x, control_wires=range(n), target_wire=n, rot_axis=axis
        )
        decomposition_2 = qml.SelectPauliRot(
            x, control_wires=range(n), target_wire=n, rot_axis=axis
        ).decomposition()

        for dec in [decomposition, decomposition_2]:
            if axis == "Y":
                assert dec[0].name == "Adjoint(S)"
                assert dec[-1].name == "S"
                dec = dec[1:-1]
            if axis in "XY":
                assert dec[0].name == "Hadamard"
                assert dec[-1].name == "Hadamard"
                dec = dec[1:-1]
            # Remaining decomposition is the same for all axis types
            assert len(dec) == 2 * 2**n
            for gate in dec[::2]:
                assert gate.name == "RZ"
                assert gate.batch_size == batch_dim
            for gate in dec[1::2]:
                assert gate.name == "CNOT"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("axis", "XYZ")
    def test_decomposition_new(self, n, axis):
        """Tests that the decomposition of SelectPauliRot is correct with the new system."""

        angles = np.random.random((2**n,))
        op = qml.SelectPauliRot(angles, control_wires=range(n), target_wire=n, rot_axis=axis)
        for rule in qml.list_decomps("SelectPauliRot"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.jax
    def test_interface_jax(self):
        """Test that SelectPauliRot works with jax"""

        from jax import numpy as jnp

        angles = [1, 2, 3, 4]

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        qs = get_tape(jnp.array(angles), wires)

        program, _ = dev.preprocess()
        tape = program([qs])
        output_jax = dev.execute(tape[0])[0]

        qs = get_tape(angles, wires)

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_jax)

    @pytest.mark.torch
    def test_interface_torch(self):
        """Test that SelectPauliRot works with torch"""

        import torch

        angles = [1, 2, 3, 4]

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        qs = get_tape(torch.tensor(angles), wires)

        program, _ = dev.preprocess()
        tape = program([qs])
        output_torch = dev.execute(tape[0])[0]

        qs = get_tape(angles, wires)

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_torch)

    @pytest.mark.tf
    def test_interface_tf(self):
        """Test that SelectPauliRot works with tensorflow"""

        import tensorflow as tf

        angles = [1, 2, 3, 4]

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        qs = get_tape(tf.Variable(angles), wires)

        program, _ = dev.preprocess()
        tape = program([qs])
        output_tf = dev.execute(tape[0])[0]

        qs = get_tape(angles, wires)

        program, _ = dev.preprocess()
        tape = program([qs])
        output = dev.execute(tape[0])[0]

        assert qml.math.allclose(output, output_tf)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test that SelectPauliRot works with jax"""

        import jax
        import jax.numpy as jnp

        angles = jnp.array([1.0, 2.0, 3.0, 4.0])

        wires = qml.registers({"control": 2, "target": 1})
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(angles):
            qml.SelectPauliRot(
                angles,
                control_wires=wires["control"],
                target_wire=wires["target"],
                rot_axis="Y",
            )
            return qml.state()

        expected_output = circuit(angles)
        generated_output = jax.jit(circuit)(angles)

        assert np.allclose(expected_output, generated_output)
        assert qml.math.get_interface(generated_output) == "jax"
