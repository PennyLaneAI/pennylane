# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the available qubit state preparation operations.
"""
import itertools as it

import numpy as np
import pytest
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.ops.qubit.attributes import Attribute, has_unitary_generator

# Dummy attribute
new_attribute = Attribute(["PauliX", "PauliY", "PauliZ", "Hadamard", "RZ"])


class TestAttribute:
    """Test addition and inclusion of operations and subclasses in attributes."""

    def test_invalid_input(self):
        """Test that anything that is not a string or Operation returns False."""
        assert 3 not in new_attribute

        # Test a dummy object that is not an Operation.
        assert object() not in new_attribute

    def test_measurement_process_input(self):
        """Test that MeasurementProcesses are valid objects to check inside Attributes"""
        assert qml.measurements.MidMeasureMP(0) not in new_attribute
        assert qml.expval(qml.PauliX(0)) not in new_attribute

    def test_string_inclusion(self):
        """Test that we can check inclusion using strings."""
        assert "PauliX" in new_attribute
        assert "RX" not in new_attribute

    def test_operation_class_inclusion(self):
        """Test that we can check inclusion using Operations."""
        assert qml.PauliZ(0) in new_attribute
        assert qml.RX(0.5, wires=0) not in new_attribute

    def test_operation_subclass_inclusion(self):
        """Test that we can check inclusion using subclasses of Operations, whether
        or not anything has been instantiated."""
        assert qml.RZ in new_attribute
        assert qml.RX not in new_attribute

    def test_invalid_addition(self):
        """Test that an error is raised if we try to add something
        other than an Operation or a string."""
        with pytest.raises(TypeError, match="can be added to an attribute"):
            new_attribute.add(0)

        assert len(new_attribute) == 5

        with pytest.raises(TypeError, match="can be added to an attribute"):
            new_attribute.add(object())

        assert len(new_attribute) == 5

    def test_inclusion_after_addition(self):
        """Test that we can add operators to the set in multiple ways."""
        new_attribute.add("RX")
        new_attribute.add(qml.PhaseShift(0.5, wires=0))
        new_attribute.add(qml.RY)

        assert "RX" in new_attribute
        assert "PhaseShift" in new_attribute
        assert "RY" in new_attribute
        assert len(new_attribute) == 8


single_scalar_single_wire_ops = [
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "U1",
]

single_scalar_multi_wire_ops = [
    "ControlledPhaseShift",
    "CRX",
    "CRY",
    "CRZ",
    "IsingXX",
    "IsingYY",
    "IsingZZ",
    "IsingXY",
    "SingleExcitation",
    "SingleExcitationPlus",
    "SingleExcitationMinus",
    "DoubleExcitation",
    "DoubleExcitationPlus",
    "DoubleExcitationMinus",
    "OrbitalRotation",
    "FermionicSWAP",
]

two_scalar_single_wire_ops = [
    "U2",
]

three_scalar_single_wire_ops = [
    "Rot",
    "U3",
]

three_scalar_multi_wire_ops = [
    "CRot",
]

# When adding an operation to the following list, you
# actually need to write a new test!
separately_tested_ops = [
    "QubitUnitary",
    "ControlledQubitUnitary",
    "DiagonalQubitUnitary",
    "SpecialUnitary",
    "PauliRot",
    "MultiRZ",
    "StatePrep",
    "AmplitudeEmbedding",
    "AngleEmbedding",
    "IQPEmbedding",
    "QAOAEmbedding",
    "PCPhase",
]


class TestSupportsBroadcasting:
    """Test that all operations in the ``supports_broadcasting`` attribute
    actually support broadcasting."""

    def test_all_marked_operations_are_tested(self):
        """Test that the subsets of the ``supports_broadcasting`` attribute
        defined above cover the entire attribute."""
        tested_ops = set(
            it.chain.from_iterable(
                [
                    single_scalar_single_wire_ops,
                    single_scalar_multi_wire_ops,
                    two_scalar_single_wire_ops,
                    three_scalar_single_wire_ops,
                    three_scalar_multi_wire_ops,
                    separately_tested_ops,
                ]
            )
        )

        assert tested_ops == qml.ops.qubit.attributes.supports_broadcasting

    @pytest.mark.parametrize("name", single_scalar_single_wire_ops)
    def test_single_scalar_single_wire_ops(self, name):
        """Test that single-scalar-parameter operations on a single wire marked
        as supporting parameter broadcasting actually do support broadcasting."""
        par = np.array([0.25, 2.1, -0.42])
        wires = ["wire0"]

        cls = getattr(qml, name)
        op = cls(par, wires=wires)

        mat1 = op.matrix()
        mat2 = cls.compute_matrix(par)
        single_mats = [cls(p, wires=wires).matrix() for p in par]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize("name", single_scalar_multi_wire_ops)
    def test_single_scalar_multi_wire_ops(self, name):
        """Test that single-scalar-parameter operations on multiple wires marked
        as supporting parameter broadcasting actually do support broadcasting."""
        par = np.array([0.25, 2.1, -0.42])
        cls = getattr(qml, name)

        # Provide up to 6 wires and take as many as the class requires
        # This assumes that the class does *not* have `num_wires=qml.operation.None`
        wires = ["wire0", 5, 41, "aux_wire", -1, 9][: cls.num_wires]
        op = cls(par, wires=wires)

        mat1 = op.matrix()
        mat2 = cls.compute_matrix(par)
        single_mats = [cls(p, wires=wires).matrix() for p in par]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize("name", two_scalar_single_wire_ops)
    def test_two_scalar_single_wire_ops(self, name):
        """Test that two-scalar-parameter operations on a single wire marked
        as supporting parameter broadcasting actually do support broadcasting."""
        par = (np.array([0.25, 2.1, -0.42]), np.array([-6.2, 0.12, 0.421]))
        wires = ["wire0"]

        cls = getattr(qml, name)
        op = cls(*par, wires=wires)

        mat1 = op.matrix()
        mat2 = cls.compute_matrix(*par)
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [cls(*p, wires=wires).matrix() for p in single_pars]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize("name", three_scalar_single_wire_ops)
    def test_three_scalar_single_wire_ops(self, name):
        """Test that three-scalar-parameter operations on a single wire marked
        as supporting parameter broadcasting actually do support broadcasting."""
        par = (
            np.array([0.25, 2.1, -0.42]),
            np.array([-6.2, 0.12, 0.421]),
            np.array([0.2, 1.1, -5.2]),
        )
        wires = ["wire0"]

        cls = getattr(qml, name)
        op = cls(*par, wires=wires)

        mat1 = op.matrix()
        mat2 = cls.compute_matrix(*par)
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [cls(*p, wires=wires).matrix() for p in single_pars]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize("name", three_scalar_multi_wire_ops)
    def test_three_scalar_multi_wire_ops(self, name):
        """Test that three-scalar-parameter operations on multiple wires marked
        as supporting parameter broadcasting actually do support broadcasting."""
        par = (
            np.array([0.25, 2.1, -0.42]),
            np.array([-6.2, 0.12, 0.421]),
            np.array([0.2, 1.1, -5.2]),
        )
        wires = ["wire0", 214]

        cls = getattr(qml, name)
        op = cls(*par, wires=wires)

        mat1 = op.matrix()
        mat2 = cls.compute_matrix(*par)
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [cls(*p, wires=wires).matrix() for p in single_pars]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    def test_qubit_unitary(self):
        """Test that QubitUnitary, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        U = np.array([unitary_group.rvs(4, random_state=state) for state in [91, 1, 4]])
        wires = [0, "9"]

        op = qml.QubitUnitary(U, wires=wires)

        mat1 = op.matrix()
        mat2 = qml.QubitUnitary.compute_matrix(U)
        single_mats = [qml.QubitUnitary(_U, wires=wires).matrix() for _U in U]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    def test_controlled_qubit_unitary(self):
        """Test that ControlledQubitUnitary, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        U = np.array([unitary_group.rvs(4, random_state=state) for state in [91, 1, 4]])
        target_wires = [0, "9"]
        control_wires = [1, "10"]
        wires = control_wires + target_wires

        op = qml.ControlledQubitUnitary(U, wires=wires)

        mat1 = op.matrix()
        single_mats = [qml.ControlledQubitUnitary(_U, wires=wires).matrix() for _U in U]

        assert qml.math.allclose(mat1, single_mats)

    def test_diagonal_qubit_unitary(self):
        """Test that DiagonalQubitUnitary, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""
        diag = np.array([[1j, 1, 1, -1j], [-1j, 1j, 1, -1], [1j, -1j, 1.0, -1]])
        wires = ["a", 5]

        op = qml.DiagonalQubitUnitary(diag, wires=wires)

        mat1 = op.matrix()
        mat2 = qml.DiagonalQubitUnitary.compute_matrix(diag)
        single_mats = [qml.DiagonalQubitUnitary(d, wires=wires).matrix() for d in diag]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    def test_special_unitary(self):
        """Test that SpecialUnitary, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""
        theta = np.array([[0.2, -0.1, 0.2], [0, 1, 0], [0.4, 0.2, 0.9], [0, 0, 0]])
        wires = ["a"]

        op = qml.SpecialUnitary(theta, wires=wires)

        mat1 = op.matrix()
        mat2 = qml.SpecialUnitary.compute_matrix(theta, 1)
        single_mats = [qml.SpecialUnitary(t, wires=wires).matrix() for t in theta]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize(
        "pauli_word, wires", [("XYZ", [0, "4", 1]), ("II", [1, 5]), ("X", [7])]
    )
    def test_pauli_rot(self, pauli_word, wires):
        """Test that PauliRot, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""
        par = np.array([0.25, 2.1, -0.42])

        op = qml.PauliRot(par, pauli_word, wires=wires)

        mat1 = op.matrix()
        mat2 = qml.PauliRot.compute_matrix(par, pauli_word=pauli_word)
        single_mats = [qml.PauliRot(p, pauli_word, wires=wires).matrix() for p in par]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize("wires", [[0, "4", 1], [1, 5], [7]])
    def test_multi_rz(self, wires):
        """Test that MultiRZ, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""
        par = np.array([0.25, 2.1, -0.42])

        op = qml.MultiRZ(par, wires=wires)

        mat1 = op.matrix()
        mat2 = qml.MultiRZ.compute_matrix(par, num_wires=len(wires))
        single_mats = [qml.MultiRZ(p, wires=wires).matrix() for p in par]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

    @pytest.mark.parametrize(
        "state_, num_wires",
        [([1.0, 0.0], 1), ([0.5, -0.5j, 0.5, -0.5], 2), (np.ones(8) / np.sqrt(8), 3)],
    )
    def test_qubit_state_vector(self, state_, num_wires):
        """Test that StatePrep, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        state = np.array([state_])
        op = qml.StatePrep(state, wires=list(range(num_wires)))
        assert op.batch_size == 1
        qml.StatePrep.compute_decomposition(state, list(range(num_wires)))
        op.decomposition()

        state = np.array([state_] * 3)
        op = qml.StatePrep(state, wires=list(range(num_wires)))
        assert op.batch_size == 3
        qml.StatePrep.compute_decomposition(state, list(range(num_wires)))
        op.decomposition()

    @pytest.mark.parametrize(
        "state, num_wires",
        [([1.0, 0.0], 1), ([0.5, -0.5j, 0.5, -0.5], 2), (np.ones(8) / np.sqrt(8), 3)],
    )
    def test_amplitude_embedding(self, state, num_wires):
        """Test that AmplitudeEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        features = np.array([state])
        op = qml.AmplitudeEmbedding(features, wires=list(range(num_wires)))
        assert op.batch_size == 1
        qml.AmplitudeEmbedding.compute_decomposition(features, list(range(num_wires)))
        op.decomposition()

        features = np.array([state] * 3)
        op = qml.AmplitudeEmbedding(features, wires=list(range(num_wires)))
        assert op.batch_size == 3
        qml.AmplitudeEmbedding.compute_decomposition(features, list(range(num_wires)))
        op.decomposition()

    @pytest.mark.parametrize(
        "angles, num_wires",
        [
            (np.array([[0.5], [2.1]]), 1),
            (np.array([[0.5, -0.5], [0.2, 1.5]]), 2),
            (np.ones((2, 5)), 5),
        ],
    )
    def test_angle_embedding(self, angles, num_wires):
        """Test that AngleEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        op = qml.AngleEmbedding(angles, wires=list(range(num_wires)))
        assert op.batch_size == 2
        qml.AngleEmbedding.compute_decomposition(angles, list(range(num_wires)), rotation=qml.RX)
        op.decomposition()

    @pytest.mark.parametrize(
        "features, num_wires",
        [
            (np.array([[0.5], [2.1]]), 1),
            (np.array([[0.5, -0.5], [0.2, 1.5]]), 2),
            (np.ones((2, 5)), 5),
        ],
    )
    def test_iqp_embedding(self, features, num_wires):
        """Test that IQPEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        op = qml.IQPEmbedding(features, wires=list(range(num_wires)))
        assert op.batch_size == 2
        qml.IQPEmbedding.compute_decomposition(
            features, list(range(num_wires)), n_repeats=2, pattern=op.hyperparameters["pattern"]
        )
        op.decomposition()

    @pytest.mark.parametrize(
        "features, weights, num_wires, batch_size",
        [
            (np.array([[0.5], [2.1]]), np.array([[0.61], [0.3]]), 1, 2),
            (np.array([[0.5, -0.5], [0.2, 1.5]]), np.ones((2, 4, 3)), 2, 2),
            (np.array([0.5, -0.5, 0.2]), np.ones((3, 2, 6)), 3, 3),
        ],
    )
    def test_qaoa_embedding(self, features, weights, num_wires, batch_size):
        """Test that QAOAEmbedding, which is marked as supporting parameter broadcasting,
        actually does support broadcasting."""

        op = qml.QAOAEmbedding(features, weights, wires=list(range(num_wires)))
        assert op.batch_size == batch_size
        qml.QAOAEmbedding.compute_decomposition(
            features, weights, wires=list(range(num_wires)), local_field=qml.RY
        )
        op.decomposition()

    def test_pcphase(self):
        """Test that the PCPhase matrix works with broadcasted parameters"""
        dim = 2
        size = 4
        broadcasted_phi = [1.23, 4.56, -0.7]

        op = qml.PCPhase(broadcasted_phi, dim=dim, wires=[0, 1])

        mat1 = qml.matrix(op)
        mat2 = op.compute_matrix(*op.parameters, **op.hyperparameters)

        mats = [
            np.diag([np.exp(1j * phi) if i < dim else np.exp(-1j * phi) for i in range(size)])
            for phi in broadcasted_phi
        ]
        expected_mat = np.array(mats)
        assert np.allclose(mat1, expected_mat)
        assert np.allclose(mat2, expected_mat)


all_qubit_operators = sorted(qml.ops.qubit.__all__)  # pylint: disable=no-member
unitarily_generated_ops = sorted(list(has_unitary_generator))


class TestHasUnitaryGenerator:
    """Test that all operations in the ``has_unitary_generator`` attribute
    actually have unitary generators."""

    @pytest.mark.parametrize("entry", unitarily_generated_ops)
    def test_generator_unitarity(self, entry):
        """Test directly that generators of the operators in the ``has_unitary_generator``
        attribute are unitary up to a factor of 2."""
        op_class = getattr(qml, entry)
        phi = 1.23
        wires = [0, 1, 2] if op_class.num_wires is None else list(range(op_class.num_wires))
        if op_class is qml.PauliRot:
            op = op_class(phi, pauli_word="XYZ", wires=wires)  # PauliRot has num_wires == None
        elif op_class is qml.PCPhase:
            op = op_class(phi, dim=(2 ** len(wires) - 1), wires=wires)
        else:
            op = op_class(phi, wires=wires)
        gen = qml.generator(op, format="observable")
        # Some generators are unitary up to a factor - in this case norm of first
        # column will be scaled by this factor, so normalize generator first.
        assert qml.is_unitary(qml.s_prod(1 / np.linalg.norm(qml.matrix(gen), axis=0)[0], gen))

    @pytest.mark.parametrize("entry", all_qubit_operators)
    def test_no_missing_entries(self, entry):
        """Test directly that generators of the operators not in the ``has_unitary_generator``
        attribute are not unitary (up to a factor of 2)."""
        if entry in unitarily_generated_ops:
            pytest.skip("Operator declared as having unitary generator")

        op_class = getattr(qml, entry)

        if not op_class.has_generator:
            pytest.skip("Operator does not have a generator")
        phi = 1.23
        wires = [0, 1, 2] if op_class.num_wires is None else list(range(op_class.num_wires))
        if op_class is qml.PauliRot:
            op = op_class(phi, pauli_word="XYZ", wires=wires)  # PauliRot has num_wires == None
        elif op_class is qml.PCPhase:
            op = op_class(phi, dim=(2 ** len(wires) - 1), wires=wires)
        else:
            op = op_class(phi, wires=wires)
        gen = qml.generator(op, format="observable")
        # Some generators are unitary up to a factor - in this case norm of first
        # column will be scaled by this factor, so normalize generator first.
        # When `gen`` is not unitary, may give divide by zero warning, but non-unitarity
        # can still be confirmed in this case.
        assert not qml.is_unitary(qml.s_prod(1 / np.linalg.norm(qml.matrix(gen), axis=0)[0], gen))
