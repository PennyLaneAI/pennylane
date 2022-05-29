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
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml

from pennylane.ops.qubit.attributes import Attribute

# Dummy attribute
new_attribute = Attribute(["PauliX", "PauliY", "PauliZ", "Hadamard", "RZ"])


class TestAttribute:
    """Test addition and inclusion of operations and subclasses in attributes."""

    def test_invalid_input(self):
        """Test that anything that is not a string or Operation throws an error."""
        # Test something that is not an object
        with pytest.raises(TypeError, match="can be checked for attribute inclusion"):
            assert 3 not in new_attribute

        # Test a dummy object that is not an Operation.
        with pytest.raises(TypeError, match="can be checked for attribute inclusion"):
            assert object() not in new_attribute

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

    def test_tensor_check(self):
        """Test that we can ask if a tensor is in the attribute."""
        assert not qml.PauliX(wires=0) @ qml.PauliZ(wires=1) in new_attribute


class TestSupportsBroadcasting:
    """Test that all operations in the ``supports_broadcasting`` attribute
    actually support broadcasting."""

    def get_args_and_kwargs(self, name):
        """Generate broadcasted parameters, wires and potential keyword arguments
        for operations that support broadcasting; Batch size always is 3."""
        cls = getattr(qml, name)
        # Default parameters and wires
        par = (np.array([0.25, 2.1, -0.42]), np.array([0.932, 0.32, 1.2]), np.array([-2.141, 0.21, -3.12]))[:cls.num_params]
        wires = list(range(cls.num_wires))
        kwargs = {}
        if name == "DiagonalQubitUnitary":
            par = (np.array([[1j, 1, 1, -1j], [-1j, 1j, 1, -1], [1j, -1j, 1., -1]]),)
            wires = ["a", 5]
        elif name in ("QubitUnitary", "ControlledQubitUnitary"):
            par = (np.array([unitary_group.rvs(4, random_state=state) for state in [91, 1, 4]]),)
            wires = [0, "9"]
            if name.startswith("Controlled"):
                kwargs["control_wires"] = [1, "10"]
        elif name == "MultiRZ":
            wires = [3, "9", 9]
        elif cls.ndim_params == (0,) * cls.num_params:
            if name=="PauliRot":
                kwargs["pauli_word"] = "XYZ"
                wires = [6, 1, "aux"]

        return par, wires, kwargs

    @pytest.mark.parametrize("name", qml.ops.qubit.attributes.supports_tensorbatching)
    def test_broadcast_init(self, name):
        par, wires, kwargs = self.get_args_and_kwargs(name)
        op = getattr(qml, name)(*par, wires=wires, **kwargs)

    @pytest.mark.parametrize("name", qml.ops.qubit.attributes.supports_tensorbatching)
    def test_broadcasted_matrix(self, name):
        par, wires, kwargs = self.get_args_and_kwargs(name)
        cls = getattr(qml, name)
        op = cls(*par, wires=wires, **kwargs)
        mat1 = op.matrix()
        mat2 = cls.compute_matrix(*par, **kwargs)
        single_pars = [tuple(p[i] for p in par) for i in range(3)]
        single_mats = [cls(*_par, wires=wires, **kwargs).matrix() for _par in single_pars]

        assert qml.math.allclose(mat1, single_mats)
        assert qml.math.allclose(mat2, single_mats)

