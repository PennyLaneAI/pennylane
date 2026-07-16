# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Test base Resource class and its associated methods
"""

# pylint: disable=unnecessary-dunder-call,protected-access
import textwrap
from dataclasses import FrozenInstanceError

import pytest

import pennylane as qp
from pennylane.core.qscript import QuantumScript
from pennylane.core.shots import Shots
from pennylane.resource.expression import Expression
from pennylane.resource.resource import (
    CircuitSpecs,
    PBCSpecsResources,
    SpecsResources,
    _count_to_str,
    num_to_letters,
    resources_from_tape,
)


@pytest.mark.parametrize("compute_depth", (True, False))
def test_specs_compute_depth(compute_depth):
    """Test that depth is skipped with `resources_from_tape`."""

    ops = [
        qp.RX(0.432, wires=0),
        qp.Rot(0.543, 0, 0.23, wires=0),
        qp.CNOT(wires=[0, "a"]),
        qp.RX(0.133, wires=4),
    ]
    obs = [qp.expval(qp.PauliX(wires="a")), qp.probs(wires=[0, "a"])]

    tape = QuantumScript(ops=ops, measurements=obs)
    resources = resources_from_tape(tape, compute_depth=compute_depth)

    assert resources.depth == (3 if compute_depth else None)


class TestSpecsResources:
    """Test the methods and attributes of the SpecsResource class"""

    @pytest.fixture
    def example_specs_resource(self):
        """Generate an example SpecsResources instance."""
        return SpecsResources(
            counts={"Hadamard": 2, "CNOT": 1},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=2,
            circuit_depth=2,
        )

    def test_depth_autoassign(self):
        """Test that the SpecsResources class auto-assigns depth as None if not provided."""

        s = SpecsResources(
            counts={"Hadamard": 2, "CNOT": 1},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=2,
        )

        assert s.depth is None

    def test_total_operations(self, example_specs_resource):
        """Test that the SpecsResources class handles `total_quantum_operations` as expected."""

        s = example_specs_resource

        assert s.total_quantum_operations == 3

    def test_immutable(self, example_specs_resource):
        """Test that SpecsResources is immutable."""

        s = example_specs_resource

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.counts = {}

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.measurement_processes = {}

        with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
            s.circuit_depth = 0

    def test_getitem(self, example_specs_resource):
        """Test that SpecsResources supports indexing via __getitem__."""

        s = example_specs_resource

        assert s["counts"] == s.counts
        assert s["quantum_operations"] == s.quantum_operations
        assert s["measurement_processes"] == s.measurement_processes
        assert s["num_allocs"] == s.num_allocs
        assert s["depth"] == s.depth

        assert s["total_quantum_operations"] == s.total_quantum_operations

        # Try nonexistent key
        with pytest.raises(
            KeyError,
            match="key 'potato' not available. Options are ",
        ):
            _ = s["potato"]

    def test_str(self, example_specs_resource):
        """Test the string representation of a SpecsResources instance."""

        s = example_specs_resource

        expected = textwrap.dedent("""\
            Quantum operations:
            - Total: 3
            - Hadamard: 2
            - CNOT: 1
            Measurement processes:
            - expval(PauliZ): 1
            Wire allocations: 2
            Circuit Depth: 2""")

        assert str(s) == expected
        assert s.to_pretty_str() == expected
        expected_indented = textwrap.indent(expected, " " * 4)
        assert s.to_pretty_str(preindent=4) == expected_indented

        # Check with no depth, gates, or measurements

        s = SpecsResources(counts={}, measurement_processes={}, num_allocs=0)

        expected = textwrap.dedent("""\
            Quantum operations:
            - No operations.
            Measurement processes:
            - No measurement processes.
            Wire allocations: 0
            Circuit Depth: Not computed""")

        expected_indented = textwrap.indent(expected, " " * 4)

        assert str(s) == expected
        assert s.to_pretty_str() == expected
        assert s.to_pretty_str(preindent=4) == expected_indented

    def test_to_dict(self, example_specs_resource):
        """Test the to_dict method of SpecsResources."""

        s = example_specs_resource

        expected = {
            "quantum_operations": {"Hadamard": 2, "CNOT": 1},
            "measurement_processes": {"expval(PauliZ)": 1},
            "num_allocs": 2,
            "circuit_depth": 2,
            "total_quantum_operations": 3,
            "vars": frozenset(),
            "extra": {},
        }

        assert s.to_dict() == expected


class TestPBCSpecsResources:
    @pytest.fixture
    def example_pbc_specs_resource(self):
        """Generate an example SpecsResources instance."""
        return PBCSpecsResources(
            counts={"Hadamard": 2, "CNOT": 1},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=2,
            any_commuting_depth=2,
            qubit_disjoint_depth=3,
        )

    def test_str(self, example_pbc_specs_resource):
        """Test the string representation of a SpecsResources instance."""

        s = example_pbc_specs_resource

        expected = textwrap.dedent("""\
            Quantum operations:
            - Total: 3
            - Hadamard: 2
            - CNOT: 1
            Measurement processes:
            - expval(PauliZ): 1
            Wire allocations: 2
            Any Commuting Depth: 2
            Qubit Disjoint Depth: 3""")

        assert str(s) == expected


class TestSymbolicSpecsResources:
    @pytest.fixture
    def example_resource(self) -> SpecsResources:
        """
        Generate an example SpecsResources instance.
        The resources roughly correspond to the following circuit:

        .. code-block:: python

            def circ():
                qp.Hadamard(0)
                qp.PauliX(0)
                for i in range(x):
                    qp.PauliX(i)
                    for _ in range(z):
                        qp.CNOT(wires=[0, 1])
                for j in range(2 * z):
                    qp.PauliZ(j)
                return expval(qp.PauliZ(0))
        """
        return SpecsResources(
            counts={
                "Hadamard": Expression({(): 1}),
                "PauliX": Expression({("x"): 1, (): 1}),
                "CNOT": Expression({("x", "z"): 1}),
                "PauliZ": Expression({("z",): 2}),
            },
            measurement_processes={"expval(PauliZ)": 1},
            # The values for allocs and depth are a bit off, but are helpful for testing substitutions
            num_allocs=Expression({("x",): 1, ("z",): 2, (): 1}),
            circuit_depth=Expression({("x", "z"): 1, ("z",): 2, ("x",): 1, (): 2}),
        )

    @pytest.fixture
    def example_resource_concrete(self) -> SpecsResources:
        """
        Generate an example SpecsResources instance for a non-dynamic circuit.

        Specifically, returns the resources for a simple Bell state circuit with a measurement.
        """
        return SpecsResources(
            counts={"Hadamard": 1, "CNOT": 1},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=1,
            circuit_depth=1,
        )

    def test_total_ops(self, example_resource):
        s = example_resource
        assert s.total_quantum_operations == Expression(
            {("x", "z"): 1, ("z",): 2, ("x",): 1, (): 2}
        )

    def test_blank_subs(self, example_resource):
        s = example_resource
        assert s.subs() == s

    def test_partial_subs(self, example_resource):
        s = example_resource

        # Substitute x=2, leaving z symbolic
        partially_substituted = s.subs({"x": 2})

        expected = SpecsResources(
            counts={
                "Hadamard": Expression({(): 1}),
                "PauliX": Expression({(): 3}),
                "CNOT": Expression({("z",): 2}),
                "PauliZ": Expression({("z",): 2}),
            },
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=Expression({("z",): 2, (): 3}),
            circuit_depth=Expression({("z",): 4, (): 4}),
        )

        assert partially_substituted == expected

    def test_full_subs(self, example_resource):
        s = example_resource

        # Substitute x=2 and z=3
        fully_substituted = s.subs({"x": 2, "z": 3})

        expected = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 3, "CNOT": 6, "PauliZ": 6},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=9,
            circuit_depth=16,
        )

        assert fully_substituted == expected

    def test_subs_kwargs(self, example_resource):
        assert example_resource.subs(x=2, z=3) == example_resource.subs({"x": 2, "z": 3})

    def test_invalid_subs(self, example_resource):
        """Test that the subs method raises a TypeError for invalid substitutions."""
        with pytest.raises(TypeError):
            example_resource.subs({"x": "not an int"})
        with pytest.raises(ValueError):
            example_resource.subs({"not a var": 3})

    def test_eq(self):
        s1 = SpecsResources(
            counts={"Hadamard": Expression({("x,"): 1})},
            measurement_processes={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            circuit_depth=Expression(1),
        )
        s2 = SpecsResources(
            counts={"Hadamard": Expression({("x,"): 1})},
            measurement_processes={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            circuit_depth=Expression(1),
        )
        s3 = SpecsResources(
            counts={"Hadamard": Expression({("z,"): 1})},
            measurement_processes={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            circuit_depth=Expression(1),
        )

        assert s1 == s2
        assert s1 != s3
        assert s2 != s3
        assert s1 != SpecsResources(
            counts={"Hadamard": 1},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=1,
            circuit_depth=1,
        )

    def test_eq_no_var(self):
        s1 = SpecsResources(
            counts={"Hadamard": Expression(1)},
            measurement_processes={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            circuit_depth=Expression(1),
        )

        s2 = SpecsResources(
            counts={"Hadamard": Expression(1)},
            measurement_processes={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            circuit_depth=Expression(1),
        )

        s3 = SpecsResources(
            counts={"Hadamard": Expression(2)},  # different value here
            measurement_processes={"expval(PauliZ)": Expression(1)},
            num_allocs=Expression(1),
            circuit_depth=Expression(1),
        )

        assert s1 == s2
        assert s1 != s3
        assert s2 != s3

        assert s1 == SpecsResources(
            counts={"Hadamard": 1},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=1,
            circuit_depth=1,
        )

    def test_eq_invalid(self, example_resource):
        assert example_resource != "not a SpecsResource"

    def test_str(self, example_resource):
        s = example_resource

        expected = "Symbolic Variables: x, z\n"
        expected += "Quantum operations:\n"
        expected += "- Total: x*z + 2*z + x + 2\n"
        expected += "- Hadamard: 1\n"
        expected += "- PauliX: x + 1\n"
        expected += "- CNOT: x*z\n"
        expected += "- PauliZ: 2*z\n"
        expected += "Measurement processes:\n"
        expected += "- expval(PauliZ): 1\n"
        expected += "Wire allocations: 2*z + x + 1\n"
        expected += "Circuit Depth: x*z + 2*z + x + 2"

        assert str(s) == expected


class TestCircuitSpecs:
    @pytest.fixture
    def example_specs_result(self):
        """Generate an example CircuitSpecs instance."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=SpecsResources(
                counts={"Hadamard": 2, "CNOT": 1},
                measurement_processes={"expval(PauliZ)": 1},
                num_allocs=2,
                circuit_depth=2,
            ),
        )

    @pytest.fixture
    def example_specs_result_multi(self):
        """Generate an example CircuitSpecs instance with multiple levels and batches."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level={1: "l1", 2: "l2"},
            resources={
                1: SpecsResources(
                    counts={"Hadamard": 4, "CNOT": 2},
                    measurement_processes={"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    num_allocs=2,
                    circuit_depth=2,
                ),
                2: [
                    SpecsResources(
                        counts={"CNOT": 1},
                        measurement_processes={"expval(PauliX)": 1},
                        num_allocs=2,
                        circuit_depth=1,
                    ),
                    SpecsResources(
                        counts={"CNOT": 1},
                        measurement_processes={"expval(PauliZ)": 1},
                        num_allocs=2,
                        circuit_depth=1,
                    ),
                ],
            },
        )

    @pytest.fixture
    def example_specs_result_multi_symbolic(self):
        """Generate an example CircuitSpecs instance with multiple levels and batches, as well as symbolic resources."""
        return CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level={1: "l1", 2: "l2"},
            resources={
                1: SpecsResources(
                    counts={
                        "Hadamard": Expression({("x",): 2, (): 2}),
                        "CNOT": Expression({("x",): 2}),
                    },
                    measurement_processes={"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    num_allocs=2,
                    circuit_depth=2,
                ),
                2: [
                    SpecsResources(
                        counts={"CNOT": Expression({("x",): 1})},
                        measurement_processes={"expval(PauliX)": 1},
                        num_allocs=2,
                        circuit_depth=1,
                    ),
                    SpecsResources(
                        counts={"CNOT": Expression({("x",): 1})},
                        measurement_processes={"expval(PauliZ)": 1},
                        num_allocs=2,
                        circuit_depth=1,
                    ),
                ],
            },
        )

    def test_blank_init(self):
        """Test that CircuitSpecss can be instantiated with no arguments."""
        r = CircuitSpecs()  # should not raise any errors

        assert r.device_name is None
        assert r.num_device_wires is None
        assert r.shots is None
        assert r.level is None
        assert r.resources is None

    def test_getitem(self, example_specs_result):
        """Test that CircuitSpecs supports indexing via __getitem__."""

        r = example_specs_result

        assert r["device_name"] == r.device_name
        assert r["num_device_wires"] == r.num_device_wires
        assert r["shots"] == r.shots
        assert r["level"] == r.level
        assert r["resources"] == r.resources

    def test_to_dict(
        self, example_specs_result, example_specs_result_multi, example_specs_result_multi_symbolic
    ):
        """Test the to_dict method of CircuitSpecs."""

        r = example_specs_result

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": 2,
            "resources": {
                "quantum_operations": {"Hadamard": 2, "CNOT": 1},
                "measurement_processes": {"expval(PauliZ)": 1},
                "num_allocs": 2,
                "circuit_depth": 2,
                "total_quantum_operations": 3,
                "vars": frozenset(),
                "extra": {},
            },
        }

        assert r.to_dict() == expected

        r = example_specs_result_multi

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": {1: "l1", 2: "l2"},
            "resources": {
                1: {
                    "quantum_operations": {"Hadamard": 4, "CNOT": 2},
                    "measurement_processes": {"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    "num_allocs": 2,
                    "circuit_depth": 2,
                    "total_quantum_operations": 6,
                    "vars": frozenset(),
                    "extra": {},
                },
                2: [
                    {
                        "quantum_operations": {"CNOT": 1},
                        "measurement_processes": {"expval(PauliX)": 1},
                        "num_allocs": 2,
                        "circuit_depth": 1,
                        "total_quantum_operations": 1,
                        "vars": frozenset(),
                        "extra": {},
                    },
                    {
                        "quantum_operations": {"CNOT": 1},
                        "measurement_processes": {"expval(PauliZ)": 1},
                        "num_allocs": 2,
                        "circuit_depth": 1,
                        "total_quantum_operations": 1,
                        "vars": frozenset(),
                        "extra": {},
                    },
                ],
            },
        }

        assert r.to_dict() == expected

        r = example_specs_result_multi_symbolic

        expected = {
            "device_name": "default.qubit",
            "num_device_wires": 5,
            "shots": Shots(1000),
            "level": {1: "l1", 2: "l2"},
            "resources": {
                1: {
                    "quantum_operations": {
                        "Hadamard": Expression({("x",): 2, (): 2}),
                        "CNOT": Expression({("x",): 2}),
                    },
                    "measurement_processes": {"expval(PauliX)": 1, "expval(PauliZ)": 1},
                    "num_allocs": 2,
                    "circuit_depth": 2,
                    "total_quantum_operations": Expression({("x",): 4, (): 2}),
                    "vars": frozenset({"x"}),
                    "extra": {},
                },
                2: [
                    {
                        "quantum_operations": {"CNOT": Expression({("x",): 1})},
                        "measurement_processes": {"expval(PauliX)": 1},
                        "num_allocs": 2,
                        "circuit_depth": 1,
                        "total_quantum_operations": Expression({("x",): 1}),
                        "vars": frozenset({"x"}),
                        "extra": {},
                    },
                    {
                        "quantum_operations": {"CNOT": Expression({("x",): 1})},
                        "measurement_processes": {"expval(PauliZ)": 1},
                        "num_allocs": 2,
                        "circuit_depth": 1,
                        "total_quantum_operations": Expression({("x",): 1}),
                        "vars": frozenset({"x"}),
                        "extra": {},
                    },
                ],
            },
        }

        assert r.to_dict() == expected

    def test_str(self, example_specs_result):
        """Test the string representation of a CircuitSpecs instance."""

        r = example_specs_result

        expected = "Device: default.qubit\n"
        expected += "Device wires: 5\n"
        expected += "Shots: Shots(total=1000)\n"
        expected += "Level: 2\n"
        expected += "\n"
        expected += r.resources.to_pretty_str()

        assert str(r) == expected

    def test_str_multi_tabular(self, example_specs_result_multi):
        """Test the tabular string representation of a CircuitSpecs instance."""

        r = example_specs_result_multi
        assert str(r) == textwrap.dedent("""\
        Device: default.qubit
        Device wires: 5
        Shots: Shots(total=1000)
        Levels:
        - 1: l1
        - 2: l2

        ↓Metric         Level→ |    1 |  2-a |  2-b
        -------------------------------------------
        Quantum operations:    |
        - Total                |    6 |    1 |    1
        - Hadamard             |    4 |    0 |    0
        - CNOT                 |    2 |    1 |    1
        Measurement processes: |
        - expval(PauliX)       |    1 |    1 |    0
        - expval(PauliZ)       |    1 |    0 |    1
        Wire allocations       |    2 |    2 |    2""")

    def test_str_multi_tabular_symbolic(self, example_specs_result_multi_symbolic):
        """Test the tabular string representation of a CircuitSpecs instance with symbolic resources."""

        r = example_specs_result_multi_symbolic
        assert str(r) == textwrap.dedent("""\
            Device: default.qubit
            Device wires: 5
            Shots: Shots(total=1000)
            Levels:
            - 1: l1
            - 2: l2

            ↓Metric         Level→ |     1 |   2-a |   2-b
            ----------------------------------------------
            Quantum operations:    |
            - Total                | 4*x+2 |     x |     x
            - Hadamard             | 2*x+2 |     0 |     0
            - CNOT                 |   2*x |     x |     x
            Measurement processes: |
            - expval(PauliX)       |     1 |     1 |     0
            - expval(PauliZ)       |     1 |     0 |     1
            Wire allocations       |     2 |     2 |     2""")

    def test_str_multi_non_tabular(self, example_specs_result_multi):
        """Test the non-tabular string representation of a CircuitSpecs instance."""
        r = example_specs_result_multi

        expected = "Device: default.qubit\n"
        expected += "Device wires: 5\n"
        expected += "Shots: Shots(total=1000)\n"
        expected += "Levels:\n"
        expected += "- 1: l1\n"
        expected += "- 2: l2\n"
        expected += "\n\n"

        expected += "Level = 1:\n"
        expected += r.resources[1].to_pretty_str(preindent=4)

        expected += "\n\n" + "-" * 60 + "\n\n"

        expected += "Level = 2:\n"
        expected += "    Batched tape a:\n"
        expected += r.resources[2][0].to_pretty_str(preindent=8)
        expected += "\n\n    Batched tape b:\n"
        expected += r.resources[2][1].to_pretty_str(preindent=8)

        assert r.to_pretty_str(tabular=False) == expected


class TestIPythonDisplays:
    """
    Test the IPython display methods for all applicable resource classes.

    Note that we don't test the `_ipython_display_` methods directly since this method does not
    return a value (it uses side-effects only). Instead, we check the output of the _repr_markdown_
    or other related methods.

    See also: https://ipython.readthedocs.io/en/stable/config/integrating.html#custom-methods
    """

    @pytest.fixture
    def example_specs_resource(self) -> SpecsResources:
        return SpecsResources(
            # Pick a number that forces scientific notation
            counts={"Hadamard": 1, "CNOT": 100_001},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=2,
            circuit_depth=2,
        )

    @pytest.fixture
    def example_symbolic_specs_resource(self) -> SpecsResources:
        return SpecsResources(
            counts={
                "Hadamard": Expression({("a", "a", "b"): 1, ("a", "a"): 1, ("a",): 1}),
                "CNOT": 1,
            },
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=2,
            circuit_depth=2,
        )

    @pytest.fixture
    def example_pbc_specs_resource(self) -> PBCSpecsResources:
        return PBCSpecsResources(
            counts={"Hadamard": 1, "CNOT": 100_001},
            measurement_processes={"expval(PauliZ)": 1},
            num_allocs=2,
            any_commuting_depth=2,
            qubit_disjoint_depth=3,
        )

    def test_specs_resources_ipython_display(self, example_specs_resource):
        """Test the IPython display of a SpecsResources instance."""
        expected = textwrap.dedent("""\
            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |
        """)
        actual = example_specs_resource._repr_markdown_()

        assert actual.strip() == expected.strip()

    def test_symbolic_specs_resources_ipython_display(self, example_symbolic_specs_resource):
        """Test the IPython display of a SpecsResources instance with symbolic data."""
        expected = textwrap.dedent("""\
            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | a\\*a\\*b + a\\*a + a + 1 |
            | Hadamard | a\\*a\\*b + a\\*a + a |
            | CNOT | 1 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |
        """)
        actual = example_symbolic_specs_resource._repr_markdown_()

        assert actual.strip() == expected.strip()

    def test_pbc_specs_resources_ipython_display(self, example_pbc_specs_resource):
        """Test the IPython display of a SpecsResources instance."""
        expected = textwrap.dedent("""\
            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **PBC Depths** | |
            | Any Commuting Depth | 2 |
            | Qubit Disjoint Depth | 3 |
        """)
        actual = example_pbc_specs_resource._repr_markdown_()

        assert actual.strip() == expected.strip()

    def test_single_level_circuit_specs_ipython_display(self, example_specs_resource):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=example_specs_resource,
        )
        actual = s._repr_markdown_(collapsible=False)
        expected = textwrap.dedent("""\
            **Circuit Specs:**

            | Metric | Value |
            | :--- | ---: |
            | **Device** | default.qubit |
            | **Device wires** | 5 |
            | **Shots** | Shots(total=1000) |
            | **Level** | 2 |

            **Resources:**

            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |
        """)

        assert actual.strip() == expected.strip()

    def test_single_level_circuit_specs_ipython_display_collapsible(self, example_specs_resource):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=example_specs_resource,
        )
        actual = s._repr_markdown_(collapsible=True)
        expected = textwrap.dedent("""\
            <details open>
            <summary>Circuit Specs</summary>

            | Metric | Value |
            | :--- | ---: |
            | **Device** | default.qubit |
            | **Device wires** | 5 |
            | **Shots** | Shots(total=1000) |
            | **Level** | 2 |

            </details>
            <details open>
            <summary>Resources</summary>

            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |

            </details>
        """)

        assert actual.strip() == expected.strip()

    def test_batched_circuit_specs_ipython_display(self, example_specs_resource):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=[example_specs_resource, example_specs_resource],
        )
        actual = s._repr_markdown_(collapsible=False)
        expected = textwrap.dedent("""\
            **Circuit Specs:**

            | Metric | Value |
            | :--- | ---: |
            | **Device** | default.qubit |
            | **Device wires** | 5 |
            | **Shots** | Shots(total=1000) |
            | **Level** | 2 |

            **Resources:**

            **Batched tape a:**

            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |

            **Batched tape b:**

            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |
        """)

        assert actual.strip() == expected.strip()

    def test_batched_circuit_specs_ipython_display_collapsible(self, example_specs_resource):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level=2,
            resources=[example_specs_resource, example_specs_resource],
        )
        actual = s._repr_markdown_(collapsible=True)
        expected = textwrap.dedent("""\
            <details open>
            <summary>Circuit Specs</summary>

            | Metric | Value |
            | :--- | ---: |
            | **Device** | default.qubit |
            | **Device wires** | 5 |
            | **Shots** | Shots(total=1000) |
            | **Level** | 2 |

            </details>
            <details open>
            <summary>Resources</summary>

            <details open>
            <summary>Batched tape a</summary>

            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |

            </details>
            <details open>
            <summary>Batched tape b</summary>

            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | Total | 1.000E+5 |
            | Hadamard | 1 |
            | CNOT | 1.000E+5 |
            | **Measurement processes:** | |
            | expval(PauliZ) | 1 |
            | **Wire allocations** | 2 |
            | **Circuit depth** | 2 |

            </details>

            </details>
            """)

        assert actual.strip() == expected.strip()

    def test_multi_level_circuit_specs_ipython_display(
        self, example_symbolic_specs_resource, example_specs_resource
    ):
        """Test the IPython display of a single-level CircuitSpecs instance."""
        s = CircuitSpecs(
            device_name="default.qubit",
            num_device_wires=5,
            shots=Shots(1000),
            level={0: "l1", 1: "l2"},
            resources={
                0: example_symbolic_specs_resource,
                1: [example_specs_resource, example_specs_resource],
            },
        )
        actual = s._repr_markdown_(collapsible=False)
        expected = textwrap.dedent("""\
            **Circuit Specs:**

            | Metric | Value |
            | :--- | ---: |
            | **Device** | default.qubit |
            | **Device wires** | 5 |
            | **Shots** | Shots(total=1000) |
            | **Levels** | |
            | 0 | l1 |
            | 1 | l2 |

            **Resources:**

            | ↓Metric / Level→ | 0 | 1-a | 1-b |
            | :--- | ---: | ---: | ---: |
            | **Quantum operations** |  |  |  |
            | Total | a\\*a\\*b + a\\*a + a + 1 | 1.000E+5 | 1.000E+5 |
            | Hadamard | a\\*a\\*b + a\\*a + a | 1 | 1 |
            | CNOT | 1 | 1.000E+5 | 1.000E+5 |
            | **Measurement processes** |  |  |  |
            | expval(PauliZ) | 1 | 1 | 1 |
            | **Wire allocations** | 2 | 2 | 2 |
        """)

        assert actual.strip() == expected.strip()

    def test_empty_resources_ipython_display(self):
        """Test the IPython display of an empty SpecsResources instance."""
        s = SpecsResources(
            counts={},
            measurement_processes={},
            num_allocs=1,
        )
        actual = s._repr_markdown_()
        expected = textwrap.dedent("""\
            | **Metric** | **Value** |
            | :--- | ---: |
            | **Quantum operations:** | |
            | *No operations* | |
            | **Measurement processes:** | |
            | *No measurement processes* | |
            | **Wire allocations** | 1 |
            | **Circuit depth** | Not computed |
        """)

        assert actual.strip() == expected.strip()


def test_count_to_str():
    """Test the _count_to_str helper function."""
    assert _count_to_str(0) == "0"
    assert _count_to_str(999) == "999"
    assert _count_to_str(1_000) == "1,000"
    assert _count_to_str(10_000) == "10,000"
    assert _count_to_str(100_000) == "1.000E+5"
    assert _count_to_str(12_345_678) == "1.235E+7"
    assert _count_to_str(Expression(0)) == "0"
    assert _count_to_str(Expression(15)) == "15"
    assert _count_to_str(Expression(100_000)) == "1.000E+5"
    assert _count_to_str(Expression({("y",): 3, ("x",): 2})) == "3*y + 2*x"


def test_num_to_letters():
    """Test the num_to_letters helper function."""
    assert num_to_letters(0) == "a"
    assert num_to_letters(1) == "b"
    assert num_to_letters(25) == "z"
    assert num_to_letters(26) == "aa"
    assert num_to_letters(27) == "ab"
    assert num_to_letters(51) == "az"
    assert num_to_letters(52) == "ba"
