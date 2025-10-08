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
Unit tests for functions needed for computing the Hamiltonian.
"""
import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
import pytest

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane import numpy as pnp
from pennylane import qchem
from pennylane.fermi import from_string


@pytest.mark.parametrize(
    "coordinates",
    [
        np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
        np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614]),
    ],
)
def test_molecular_hamiltonian_numpy_data(coordinates):
    """Test that if numpy data is used with molecular hamiltonian, that numpy data is outputted."""
    symbols = ["H", "H"]

    H, _ = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    assert all(qml.math.get_interface(d) == "numpy" for d in H.data)


@pytest.mark.torch
def test_error_torch_data_molecular_hamiltonian():
    """Test that an error is raised if torch data is used with molecular hamiltonian."""
    import torch

    x = torch.tensor([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    with pytest.raises(ValueError, match="unsupported interface torch"):
        qml.qchem.molecular_hamiltonian(["H", "H"], x)


@pytest.mark.parametrize(
    ("symbols", "geometry", "core", "active", "e_core", "one_ref", "two_ref"),
    [
        (
            ["H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            None,
            None,
            pnp.array([1.0000000000321256]),
            # computed with OpenFermion using molecule.one_body_integrals
            pnp.array([[-1.39021927e00, -1.28555566e-16], [-3.52805508e-16, -2.91653305e-01]]),
            # computed with OpenFermion using molecule.two_body_integrals
            pnp.array(
                [
                    [
                        [[7.14439079e-01, 6.62555256e-17], [2.45552260e-16, 1.70241444e-01]],
                        [[2.45552260e-16, 1.70241444e-01], [7.01853156e-01, 6.51416091e-16]],
                    ],
                    [
                        [[6.62555256e-17, 7.01853156e-01], [1.70241444e-01, 2.72068603e-16]],
                        [[1.70241444e-01, 2.72068603e-16], [6.51416091e-16, 7.38836693e-01]],
                    ],
                ]
            ),
        ),
        (
            ["H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            [],
            [0, 1],
            pnp.array([1.0000000000321256]),
            # computed with OpenFermion using molecule.one_body_integrals
            pnp.array([[-1.39021927e00, -1.28555566e-16], [-3.52805508e-16, -2.91653305e-01]]),
            # computed with OpenFermion using molecule.two_body_integrals
            pnp.array(
                [
                    [
                        [[7.14439079e-01, 6.62555256e-17], [2.45552260e-16, 1.70241444e-01]],
                        [[2.45552260e-16, 1.70241444e-01], [7.01853156e-01, 6.51416091e-16]],
                    ],
                    [
                        [[6.62555256e-17, 7.01853156e-01], [1.70241444e-01, 2.72068603e-16]],
                        [[1.70241444e-01, 2.72068603e-16], [6.51416091e-16, 7.38836693e-01]],
                    ],
                ]
            ),
        ),
        (
            ["Li", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            [0, 1, 2, 3],
            [4, 5],
            # reference values of e_core, one and two are computed with our initial prototype code
            pnp.array([-5.141222763432437]),
            pnp.array([[1.17563204e00, -5.75186616e-18], [-5.75186616e-18, 1.78830226e00]]),
            pnp.array(
                [
                    [
                        [[3.12945511e-01, 4.79898448e-19], [4.79898448e-19, 9.78191587e-03]],
                        [[4.79898448e-19, 9.78191587e-03], [3.00580620e-01, 4.28570365e-18]],
                    ],
                    [
                        [[4.79898448e-19, 3.00580620e-01], [9.78191587e-03, 4.28570365e-18]],
                        [[9.78191587e-03, 4.28570365e-18], [4.28570365e-18, 5.10996835e-01]],
                    ],
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "use_jax",
    [
        (False),
        pytest.param(True, marks=pytest.mark.jax),
    ],
)
def test_electron_integrals(symbols, geometry, core, active, e_core, one_ref, two_ref, use_jax):
    r"""Test that electron_integrals returns the correct values."""

    if use_jax:
        geometry = qml.math.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], like="jax")

    mol = qchem.Molecule(symbols, geometry)
    args = [geometry, mol.coeff, mol.alpha] if use_jax else []

    e, one, two = qchem.electron_integrals(mol, core=core, active=active)(*args)

    assert pnp.allclose(e, e_core)
    assert pnp.allclose(one, one_ref)
    assert pnp.allclose(two, two_ref)


@pytest.mark.parametrize(
    "use_jax",
    [
        (False),
        pytest.param(True, marks=pytest.mark.jax),
    ],
)
@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "h_ref"),
    [
        (
            ["H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            pnp.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # Hamiltonian coefficients and operators computed with OpenFermion using
            # molecule = openfermion.MolecularData(geometry, basis, multiplicity, charge)
            # molecule = run_pyscf(molecule, run_scf=1)
            # openfermion.transforms.get_fermion_operator(molecule.get_molecular_hamiltonian())
            # The "^" symbols in the operators are replaced with "+"
            1.0000000206358097 * from_string("")
            + -1.3902192781338518 * from_string("0+ 0")
            + 0.35721954051077603 * from_string("0+ 0+ 0 0")
            + 0.08512072166002102 * from_string("0+ 0+ 2 2")
            + 0.35721954051077603 * from_string("0+ 1+ 1 0")
            + 0.08512072166002102 * from_string("0+ 1+ 3 2")
            + 0.08512072166002102 * from_string("0+ 2+ 0 2")
            + 0.3509265790433101 * from_string("0+ 2+ 2 0")
            + 0.08512072166002102 * from_string("0+ 3+ 1 2")
            + 0.3509265790433101 * from_string("0+ 3+ 3 0")
            + 0.35721954051077603 * from_string("1+ 0+ 0 1")
            + 0.08512072166002102 * from_string("1+ 0+ 2 3")
            + -1.3902192781338518 * from_string("1+ 1")
            + 0.35721954051077603 * from_string("1+ 1+ 1 1")
            + 0.08512072166002102 * from_string("1+ 1+ 3 3")
            + 0.08512072166002102 * from_string("1+ 2+ 0 3")
            + 0.3509265790433101 * from_string("1+ 2+ 2 1")
            + 0.08512072166002102 * from_string("1+ 3+ 1 3")
            + 0.3509265790433101 * from_string("1+ 3+ 3 1")
            + 0.35092657904330926 * from_string("2+ 0+ 0 2")
            + 0.08512072166002102 * from_string("2+ 0+ 2 0")
            + 0.35092657904330926 * from_string("2+ 1+ 1 2")
            + 0.08512072166002102 * from_string("2+ 1+ 3 0")
            + -0.29165329244211186 * from_string("2+ 2")
            + 0.08512072166002102 * from_string("2+ 2+ 0 0")
            + 0.36941834777609744 * from_string("2+ 2+ 2 2")
            + 0.08512072166002102 * from_string("2+ 3+ 1 0")
            + 0.36941834777609744 * from_string("2+ 3+ 3 2")
            + 0.35092657904330926 * from_string("3+ 0+ 0 3")
            + 0.08512072166002102 * from_string("3+ 0+ 2 1")
            + 0.35092657904330926 * from_string("3+ 1+ 1 3")
            + 0.08512072166002102 * from_string("3+ 1+ 3 1")
            + 0.08512072166002102 * from_string("3+ 2+ 0 1")
            + 0.36941834777609744 * from_string("3+ 2+ 2 3")
            + -0.29165329244211186 * from_string("3+ 3")
            + 0.08512072166002102 * from_string("3+ 3+ 1 1")
            + 0.36941834777609744 * from_string("3+ 3+ 3 3"),
        )
    ],
)
def test_fermionic_hamiltonian(use_jax, symbols, geometry, alpha, h_ref):
    r"""Test that using fermionic_hamiltonian returns the correct values."""
    if use_jax:
        geometry = qml.math.array(geometry, like="jax")
        alpha = qml.math.array(alpha, like="jax")

    mol = qchem.Molecule(symbols, geometry, alpha=alpha)
    args = [geometry, mol.coeff, mol.alpha] if use_jax else [alpha]
    h = qchem.fermionic_hamiltonian(mol)(*args)

    h.simplify(tol=1e-7)

    assert pnp.allclose(list(h.values()), list(h_ref.values()))
    assert h.keys() == h_ref.keys()


@pytest.mark.parametrize(
    "use_jax",
    [
        (False),
        pytest.param(True, marks=pytest.mark.jax),
    ],
)
@pytest.mark.parametrize(
    ("symbols", "geometry", "h_ref_data"),
    [
        (
            ["H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # computed with qchem.convert_observable and an OpenFermion Hamiltonian; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                pnp.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    I(0),
                    Z(0),
                    Z(1),
                    Z(0) @ Z(1),
                    Y(0) @ X(1) @ X(2) @ Y(3),
                    Y(0) @ Y(1) @ X(2) @ X(3),
                    X(0) @ X(1) @ Y(2) @ Y(3),
                    X(0) @ Y(1) @ Y(2) @ X(3),
                    Z(2),
                    Z(0) @ Z(2),
                    Z(3),
                    Z(0) @ Z(3),
                    Z(1) @ Z(2),
                    Z(1) @ Z(3),
                    Z(2) @ Z(3),
                ],
            ),
        )
    ],
)
def test_diff_hamiltonian(use_jax, symbols, geometry, h_ref_data):
    r"""Test that diff_hamiltonian returns the correct Hamiltonian."""
    if use_jax:
        geometry = qml.math.array(geometry, like="jax")

    mol = qchem.Molecule(symbols, geometry)
    args = [geometry, mol.coeff, mol.alpha] if use_jax else []

    h = qchem.diff_hamiltonian(mol)(*args)

    ops = list(map(qml.simplify, h_ref_data[1]))
    h_ref_data = qml.Hamiltonian(h_ref_data[0], ops)

    assert pnp.allclose(pnp.sort(h.terms()[0]), pnp.sort(h_ref_data.terms()[0]))
    assert qml.Hamiltonian(pnp.ones(len(h.terms()[0])), h.terms()[1]) == (
        qml.Hamiltonian(pnp.ones(len(h_ref_data.terms()[0])), h_ref_data.terms()[1])
    )

    assert isinstance(h, qml.ops.Sum)

    wire_order = h_ref_data.wires
    assert pnp.allclose(
        qml.matrix(h, wire_order=wire_order),
        qml.matrix(h_ref_data, wire_order=wire_order),
    )


@pytest.mark.parametrize(
    "use_jax",
    [
        (False),
        pytest.param(True, marks=pytest.mark.jax),
    ],
)
def test_diff_hamiltonian_active_space(use_jax):
    r"""Test that diff_hamiltonian works when an active space is defined."""

    symbols = ["H", "H", "H"]
    geometry = pnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    if use_jax:
        geometry = qml.math.array(geometry, like="jax")

    mol = qchem.Molecule(symbols, geometry, charge=1)
    args = [geometry, mol.coeff, mol.alpha] if use_jax else [geometry]

    h = qchem.diff_hamiltonian(mol, core=[0], active=[1, 2])(*args)

    assert isinstance(h, qml.ops.Sum)


@pytest.mark.parametrize(
    ("symbols", "geometry", "core", "active", "charge"),
    [
        (
            ["H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            None,
            None,
            0,
        ),
        (
            ["H", "H", "H"],
            pnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 0.0]]),
            [0],
            [1, 2],
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_jax",
    [
        (False),
        pytest.param(True, marks=pytest.mark.jax),
    ],
)
def test_diff_hamiltonian_wire_order(symbols, geometry, core, active, charge, use_jax):
    r"""Test that diff_hamiltonian has an ascending wire order."""
    if use_jax:
        geometry = qml.math.array(geometry, like="jax")

    mol = qchem.Molecule(symbols, geometry, charge)
    args = [geometry, mol.coeff, mol.alpha] if use_jax else [geometry]

    h = qchem.diff_hamiltonian(mol, core=core, active=active)(*args)

    assert h.wires.tolist() == sorted(h.wires.tolist())


def test_gradient_expvalH():
    r"""Test that the gradient of expval(H) computed with ``qml.grad`` is equal to the value
    obtained with the finite difference method."""
    symbols = ["H", "H"]
    geometry = (
        pnp.array([[0.0, 0.0, -0.3674625962], [0.0, 0.0, 0.3674625962]], requires_grad=False)
        / 0.529177210903
    )
    alpha = pnp.array(
        [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=True,
    )

    mol = qchem.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    dev = qml.device("default.qubit", wires=4)

    def energy(mol):
        @qml.qnode(dev)
        def circuit(*args):
            qml.PauliX(0)
            qml.PauliX(1)
            qml.DoubleExcitation(0.22350048111151138, wires=[0, 1, 2, 3])
            h_qubit = qchem.diff_hamiltonian(mol)(*args)
            return qml.expval(h_qubit)

        return circuit

    grad_qml = qml.grad(energy(mol), argnum=0)(*args)

    alpha_1 = pnp.array(
        [[3.42515091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=False,
    )  # alpha[0][0] -= 0.0001

    alpha_2 = pnp.array(
        [[3.42535091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=False,
    )  # alpha[0][0] += 0.0001

    e_1 = energy(mol)(*[alpha_1])
    e_2 = energy(mol)(*[alpha_2])

    grad_finitediff = (e_2 - e_1) / 0.0002

    assert pnp.allclose(grad_qml[0][0], grad_finitediff)


@pytest.mark.jax
class TestJax:
    def test_gradient_jax_array(self):
        r"""Test that the gradient of expval(H) computed with ``jax.grad`` is equal to the value
        obtained with the finite difference method when using ``argnum`` and jax."""
        import jax

        symbols = ["H", "H"]
        geometry = (
            qml.math.array([[0.0, 0.0, -0.3674625962], [0.0, 0.0, 0.3674625962]], like="jax")
            / 0.529177210903
        )
        alpha = qml.math.array(
            [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            like="jax",
        )
        mol = qchem.Molecule(symbols, geometry, alpha=alpha)
        args = [geometry, mol.coeff, mol.alpha]
        dev = qml.device("default.qubit", wires=4)

        def energy():
            @qml.qnode(dev, interface="jax")
            def circuit(*args):
                qml.PauliX(0)
                qml.PauliX(1)
                qml.DoubleExcitation(0.22350048111151138, wires=[0, 1, 2, 3])
                mol = qml.qchem.Molecule(symbols, geometry, alpha=args[2], coeff=args[1])
                h_qubit = qchem.diff_hamiltonian(mol)(*args)
                return qml.expval(h_qubit)

            return circuit

        grad_jax = jax.grad(energy(), argnums=2)(*args)

        # Finite Differences
        alpha_1 = qml.math.array(
            [[3.42515091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            like="jax",
        )  # alpha[0][0] -= 0.0001

        alpha_2 = qml.math.array(
            [[3.42535091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            like="jax",
        )  # alpha[0][0] += 0.0001

        args_1 = [geometry, mol.coeff, alpha_1]
        args_2 = [geometry, mol.coeff, alpha_2]
        e_1 = energy()(*args_1)
        e_2 = energy()(*args_2)

        grad_finitediff = (e_2 - e_1) / 0.0002

        assert qml.math.allclose(grad_jax[0][0], grad_finitediff, rtol=1e-02)
