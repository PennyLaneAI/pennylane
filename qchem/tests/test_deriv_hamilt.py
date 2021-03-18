import numpy as np
import pytest

import pennylane as qml


coeffs = []
coeffs.append(
    [
        0.277704449278815,
        -0.0017650046150397003,
        -0.0017650046150397003,
        -0.12236476421393916,
        -0.12236476421393916,
        0.0016333663759190292,
        -0.0016333663759190292,
        -0.0016333663759190292,
        0.0016333663759190292,
        0.006294159683373213,
        0.007927526059292589,
        0.007927526059292589,
        0.006294159683373213,
        0.008903333057769247,
    ]
)


coeffs.append(
    [
        0.09228770579170487,
        0.0024877619569541043,
        0.0024877619569541043,
        -0.04158910777451452,
        -0.04158910777451452,
        0.0010910266418804812,
        -0.0010910266418804812,
        -0.0010910266418804812,
        0.0010910266418804812,
        -0.005093392871888258,
        -0.004002366230010379,
        -0.004002366230010379,
        -0.005093392871888258,
        -0.004008745811959202,
    ]
)


coeffs.append([])


coeffs.append(
    [
        -0.04992170173352406,
        -0.0007449368397277611,
        -0.0007449368397249856,
        0.022335251763061503,
        0.022335251763055952,
        -0.00048133012454602137,
        0.00048133012454602137,
        0.00048133012454602137,
        -0.00048133012454602137,
        0.001372950451092314,
        0.0008916203265485478,
        0.0008916203265485478,
        0.001372950451092314,
        0.0007544964083811001,
    ]
)


coeffs.append(
    [
        -0.2760200644090105,
        -0.0042126833568884026,
        -0.004212683356880076,
        0.12339079287995991,
        0.12339079287995158,
        -0.0026899511866487064,
        0.0026899511866487064,
        0.0026899511866487064,
        -0.0026899511866487064,
        0.007669908781263168,
        0.004979957594614115,
        0.004979957594614115,
        0.007669908781263168,
        0.004741829225737848,
    ]
)


coeffs.append([])


coeffs.append(
    [
        -0.22777344340170202,
        0.002520550384513265,
        0.002520550384513265,
        0.10001562018918109,
        0.10001562018918664,
        -0.0011519091794620281,
        0.0011519091794620281,
        0.0011519091794620281,
        -0.0011519091794620281,
        -0.007675113089958274,
        -0.008827022269417006,
        -0.008827022269417006,
        -0.007675113089958274,
        -0.00965003120832475,
    ]
)


coeffs.append(
    [
        0.18372596167068878,
        0.0017153684828685734,
        0.0017153684828796756,
        -0.08178978964119843,
        -0.0817897896412012,
        0.0015987756601037323,
        -0.0015987756601037323,
        -0.0015987756601037323,
        0.0015987756601037323,
        -0.0025694313453117346,
        -0.0009706556852062675,
        -0.0009706556852062675,
        -0.0025694313453117346,
        -0.0007414855944548604,
    ]
)


coeffs.append([])

ops = [
    qml.Identity(wires=[0]),
    qml.PauliZ(wires=[0]),
    qml.PauliZ(wires=[1]),
    qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[3]),
    qml.PauliY(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
]


@pytest.mark.parametrize(
    ("i", "coeffs", "ops"),
    [
        (0, coeffs[0], ops),
        (1, coeffs[1], ops),
        (2, [], []),
        (3, coeffs[3], ops),
        (4, coeffs[4], ops),
        (5, [], []),
        (6, coeffs[6], ops),
        (7, coeffs[7], ops),
        (8, [], []),
    ],
)
def test_finit_diff_hamilt(i, coeffs, ops, tol, tmpdir):
    r"""Tests the correctness of the derivative of a molecular Hamiltonian calculated
    by the 'finite_diff' function."""

    # parametrized Hamiltonian of the water molecule
    def hamilt(x):
        return qml.qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
        )[0]

    x = np.array(
        [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
    )

    deriv = qml.finite_diff(hamilt, N=1, argnum=0, idx=[i])(x)

    assert len(deriv) == len(x)

    calc_coeffs = np.array(deriv[i].coeffs)
    exp_coeffs = np.array(coeffs)

    assert np.allclose(calc_coeffs, exp_coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv[i].ops, ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv[i].ops, ops))


@pytest.mark.parametrize(
    ("idx"),
    [
        ([0]),
        ([0, 5]),
    ],
)
def test_grad_components(idx, tmpdir):
    r"""Tests that the only the specified components of the gradients are calculated
    by the 'finite_diff' function."""

    # parametrized Hamiltonian of the water molecule
    def hamilt(x):
        return qml.qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
        )[0]

    x = np.array(
        [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
    )

    deriv = qml.finite_diff(hamilt, N=1, argnum=0, idx=idx)(x)

    for i, _deriv in enumerate(deriv):
        if i in idx:
            assert isinstance(_deriv, qml.vqe.Hamiltonian)
        else:
            assert _deriv == 0


coeffs_00 = [
    0.3477257797612765,
    -0.030524309531099192,
    -0.030524309529433857,
    -0.1771190461385963,
    -0.1771190461385963,
    -0.03147624232327506,
    0.03147624232327506,
    0.03147624232327506,
    -0.03147624232327506,
    0.045618439534766964,
    0.014142197210798013,
    0.014142197210798013,
    0.045618439534766964,
    0.6647460733938404,
]
coeffs_34 = [
    0.09246463918088921,
    0.02856614438190297,
    0.02856614438190297,
    0.10700948302716506,
    0.10700948302744262,
    0.017922142783628747,
    -0.017922142783628747,
    -0.017922142783628747,
    0.017922142783628747,
    -0.028381602737159728,
    -0.010459459953426897,
    -0.010459459953426897,
    -0.028381602737159728,
    -0.37581963724941936,
]


@pytest.mark.parametrize(
    ("idx", "coeffs", "ops"),
    [
        ([0, 0], coeffs_00, ops),
        ([3, 4], coeffs_34, ops),
        ([4, 3], coeffs_34, ops),
        ([1, 2], [], []),
    ],
)
def test_second_derivative_hamilt(idx, coeffs, ops, tol, tmpdir):
    r"""Tests the correctness of the second-order derivative of a molecular Hamiltonian calculated
    by the 'finite_diff' function."""

    # parametrized Hamiltonian of the water molecule
    def hamilt(x):
        return qml.qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
        )[0]

    x = np.array(
        [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
    )

    deriv2 = qml.finite_diff(hamilt, N=2, argnum=0, idx=idx)(x)

    calc_coeffs = np.array(deriv2.coeffs)
    exp_coeffs = np.array(coeffs)

    assert np.allclose(calc_coeffs, exp_coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv2.ops, ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv2.ops, ops))
