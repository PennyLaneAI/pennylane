from functools import singledispatch
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.pauli import PauliSentence, PauliWord
from bosonic import BoseWord, BoseSentence


def vib_obs(one, modes=None, modals=None, two=None, three=None, cutoff=1e-5):
    r"""Build a vibrational observable in the Christiansen form (C-form) and map it
    to the Pauli basis

    Args:
        one (array): 3D array with one-body matrix elements
        modes (int): number of vibrational modes, detects from 'one' if none is provided
        modals (array): 1D array with the number of allowed vibrational modals for each mode, detects from 'one' if none is provided
        two (array): 6D array with two-body matrix elements
        three (array): 9D array with three-body matrix elements

    Returns:
        tuple[int, Union[PauliSentence, Operator]]: the number of qubits and a linear combination of qubit operators
    """
    if modes is None:
        modes = np.shape(one)[0]

    if modals is None:
        imax = np.shape(one)[1]
        modals = imax * np.ones(modes, dtype=int)

    idx = {}  # dictionary mapping the tuple (l,n) to an index in the qubit register
    counter = 0
    for l in range(modes):
        for n in range(modals[l]):
            key = (l, n)
            idx[key] = counter
            counter += 1

    obs = {}  # second-quantized Hamiltonian

    # one-body terms
    for l in range(modes):
        for k_l in range(modals[l]):
            for h_l in range(modals[l]):
                (i0, i1) = ((l, k_l), (l, h_l))
                w = BoseWord({(0, idx[i0]): "+", (1, idx[i1]): "-"})
                obs[w] = one[l, k_l, h_l]

    # two-body terms
    if not two is None:
        for l in range(modes):
            for m in [p for p in range(modes) if p != l]:
                for k_l in range(modals[l]):
                    for h_l in range(modals[l]):
                        for k_m in range(modals[m]):
                            for h_m in range(modals[m]):
                                (i0, i1, i2, i3) = (
                                    (l, k_l),
                                    (m, k_m),
                                    (l, h_l),
                                    (m, h_m),
                                )
                                w = BoseWord(
                                    {
                                        (0, idx[i0]): "+",
                                        (1, idx[i1]): "+",
                                        (2, idx[i2]): "-",
                                        (3, idx[i3]): "-",
                                    }
                                )
                                obs[w] = two[l, m, k_l, k_m, h_l, h_m]

    # three-body terms
    if not three is None:
        for l in range(modes):
            for m in [p for p in range(modes) if p != l]:
                for n in [p for p in range(modes) if p != l and p != m]:
                    for k_l in range(modals[l]):
                        for h_l in range(modals[l]):
                            for k_m in range(modals[m]):
                                for h_m in range(modals[m]):
                                    for k_n in range(modals[n]):
                                        for h_n in range(modals[n]):
                                            (i0, i1, i2, i3, i4, i5) = (
                                                (l, k_l),
                                                (m, k_m),
                                                (n, k_n),
                                                (l, h_l),
                                                (m, h_m),
                                                (n, h_n),
                                            )
                                            w = BoseWord(
                                                {
                                                    (0, idx[i0]): "+",
                                                    (1, idx[i1]): "+",
                                                    (2, idx[i2]): "+",
                                                    (3, idx[i3]): "-",
                                                    (4, idx[i4]): "-",
                                                    (5, idx[i5]): "-",
                                                }
                                            )
                                            obs[w] = three[l, m, n, k_l, k_m, k_n, h_l, h_m, h_n]

    obs_sq = BoseSentence(obs)
    #    obs_pl = jordan_wigner(obs_sq, ps=True)
    #    obs_pl.simplify(tol=cutoff)

    return (np.sum(modals), obs_pl)


def vib_from_modes(H_arr, modals=None, cutoff=1e-5):
    """
    Returns vibrational observable as PennyLane qubit operator object

    Arguments:
        - H_arr: array containing n-mode expansion of Hamiltonian as:
            H_arr = [H1, H2, ...] for M the number of modes and imax the maximum basis size over all modes:
                H1 -> [M,imax,imax]
                H2 -> [M,M,imax,imax,imax,imax]
                    .
                    .
                    .
        - modals = [N1,N2,...,Nm] containing number of modals Nk for each mode k. Defaults to [imax,imax,...,imax] if none is provided
    """
    n = len(H_arr)
    if n > 3:
        raise ValueError(
            "Trying to build Hamiltonian operator for n={}>3 modes, not implemented!".format(n)
        )

    if n == 1:
        return vib_obs(H_arr[0], modals=modals, cutoff=cutoff)
    elif n == 2:
        return vib_obs(H_arr[0], modals=modals, two=H_arr[1], cutoff=cutoff)
    elif n == 3:
        return vib_obs(H_arr[0], modals=modals, two=H_arr[1], three=H_arr[2], cutoff=cutoff)
