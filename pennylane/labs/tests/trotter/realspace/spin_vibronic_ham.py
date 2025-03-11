""" "
Spin-vibronic Hamiltonian details as specified in the SI of

T. Northey, T.J. Penfold,
The intersystem crossing mechanism of an ultrapure blue organoboron emitter,
Organic Electronics,
Volume 59,
2018,
Pages 45-48,
"""

import numpy as np

# pylint: disable=too-many-statements


def get_coeffs():
    """Get the frequences and Taylor coefficients of the spin-vibronic Hamiltonian"""

    Psis = ["S1", "S2", "S3", "S4", "T1", "T2"]
    Psis = {k: i for i, k in enumerate(Psis)}
    Qs = ["1", "2", "3", "7", "11", "16", "21", "22", "23", "24"]
    Qs = {k: i for i, k in enumerate(Qs)}

    n_states = len(Psis)
    n_modes = len(Qs)

    n_blocks = n_states

    # frequencies (eV)
    omegas = np.array(
        [
            0.00297,
            0.00306,
            0.00571,
            0.00970,
            0.02257,
            0.03193,
            0.04514,
            0.05004,
            0.05107,
            0.05240,
        ]
    )

    # Vertical excitation energies
    equilibrium_energies = np.array([3.33, 3.86, 3.90, 3.95, 2.74, 3.35])

    # Vertical excitation energies offsetted
    offset = (
        np.min(equilibrium_energies)
        + (np.max(equilibrium_energies) - np.min(equilibrium_energies)) / 2
    )
    equilibrium_energies_offsetted = equilibrium_energies - offset

    lambdas = np.zeros((n_blocks, n_blocks))
    for i in range(n_states):
        lambdas[i, i] = equilibrium_energies_offsetted[i]

    # On-diagonal linear couplings (eV)
    kappas = np.zeros((n_blocks, n_modes))

    # Off-diagonal linear couplings (eV)
    alphas = np.zeros((n_blocks, n_blocks, n_modes))

    # On-diagonal quadratic couplings (eV)
    # Only Q_i^2 included, no Q_i * Q_j.
    gammas = np.zeros((n_blocks, n_modes))

    # Populating coupling arrays:

    kappas[Psis["S1"], Qs["11"]] = -0.02450
    kappas[Psis["S2"], Qs["11"]] = -0.02770
    kappas[Psis["S3"], Qs["11"]] = -0.03170
    kappas[Psis["S4"], Qs["11"]] = -0.02400
    kappas[Psis["S1"], Qs["2"]] = 0.00370
    kappas[Psis["S2"], Qs["2"]] = 0.00400
    kappas[Psis["S3"], Qs["2"]] = -0.00570
    kappas[Psis["S4"], Qs["2"]] = 0.01360
    kappas[Psis["S1"], Qs["7"]] = 0.00930
    kappas[Psis["S2"], Qs["7"]] = 0.00800
    kappas[Psis["S3"], Qs["7"]] = 0.01360
    kappas[Psis["S4"], Qs["7"]] = 0.00530
    kappas[Psis["S1"], Qs["16"]] = -0.00120
    kappas[Psis["S2"], Qs["16"]] = 0.00520
    kappas[Psis["S3"], Qs["16"]] = 0.00220
    kappas[Psis["S4"], Qs["16"]] = 0.01070
    kappas[Psis["S1"], Qs["22"]] = 0.01470
    kappas[Psis["S2"], Qs["22"]] = 0.00650
    kappas[Psis["S3"], Qs["22"]] = 0.00180
    kappas[Psis["S4"], Qs["22"]] = 0.00330
    kappas[Psis["S1"], Qs["24"]] = 0.00770
    kappas[Psis["S2"], Qs["24"]] = -0.01530
    kappas[Psis["S3"], Qs["24"]] = -0.01750
    kappas[Psis["S4"], Qs["24"]] = -0.01610
    kappas[Psis["T1"], Qs["11"]] = -0.02200
    kappas[Psis["T2"], Qs["11"]] = -0.02010
    kappas[Psis["T1"], Qs["2"]] = 0.00160
    kappas[Psis["T2"], Qs["2"]] = 0.00110
    kappas[Psis["T1"], Qs["7"]] = 0.00630
    kappas[Psis["T2"], Qs["7"]] = 0.00390
    kappas[Psis["T1"], Qs["16"]] = 0.00090
    kappas[Psis["T2"], Qs["16"]] = -0.00529
    kappas[Psis["T1"], Qs["22"]] = 0.00733
    kappas[Psis["T2"], Qs["22"]] = 0.01682
    kappas[Psis["T1"], Qs["24"]] = 0.00781
    kappas[Psis["T2"], Qs["24"]] = 0.02151

    alphas[Psis["S3"], Psis["S4"], Qs["11"]] = 0.00081
    alphas[Psis["S1"], Psis["S2"], Qs["1"]] = 0.01832
    alphas[Psis["S2"], Psis["S3"], Qs["1"]] = 0.00730
    alphas[Psis["S2"], Psis["S4"], Qs["1"]] = 0.00147
    alphas[Psis["S3"], Psis["S4"], Qs["2"]] = 0.00130
    alphas[Psis["S2"], Psis["S3"], Qs["3"]] = 0.00067
    alphas[Psis["S1"], Psis["S3"], Qs["16"]] = 0.01905
    alphas[Psis["S1"], Psis["S4"], Qs["16"]] = 0.00464
    alphas[Psis["S3"], Psis["S4"], Qs["16"]] = -0.00208
    alphas[Psis["S1"], Psis["S2"], Qs["21"]] = -0.01896
    alphas[Psis["S2"], Psis["S3"], Qs["21"]] = 0.01533
    alphas[Psis["S2"], Psis["S4"], Qs["21"]] = 0.01696
    alphas[Psis["S1"], Psis["S3"], Qs["22"]] = -0.00050
    alphas[Psis["S1"], Psis["S4"], Qs["22"]] = 0.00860
    alphas[Psis["S3"], Psis["S4"], Qs["22"]] = 0.01533
    alphas[Psis["S1"], Psis["S2"], Qs["23"]] = 0.03424
    alphas[Psis["S2"], Psis["S4"], Qs["23"]] = -0.00406
    alphas[Psis["S1"], Psis["S3"], Qs["24"]] = -0.01030
    alphas[Psis["S1"], Psis["S4"], Qs["24"]] = -0.01099
    alphas[Psis["S3"], Psis["S4"], Qs["24"]] = 0.00248
    alphas[Psis["T1"], Psis["T2"], Qs["1"]] = 0.01681
    alphas[Psis["T1"], Psis["T2"], Qs["3"]] = 0.03128
    alphas[Psis["T1"], Psis["T2"], Qs["21"]] = -0.00260

    gammas[Psis["S1"], Qs["11"]] = 0.00450
    gammas[Psis["S2"], Qs["11"]] = 0.00430
    gammas[Psis["S3"], Qs["11"]] = 0.00500
    gammas[Psis["S4"], Qs["11"]] = 0.00460
    gammas[Psis["S1"], Qs["1"]] = 0.04660
    gammas[Psis["S2"], Qs["1"]] = 0.02830
    gammas[Psis["S3"], Qs["1"]] = 0.00470
    gammas[Psis["S4"], Qs["1"]] = 0.03780
    gammas[Psis["S1"], Qs["2"]] = 0.02280
    gammas[Psis["S2"], Qs["2"]] = 0.01830
    gammas[Psis["S3"], Qs["2"]] = 0.02040
    gammas[Psis["S4"], Qs["2"]] = 0.01990
    gammas[Psis["S1"], Qs["3"]] = 0.05130
    gammas[Psis["S2"], Qs["3"]] = 0.04150
    gammas[Psis["S3"], Qs["3"]] = 0.04090
    gammas[Psis["S4"], Qs["3"]] = 0.05230
    gammas[Psis["S1"], Qs["7"]] = 0.01430
    gammas[Psis["S2"], Qs["7"]] = 0.01470
    gammas[Psis["S3"], Qs["7"]] = 0.01450
    gammas[Psis["S4"], Qs["7"]] = 0.01440
    gammas[Psis["S1"], Qs["16"]] = 0.00450
    gammas[Psis["S2"], Qs["16"]] = 0.00560
    gammas[Psis["S3"], Qs["16"]] = 0.00490
    gammas[Psis["S4"], Qs["16"]] = 0.00450
    gammas[Psis["S2"], Qs["21"]] = -0.00250
    gammas[Psis["S3"], Qs["21"]] = 0.00020
    gammas[Psis["S4"], Qs["21"]] = -0.00020
    gammas[Psis["S1"], Qs["22"]] = -0.00090
    gammas[Psis["S2"], Qs["22"]] = -0.00080
    gammas[Psis["S3"], Qs["22"]] = -0.00090
    gammas[Psis["S4"], Qs["22"]] = -0.00090
    gammas[Psis["S1"], Qs["23"]] = -0.00110
    gammas[Psis["S2"], Qs["23"]] = -0.00200
    gammas[Psis["S3"], Qs["23"]] = 0.00110
    gammas[Psis["S4"], Qs["23"]] = 0.00110
    gammas[Psis["T1"], Qs["11"]] = 0.00035
    gammas[Psis["T2"], Qs["11"]] = -0.00086
    gammas[Psis["T1"], Qs["1"]] = 0.00318
    gammas[Psis["T2"], Qs["1"]] = 0.00216
    gammas[Psis["T1"], Qs["2"]] = 0.00068
    gammas[Psis["T2"], Qs["2"]] = 0.00061
    gammas[Psis["T1"], Qs["3"]] = 0.00926
    gammas[Psis["T2"], Qs["3"]] = 0.00580
    gammas[Psis["T1"], Qs["7"]] = 0.00045
    gammas[Psis["T2"], Qs["7"]] = 0.00028
    gammas[Psis["T1"], Qs["16"]] = 0.00011
    gammas[Psis["T2"], Qs["16"]] = 0.00002
    gammas[Psis["T1"], Qs["21"]] = -0.00066
    gammas[Psis["T2"], Qs["21"]] = -0.00160
    gammas[Psis["T1"], Qs["22"]] = -0.00109
    gammas[Psis["T2"], Qs["22"]] = -0.00160
    gammas[Psis["T1"], Qs["23"]] = -0.00590
    gammas[Psis["T2"], Qs["23"]] = -0.00611
    gammas[Psis["T1"], Qs["24"]] = -0.00060
    gammas[Psis["T2"], Qs["23"]] = -0.00134

    betas = np.zeros((n_blocks, n_blocks, n_modes, n_modes))

    for i in range(n_states):
        for j in range(n_modes):
            betas[i][i][j][j] = gammas[i][j]

    for j in range(n_states):
        alphas[j][j] = kappas[j]
        for k in range(j):
            alphas[j][k] = alphas[k][j]

    return omegas, [lambdas, alphas, betas]
