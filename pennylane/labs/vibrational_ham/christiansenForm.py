import numpy as np
from scipy.special import factorial
import itertools
from tqdm.auto import tqdm

from mpi4py import MPI

COMM = MPI.COMM_WORLD

au_to_cm = 219475


def get_cform_kin(freqs, nbos):
    # action of kinetic energy operator for m=1,...,M modes with frequencies freqs[m]
    nmodes = len(freqs)
    all_mode_combos = []
    for aa in range(nmodes):
        all_mode_combos.append([aa])
    all_bos_combos = list(itertools.product(range(nbos), range(nbos)))

    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_K_mat = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii] in all_mode_combos:

        ii = int(ii)

        mm = 0
        for [ki, hi] in boscombos_on_rank:
            m_const = freqs[ii] / 4
            ind = nn * len(boscombos_on_rank) + mm
            if ki == hi:
                local_K_mat[ind] += m_const * (2 * ki + 1)
            if ki == hi + 2:
                local_K_mat[ind] -= m_const * np.sqrt((hi + 2) * (hi + 1))
            if ki == hi - 2:
                local_K_mat[ind] -= m_const * np.sqrt((hi - 1) * hi)
            mm += 1
        nn += 1
    return local_K_mat


def get_cform_localized_kin_twobody(pes_gen, nbos):
    # action of kinetic energy operator for m=1,...,M localized modes with frequencies freqs[m]
    # note that normal modes make this term zero, only appears for non-normal displacements
    nmodes = len(pes_gen.freqs)

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            all_mode_combos.append([aa, bb])
    all_bos_combos = list(itertools.product(range(nbos), range(nbos), range(nbos), range(nbos)))

    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_kin_cform_twobody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii, jj] in tqdm(all_mode_combos, desc="C-form two-body: mode combinations"):

        ii, jj = int(ii), int(jj)
        # skip through the things that are not needed
        if jj >= ii:
            nn += 1
            continue

        Usum = np.einsum("i,i->", pes_gen.uloc[:, ii], pes_gen.uloc[:, jj])
        m_const = Usum * np.sqrt(pes_gen.freqs[ii] * pes_gen.freqs[jj]) / 4

        mm = 0
        for [ki, kj, hi, hj] in tqdm(boscombos_on_rank, desc="C-form two-body: boson combinations"):
            ind = nn * len(boscombos_on_rank) + mm
            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            if ki == hi + 1 and kj == hj + 1:
                local_kin_cform_twobody[ind] -= m_const * np.sqrt((hi + 1) * (hj + 1))
            if ki == hi + 1 and kj == hj - 1:
                local_kin_cform_twobody[ind] += m_const * np.sqrt((hi + 1) * hj)
            if ki == hi - 1 and kj == hj + 1:
                local_kin_cform_twobody[ind] += m_const * np.sqrt(hi * (hj + 1))
            if ki == hi - 1 and kj == hj - 1:
                local_kin_cform_twobody[ind] -= m_const * np.sqrt(hi * hj)

            mm += 1
        nn += 1

    return local_kin_cform_twobody


def get_cform_onebody(pes_gen, nbos):
    """
    Use the one-body potential energy surface and evaluate the modal integrals
    to find the Christiansen form Hamiltonian.
    """

    nmodes = len(pes_gen.freqs)
    all_mode_combos = []
    for aa in range(nmodes):
        all_mode_combos.append([aa])
    all_bos_combos = list(itertools.product(range(nbos), range(nbos)))

    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_ham_cform_onebody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii] in tqdm(all_mode_combos, desc="C-form one-body: mode combinations"):

        ii = int(ii)

        mm = 0
        for [ki, hi] in tqdm(boscombos_on_rank, desc="C-form one-body: boson combinations"):

            sqrt = (2 ** (ki + hi) * factorial(ki) * factorial(hi) * np.pi) ** (-0.5)
            order_k = np.zeros(nbos)
            order_k[ki] = 1.0
            order_h = np.zeros(nbos)
            order_h[hi] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_k, [-1, 1])(pes_gen.gauss_grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_h, [-1, 1])(pes_gen.gauss_grid)
            quadrature = np.sum(
                pes_gen.gauss_weights * pes_gen.pes_onebody[ii, :] * hermite_ki * hermite_hi
            )
            full_coeff = sqrt * quadrature  # * 219475 for converting into cm^-1
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_onebody[ind] += full_coeff
            mm += 1
        nn += 1
    return local_ham_cform_onebody + get_cform_kin(pes_gen.freqs, nbos)


def get_dipole_cform_onebody(dipole_onebody, nbos, gauss_grid, gauss_weights):
    """
    Use the one-body dipole functions and evaluate the modal integrals
    to find the Christiansen form of the dipole term.
    """

    nmodes = dipole_onebody.shape[0]
    all_mode_combos = []
    for aa in range(nmodes):
        all_mode_combos.append([aa])
    all_bos_combos = list(itertools.product(range(nbos), range(nbos)))

    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_dipole_cform_onebody = np.zeros((len(all_mode_combos) * chunksize, 3))

    nn = 0
    for [ii] in tqdm(all_mode_combos, desc="C-form dipole one-body: mode combinations"):

        ii = int(ii)

        mm = 0
        for [ki, hi] in tqdm(boscombos_on_rank, desc="C-form dipole one-body: boson combinations"):

            ki, hi = int(ki), int(hi)
            sqrt = (2 ** (ki + hi) * factorial(ki) * factorial(hi) * np.pi) ** (-0.5)
            order_k = np.zeros(nbos)
            order_k[ki] = 1.0
            order_h = np.zeros(nbos)
            order_h[hi] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_k, [-1, 1])(gauss_grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_h, [-1, 1])(gauss_grid)
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.sum(
                    gauss_weights * dipole_onebody[ii, :, alpha] * hermite_ki * hermite_hi
                )
                full_coeff = sqrt * quadrature  # * 219475 for converting into cm^-1
                local_dipole_cform_onebody[ind, alpha] += full_coeff
            mm += 1
        nn += 1

    return local_dipole_cform_onebody


def get_cform_twobody(pes_gen, nbos):
    """
    Use the two-body potential energy surface and evaluate the modal integrals
    to find the Christiansen form Hamiltonian.
    """

    nmodes = pes_gen.pes_twobody.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            all_mode_combos.append([aa, bb])
    all_bos_combos = list(itertools.product(range(nbos), range(nbos), range(nbos), range(nbos)))

    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_ham_cform_twobody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii, jj] in tqdm(all_mode_combos, desc="C-form two-body: mode combinations"):

        ii, jj = int(ii), int(jj)
        # skip through the things that are not needed
        if jj >= ii:
            nn += 1
            continue

        mm = 0
        for [ki, kj, hi, hj] in tqdm(boscombos_on_rank, desc="C-form two-body: boson combinations"):

            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            sqrt = (
                2 ** (ki + kj + hi + hj)
                * factorial(ki)
                * factorial(kj)
                * factorial(hi)
                * factorial(hj)
            ) ** (-0.5) / np.pi
            order_ki = np.zeros(nbos)
            order_ki[ki] = 1.0
            order_kj = np.zeros(nbos)
            order_kj[kj] = 1.0
            order_hi = np.zeros(nbos)
            order_hi[hi] = 1.0
            order_hj = np.zeros(nbos)
            order_hj[hj] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_ki, [-1, 1])(pes_gen.gauss_grid)
            hermite_kj = np.polynomial.hermite.Hermite(order_kj, [-1, 1])(pes_gen.gauss_grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_hi, [-1, 1])(pes_gen.gauss_grid)
            hermite_hj = np.polynomial.hermite.Hermite(order_hj, [-1, 1])(pes_gen.gauss_grid)
            quadrature = np.einsum(
                "a,b,a,b,ab,a,b->",
                pes_gen.gauss_weights,
                pes_gen.gauss_weights,
                hermite_ki,
                hermite_kj,
                pes_gen.pes_twobody[ii, jj, :, :],
                hermite_hi,
                hermite_hj,
            )
            full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_twobody[ind] += full_coeff
            mm += 1
        nn += 1
    return local_ham_cform_twobody


def get_dipole_cform_twobody(dipole_twobody, nbos, gauss_grid, gauss_weights):
    """
    Use the two-body potential energy surface and evaluate the modal integrals
    to find the Christiansen form of two-mode dipole.
    """

    nmodes = dipole_twobody.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            all_mode_combos.append([aa, bb])
    all_bos_combos = list(itertools.product(range(nbos), range(nbos), range(nbos), range(nbos)))

    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_dipole_cform_twobody = np.zeros((len(all_mode_combos) * chunksize, 3))

    nn = 0
    for [ii, jj] in tqdm(all_mode_combos, desc="C-form dipole two-body: mode combinations"):

        ii, jj = int(ii), int(jj)
        # skip through the things that are not needed
        if jj >= ii:
            nn += 1
            continue

        mm = 0
        for [ki, kj, hi, hj] in tqdm(
            boscombos_on_rank, desc="C-form dipole two-body: boson combinations"
        ):

            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            sqrt = (
                2 ** (ki + kj + hi + hj)
                * factorial(ki)
                * factorial(kj)
                * factorial(hi)
                * factorial(hj)
            ) ** (-0.5) / np.pi
            order_ki = np.zeros(nbos)
            order_ki[ki] = 1.0
            order_kj = np.zeros(nbos)
            order_kj[kj] = 1.0
            order_hi = np.zeros(nbos)
            order_hi[hi] = 1.0
            order_hj = np.zeros(nbos)
            order_hj[hj] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_ki, [-1, 1])(pes_gen.gauss_grid)
            hermite_kj = np.polynomial.hermite.Hermite(order_kj, [-1, 1])(pes_gen.gauss_grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_hi, [-1, 1])(pes_gen.gauss_grid)
            hermite_hj = np.polynomial.hermite.Hermite(order_hj, [-1, 1])(pes_gen.gauss_grid)
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.einsum(
                    "a,b,a,b,ab,a,b->",
                    pes_gen.gauss_weights,
                    pes_gen.gauss_weights,
                    hermite_ki,
                    hermite_kj,
                    dipole_twobody[ii, jj, :, :, alpha],
                    hermite_hi,
                    hermite_hj,
                )
                full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
                local_dipole_cform_twobody[ind, alpha] += full_coeff
            mm += 1
        nn += 1

    return local_dipole_cform_twobody


def get_cform_threebody(pes_gen, pes_threebody, nbos):
    """
    Use the three-body potential energy surface and evaluate the modal integrals
    to find the Christiansen form Hamiltonian.
    """

    nmodes = pes_threebody.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            for cc in range(nmodes):
                all_mode_combos.append([aa, bb, cc])
    all_bos_combos = list(
        itertools.product(
            range(nbos), range(nbos), range(nbos), range(nbos), range(nbos), range(nbos)
        )
    )
    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_ham_cform_threebody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii1, ii2, ii3] in tqdm(all_mode_combos, desc="C-form three-body: mode combinations"):

        ii1, ii2, ii3 = int(ii1), int(ii2), int(ii3)
        # skip the objects that are not needed
        if ii2 >= ii1 or ii3 >= ii2:
            nn += 1
            continue

        mm = 0
        for [k1, k2, k3, h1, h2, h3] in tqdm(
            boscombos_on_rank, desc="C-form three-body: boson combinations"
        ):

            k1, k2, k3, h1, h2, h3 = int(k1), int(k2), int(k3), int(h1), int(h2), int(h3)
            sqrt = (
                2 ** (k1 + k2 + k3 + h1 + h2 + h3)
                * factorial(k1)
                * factorial(k2)
                * factorial(k3)
                * factorial(h1)
                * factorial(h2)
                * factorial(h3)
            ) ** (-0.5) / (np.pi**1.5)
            order_k1 = np.zeros(nbos)
            order_k1[k1] = 1.0
            order_k2 = np.zeros(nbos)
            order_k2[k2] = 1.0
            order_k3 = np.zeros(nbos)
            order_k3[k3] = 1.0
            order_h1 = np.zeros(nbos)
            order_h1[h1] = 1.0
            order_h2 = np.zeros(nbos)
            order_h2[h2] = 1.0
            order_h3 = np.zeros(nbos)
            order_h3[h3] = 1.0
            hermite_k1 = np.polynomial.hermite.Hermite(order_k1, [-1, 1])(pes_gen.gauss_grid)
            hermite_k2 = np.polynomial.hermite.Hermite(order_k2, [-1, 1])(pes_gen.gauss_grid)
            hermite_k3 = np.polynomial.hermite.Hermite(order_k3, [-1, 1])(pes_gen.gauss_grid)
            hermite_h1 = np.polynomial.hermite.Hermite(order_h1, [-1, 1])(pes_gen.gauss_grid)
            hermite_h2 = np.polynomial.hermite.Hermite(order_h2, [-1, 1])(pes_gen.gauss_grid)
            hermite_h3 = np.polynomial.hermite.Hermite(order_h3, [-1, 1])(pes_gen.gauss_grid)

            quadrature = np.einsum(
                "a,b,c,a,b,c,abc,a,b,c->",
                pes_gen.gauss_weights,
                pes_gen.gauss_weights,
                pes_gen.gauss_weights,
                hermite_k1,
                hermite_k2,
                hermite_k3,
                pes_threebody[ii1, ii2, ii3, :, :, :],
                hermite_h1,
                hermite_h2,
                hermite_h3,
            )
            full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_threebody[ind] = full_coeff
            mm += 1
        nn += 1
    return local_ham_cform_threebody


def get_dipole_cform_threebody(dipole_threebody, nbos, gauss_grid, gauss_weights):
    """
    Use the three-body dipole surface and evaluate the modal integrals
    to find the Christiansen form dipole.
    """
    nmodes = dipole_threebody.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            for cc in range(nmodes):
                all_mode_combos.append([aa, bb, cc])
    all_bos_combos = list(
        itertools.product(
            range(nbos), range(nbos), range(nbos), range(nbos), range(nbos), range(nbos)
        )
    )
    rank = COMM.Get_rank()
    size = COMM.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_dipole_cform_threebody = np.zeros((len(all_mode_combos) * chunksize, 3))

    nn = 0
    for [ii1, ii2, ii3] in tqdm(
        all_mode_combos, desc="C-form dipole three-body: mode combinations"
    ):

        ii1, ii2, ii3 = int(ii1), int(ii2), int(ii3)
        # skip the objects that are not needed
        if ii2 >= ii1 or ii3 >= ii2:
            nn += 1
            continue

        mm = 0
        for [k1, k2, k3, h1, h2, h3] in tqdm(
            boscombos_on_rank, desc="C-form dipole three-body: boson combinations"
        ):

            k1, k2, k3, h1, h2, h3 = int(k1), int(k2), int(k3), int(h1), int(h2), int(h3)
            sqrt = (
                2 ** (k1 + k2 + k3 + h1 + h2 + h3)
                * factorial(k1)
                * factorial(k2)
                * factorial(k3)
                * factorial(h1)
                * factorial(h2)
                * factorial(h3)
            ) ** (-0.5) / (np.pi**1.5)
            order_k1 = np.zeros(nbos)
            order_k1[k1] = 1.0
            order_k2 = np.zeros(nbos)
            order_k2[k2] = 1.0
            order_k3 = np.zeros(nbos)
            order_k3[k3] = 1.0
            order_h1 = np.zeros(nbos)
            order_h1[h1] = 1.0
            order_h2 = np.zeros(nbos)
            order_h2[h2] = 1.0
            order_h3 = np.zeros(nbos)
            order_h3[h3] = 1.0
            hermite_k1 = np.polynomial.hermite.Hermite(order_k1, [-1, 1])(gauss_grid)
            hermite_k2 = np.polynomial.hermite.Hermite(order_k2, [-1, 1])(gauss_grid)
            hermite_k3 = np.polynomial.hermite.Hermite(order_k3, [-1, 1])(gauss_grid)
            hermite_h1 = np.polynomial.hermite.Hermite(order_h1, [-1, 1])(gauss_grid)
            hermite_h2 = np.polynomial.hermite.Hermite(order_h2, [-1, 1])(gauss_grid)
            hermite_h3 = np.polynomial.hermite.Hermite(order_h3, [-1, 1])(gauss_grid)
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.einsum(
                    "a,b,c,a,b,c,abc,a,b,c->",
                    gauss_weights,
                    gauss_weights,
                    gauss_weights,
                    hermite_k1,
                    hermite_k2,
                    hermite_k3,
                    dipole_threebody[ii1, ii2, ii3, :, :, :, alpha],
                    hermite_h1,
                    hermite_h2,
                    hermite_h3,
                )
                full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
                local_dipole_cform_threebody[ind, alpha] = full_coeff
            mm += 1
        nn += 1

    return local_dipole_cform_threebody


def get_christiansen_form(pes, nbos=16, do_cubic=True):
    r"""Generates the vibrational Hamiltonian in Christiansen form

    Args:
      pes: Build_pes object.
      nbos: number of modal basis functions per-mode
    """

    local_ham_cform_onebody = get_cform_onebody(pes, nbos)
    COMM.Barrier()
    print("onebody: ", local_ham_cform_onebody.shape)
    local_ham_cform_twobody = get_cform_twobody(pes, nbos)
    if pes.localize:
        local_ham_cform_twobody += get_cform_localized_kin_twobody(pes, nbos)
    COMM.Barrier()

    if do_cubic:
        pes_threebody = np.array(COMM.bcast(pes.pes_threebody, root=0))
        local_ham_cform_threebody = get_cform_threebody(pes, pes_threebody, nbos)
        COMM.Barrier()

    return local_ham_cform_onebody, local_ham_cform_twobody, local_ham_cform_threebody
