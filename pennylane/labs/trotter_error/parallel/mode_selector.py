from copy import deepcopy

import numpy as np


def mode_pnorm(couplings, p=1, omega=None, freq_usage="penalty"):

    cumulative = np.abs(np.array(couplings))
    norm = (sum(cumulative**p)) ** (1 / p)
    if omega is None:
        return norm

    if freq_usage == "penalty":
        print("true")
        print(norm)
        print(omega)
        return norm / omega
    elif freq_usage == "bonus":
        return omega * norm
    else:
        raise ValueError("freq_usage not recognized as valid.")


def rank_modes(omegas, couplings, ranking_type="C1N"):

    valid_types = ["C1N", "C2N", "F", "C1NDIVF", "C1NMULF", "C1N+F", "PT"]
    n_states = np.shape(couplings[0])[0]
    n_modes = len(omegas)

    if ranking_type.upper() not in valid_types:
        raise ValueError(
            f"Ranking type {ranking_type} not recognized. Must be one of: {valid_types}"
        )

    non_constant_couplings = [couplings[order] for order in couplings.keys() if order > 0]

    mode_interactions = {}
    mode_scores = {}
    for mode_idx in range(len(omegas)):
        interactions = []
        for coupling_arr in non_constant_couplings:
            if len(np.shape(coupling_arr)) == 3:  # linear term
                couplings_flat = coupling_arr[:, :, mode_idx].flatten()
                for coupling in couplings_flat:
                    interactions.append(coupling)
            elif len(np.shape(coupling_arr)) == 4:  # quadratic term
                couplings_flat = coupling_arr[:, :, mode_idx, :].flatten()
                for coupling in couplings_flat:
                    interactions.append(coupling)
        mode_interactions[mode_idx] = interactions

    for mode_idx in mode_interactions:
        if ranking_type.upper() == "C1N":
            mode_scores[mode_idx] = mode_pnorm(mode_interactions[mode_idx], p=1, omega=None)
        elif ranking_type.upper() == "C2N":
            mode_scores[mode_idx] = mode_pnorm(mode_interactions[mode_idx], p=2, omega=None)
        elif ranking_type.upper() == "F":
            mode_scores[mode_idx] = omegas[mode_idx]
        elif ranking_type.upper() == "C1NDIVF":
            mode_scores[mode_idx] = mode_pnorm(
                mode_interactions[mode_idx], p=1, omega=omegas[mode_idx], freq_usage="penalty"
            )
        elif ranking_type.upper() == "C1NMULF":
            mode_scores[mode_idx] = mode_pnorm(
                mode_interactions[mode_idx], p=1, omega=omegas[mode_idx], freq_usage="bonus"
            )
        elif ranking_type.upper() == "C1N+F":
            mode_scores[mode_idx] = mode_pnorm(mode_interactions[mode_idx], p=1) + abs(
                omegas[mode_idx]
            )

        elif ranking_type.upper() == "PT":
            # Looks at maximum "effective coupling" magnitude |lambda^(i,j)/(E_i - E_j - omega)|
            measures = []
            for I in range(n_states):
                for J in range(I, n_states):
                    measures.append(eff_coupling_strength(omegas, couplings, I, J, mode_idx))
            mode_scores[mode_idx] = max(measures)

    return mode_scores


def truncate_by_states(coupling_arrays, states_to_keep):

    truncated_coupling_arrays = {}

    m = np.shape(coupling_arrays[1])[-1]

    for idx, coupling_arr in coupling_arrays.items():
        dim = len(np.shape(coupling_arr))
        if dim == 2:  # 0th order in vib
            truncated_cpl_arr = coupling_arrays[idx][np.ix_(states_to_keep, states_to_keep)]
        elif dim == 3:  # 1st order in vib
            truncated_cpl_arr = coupling_arrays[idx][
                np.ix_(states_to_keep, states_to_keep, np.arange(m))
            ]
        elif dim == 4:  # 2nd order in vib
            truncated_cpl_arr = coupling_arrays[idx][
                np.ix_(states_to_keep, states_to_keep, np.arange(m), np.arange(m))
            ]
        else:
            raise ValueError(f"dimension {dim} not explicitly handled yet.")
        truncated_coupling_arrays[idx] = truncated_cpl_arr
    return truncated_coupling_arrays


def get_truncated_coupling_arrays(
    coupling_arrays, selected_modes, selected_states=None, only_diagonal=True
):

    # map the indices of the selected_modes to [0,1,.. m-1] and return the relevant coupling arrays.
    m = len(selected_modes)

    for idx, coupling_arr in coupling_arrays.items():
        if idx == 1:  # linear terms
            subcoupling_arr_lin = coupling_arr[:, :, selected_modes]  # Shape (N, N, M')
        elif idx == 2:  # quadratic term
            if only_diagonal:
                subcoupling_arr_quad = np.stack(
                    [coupling_arr[:, :, i, i] for i in selected_modes], axis=-1
                )
            else:
                subcoupling_arr_quad = coupling_arr[:, :, selected_modes, :][
                    :, :, :, selected_modes
                ]  # Shape (N, N, M', M')

    return [coupling_arrays[0], subcoupling_arr_lin, subcoupling_arr_quad]


def get_reduced_model(omegas, couplings, m_max, order_max=2, states=None, strategy=None):
    """
    `omegas` and `couplings` are the original omega and coupling tensors

    `m_max` is the number of modes to be included in the reduced model tensors

    `states` should be a list of indices specifying the states to keep, otherwise all
    states are kept.

    `strategy` is a keyword specifying how modes are ranked by importance

    """

    # truncate by interaction order. "1" is linear in Qs, "2" is quadratic in Qs, etc.
    couplings_red = {order: couplings[order] for order in couplings if order <= order_max}
    if strategy is None:
        strategy = "C1N+F"

    # truncate by states
    if states is not None:  #
        couplings_red = truncate_by_states(couplings_red, states)
    # print(f"m_max: {m_max}")
    mode_measures = rank_modes(omegas, couplings_red, ranking_type=strategy)
    modes_ordered = dict(sorted(mode_measures.items(), key=lambda item: item[1], reverse=True))
    # print(f"{'*'*50}\nMODES SORTED BY IMPORTANCE ({strategy}):")
    # for m in modes_ordered:
    #     print(f'Q{m} : {modes_ordered[m]}')
    # print(f"{'*'*50}\n")
    modes_keep = list(modes_ordered.keys())[:m_max]
    omegas_red = omegas[modes_keep]

    for order in couplings_red:
        if order == 1:
            couplings_red[order] = couplings_red[order][:, :, modes_keep]
        elif order == 2:
            couplings_red[order] = couplings_red[order][:, :, modes_keep, :][:, :, :, modes_keep]
        elif order == 3:
            raise ValueError("cant handle cubic order yet")

    return omegas_red, couplings_red


def get_reduced_model_manual(
    omegas, couplings, m_max, order_max=2, selected_modes=None, selected_states=None
):
    """
    `omegas` and `couplings` are the original omega and coupling tensors

    `m_max` is the number of modes to be included in the reduced model tensors

    `states` should be a list of indices specifying the states to keep, otherwise all
    states are kept.

    `strategy` is a keyword specifying how modes are ranked by importance

    """

    # truncate by interaction order. "1" is linear in Qs, "2" is quadratic in Qs, etc.
    couplings_red = {order: couplings[order] for order in couplings if order <= order_max}

    # truncate by states
    if selected_states is not None:  #
        couplings_red = truncate_by_states(couplings_red, selected_states)
    print(f"m_max: {m_max}")

    omegas_red = omegas[selected_modes]

    for order in couplings_red:
        if order == 1:
            couplings_red[order] = couplings_red[order][:, :, selected_modes]
        elif order == 2:
            couplings_red[order] = couplings_red[order][:, :, selected_modes, :][
                :, :, :, selected_modes
            ]
        elif order == 3:
            raise ValueError("cant handle cubic order yet")
        elif order == 4:
            raise ValueError("cant handle quairt order yet")

    return omegas_red, couplings_red


def eff_coupling_strength(omegas, couplings, I, J, i):
    constant_couplings = couplings[0]
    linear_couplings = couplings[1]
    numer = linear_couplings[I, J, i]
    denom = constant_couplings[J, J] - constant_couplings[I, I] - omegas[i]
    return np.log10(np.abs(numer / denom))
