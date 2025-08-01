import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def ob_correction(tbt, spin_orb=False):
    #returns correction to one-body tensor coming from tbt inside fermionic operator F
    if spin_orb:
        spin_fact = 1/2
        print("Obtaining one-body correction for spin-orbitals, be wary of workflow!")
    else:
        spin_fact = 1
    obt_corr = spin_fact * np.einsum('ijkk->ij', tbt)
    
    return obt_corr

def pauli_1_norm(obt, tbt, split_spin=True):
    '''
    Formally 1-norm of 2-body tensor has different contributions from spin sectors (split_spin=True)
    In practice we usually implement the LCU without splitting the spin since the PREP cost is way lower 
    '''
    N = obt.shape[0]
    lambda_1 = np.sum(np.abs(obt + ob_correction(tbt)))
    
    if split_spin:
        lambda_2 = 0.25 * np.sum(np.abs(tbt))
        for r in range(N):
            for p in range(r+1,N):
                for q in range(N):
                    for s in range(q+1,N):
                        lambda_2 += 0.5 * np.abs(tbt[p,q,r,s] - tbt[p,s,r,q])
    else:
        lambda_2 = 0.5 * np.sum(np.abs(tbt))


    return lambda_1+lambda_2

def symmetry_builder(N):
    #returns Ne and Neˆ2, both symmetries can be represented in orbitals (so no need for spin-orbitals)
    Ne_obt = np.zeros((N, N))
    for i in range(N):
        Ne_obt[i,i] = 1

    Ne2_tbt = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            Ne2_tbt[i,i,j,j] = 1

    return Ne_obt, Ne2_tbt

def bliss_linprog(one_body, two_body, eta, verbose=False, model="highs"):
    """
    eta: number of electrons
    """
    if verbose:
        print(f"Starting BLISS linprog routine with 1-norm {pauli_1_norm(one_body, two_body)}")

    N = one_body.shape[0]
    Ne_obt, Ne2_tbt = symmetry_builder(N)

    obt_len = N*N
    tbt_full_len = N**4
    tbt_sym_len = int((N*(N-1)/2)**2)

    # Variables: [t1, t2, omat..., obt..., tbt1..., tbt2...]
    len_onenorm = obt_len + tbt_full_len + tbt_sym_len
    L = 2 + obt_len 
    M = L + len_onenorm
    t1_idx = 0
    t2_idx = 1
    omat_idx = 2
    obt_idx = omat_idx + obt_len
    tbt1_idx = obt_idx + obt_len
    tbt2_idx = tbt1_idx + tbt_full_len
    
    # Objective: Minimize sum(obt) + sum(tbt1) + sum(tbt2)
    c = np.zeros(M)
    c[obt_idx:] = 1
    

    # create lambda vector for starting 1-norm and dictionary for Omat flattener
    lambda_vec = np.zeros(len_onenorm)
    obt_corr = one_body + ob_correction(two_body) 
    O_dict = {}
    
    idx = 0
    for ii in range(N):
        for jj in range(N):
            lambda_vec[idx] = obt_corr[ii,jj]
            O_dict[(ii,jj)] = idx + 2 #plus two comes from t1 and t2 offset
            idx += 1

    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                for ll in range(N):
                    lambda_vec[idx] = 0.25*two_body[ii,jj,kk,ll]
                    idx += 1

    for ii in range(N):
        for jj in range(N):
            for kk in range(ii):
                for ll in range(jj):
                    lambda_vec[idx] = 0.5*(two_body[ii,jj,kk,ll] - two_body[ii,ll,kk,jj])
                    idx += 1
    
    ## Now create A matrix
    A_mat = np.zeros((len_onenorm,L))
    # one-body components
    idx = 0
    for ii in range(N):
        for jj in range(N):
            if ii == jj:
                # t1 * Ne
                A_mat[idx, t1_idx] += 1
                # two-body correction: t2 * N * Ne_obt
                A_mat[idx, t2_idx] += N
                # two-body correction: delta_ij * sum_k O_kk
                for kk in range(N):
                    kk_idx = O_dict[(kk,kk)]
                    A_mat[idx, kk_idx] += 1
            
            # -eta * O_ij + two_body correction: O_ij * N
            A_mat[idx, O_dict[(ii,jj)]] += N - eta

            idx += 1

    # full two-body components
    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                for ll in range(N):
                    if ii == jj and kk == ll:
                        A_mat[idx, t2_idx] += 0.25
                    if kk == ll:
                        A_mat[idx, O_dict[(ii,jj)]] += 0.25
                    if ii == jj:
                        A_mat[idx, O_dict[(kk,ll)]] += 0.25
                    idx += 1

    # symmetric two-body components
    for ii in range(N):
        for jj in range(N):
            for kk in range(ii):
                for ll in range(jj):
                    if ii == jj and kk == ll:
                        A_mat[idx, t2_idx] += 0.5
                    if ii == ll and kk == jj:
                        A_mat[idx, t2_idx] -= 0.5
                    if kk == ll:
                        A_mat[idx, O_dict[(ii,jj)]] += 0.5
                    if ii == jj:
                        A_mat[idx, O_dict[(kk,ll)]] += 0.5
                    if kk == jj:
                        A_mat[idx, O_dict[(ii,ll)]] -= 0.5
                    if ii == ll:
                        A_mat[idx, O_dict[(kk,jj)]] -= 0.5
                    idx += 1

    A_ub = lil_matrix((2*len_onenorm, M), dtype=np.float64)
    A_ub[:len_onenorm, :L] = A_mat
    A_ub[len_onenorm:, :L] = -A_mat
    for mm in range(len_onenorm):
        A_ub[mm, L+mm] = -1
        A_ub[len_onenorm+mm, L+mm] = -1
    A_ub = csr_matrix(A_ub)

    b_ub = np.zeros(2*len_onenorm)
    b_ub[:len_onenorm] = lambda_vec
    b_ub[len_onenorm:] = -lambda_vec

    # Bounds:   - t1, t2, omat are free real numbers; obt, tbt1, tbt2 are positive real numbers
    bounds = [(None, None)] * L + [(0,None)] * len_onenorm 

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=model)

    if not res.success:
        raise RuntimeError("BLISS linprog failed: " + res.message)

    # Extract solution
    t1 = res.x[t1_idx]
    t2 = res.x[t2_idx]
    omat = res.x[omat_idx:omat_idx+obt_len].reshape((N, N))
    O = (omat + omat.T) / 2
    
    if verbose == 2:
        print("Finished BLISS optimization, found parameters")
        print(f"t1={t1:.2e}, t2={t2:.2e}, O =\n")
        print(np.round(O, decimals=3))

    # Build symmetry shift operator S
    s2_tbt = t2 * Ne2_tbt
    for i in range(N):
        for j in range(N):
            for k in range(N):
                s2_tbt[i, j, k, k] += O[i, j]
                s2_tbt[k, k, i, j] += O[i, j]
    
    s1_obt = t1 * Ne_obt - eta*O
    const_shift = -t1*eta - t2*eta**2

    new_obt = one_body - s1_obt
    new_tbt = two_body - s2_tbt

    if verbose:
        print(f"Finished BLISS linprog routine with 1-norm {pauli_1_norm(new_obt, new_tbt)}")


    return new_obt, new_tbt

def bliss_cdf_frag(lamb_mat, eta, model="highs"):
    N = lamb_mat.shape[0]
    Ne_obt, Ne2_tbt = symmetry_builder(N)

    lam_len = int(N*(N+1)/2)
    L = 1 + N
    M = L + lam_len


    # variables: [mu2, theta..., lam(up_diag)...]
    c = np.zeros(M)
    c[N+1:] = 1

    lambda_vec = np.zeros(lam_len)
    idx_dict = {}
    idx = 0
    for ii in range(N):
        for jj in range(ii+1):
            if ii == jj:
                lambda_vec[idx] = 0.5 * lamb_mat[ii,ii]
            else:
                lambda_vec[idx] = 2 * lamb_mat[ii,jj]
            idx_dict[(ii,jj)] = idx + 1 #+1 account for mu2 component
            idx += 1

    A_mat = np.zeros((lam_len, L))

    idx = 0
    for ii in range(N):
        for jj in range(ii+1):
            if ii == jj:
                A_mat[idx, :] = 0.5
            else:
                A_mat[idx, 0] = 2
                A_mat[idx, ii+1] = 1
                A_mat[idx, jj+1] = 1
            idx += 1

    A_ub = np.zeros((2*lam_len, M))
    A_ub[:lam_len, :L] = A_mat
    A_ub[lam_len:, :L] = -A_mat
    for mm in range(lam_len):
        A_ub[mm, L+mm] = -1
        A_ub[lam_len+mm, L+mm] = -1

    b_ub = np.zeros(2*lam_len)
    b_ub[:lam_len] = lambda_vec
    b_ub[lam_len:] = -lambda_vec

    # Bounds: t1, t2, omat, obt, tbt1, tbt2 are all free real numbers
    bounds = [(None, None)] * L + [(0,None)] * lam_len 

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=model)

    if not res.success:
        raise RuntimeError("BLISS linprog failed: " + res.message)

    # Extract solution
    mu2 = res.x[0]
    theta_vec = res.x[1:N+1]

    new_lamb_mat = np.copy(lamb_mat)
    for ii in range(N):
        for jj in range(N):
            new_lamb_mat[ii,jj] -= (theta_vec[ii] + theta_vec[jj])/2
            if ii == jj:
                new_lamb_mat[ii,ii] -= mu2

    return new_lamb_mat

def cdf_ob_correction(cores, leaves):
    ob_corr = 2*np.einsum('rip,rjp,rpq,rkq,rkq->ij', leaves, leaves, cores, leaves, leaves)

    return ob_corr

def cdf_reconstruct(cores, leaves):
    return np.einsum('rip,rjp,rpq,rkq,rlq->ijkl', leaves, leaves, cores, leaves, leaves)

def cdf_one_norm(obt, cores, leaves):
    N = obt.shape[0]
    lam2 = 0

    for ii in range(N):
        lam2 += 0.5*np.sum(np.abs(cores[:,ii,ii]))
        for jj in range(ii):
            lam2 += 2*np.sum(np.abs(cores[:,ii,jj]))

    obt_corrected = obt + cdf_ob_correction(cores, leaves)
    D,U = np.linalg.eigh(obt_corrected)
    lam1 = np.sum(np.abs(D))

    return lam1 + lam2

def bliss_one_body(obt):
    D,_ = np.linalg.eigh(obt)

    return np.median(D)

def bliss_cdf(obt, cores, leaves, eta, verbose=False, model="highs"):
    """
    eta: number of electrons
    """
    R, N, _ = cores.shape

    if verbose:
        print(f"Initial one-norm of CDF is {cdf_one_norm(obt, cores, leaves):.2e}")
    
    new_cores = np.zeros_like(cores)

    for rr in range(R):
        new_cores[rr, ::] = bliss_cdf_frag(cores[rr,::], eta, model=model)


    obt_corr = obt + cdf_ob_correction(new_cores, leaves)
    mu1 = bliss_one_body(obt_corr)

    obt_bliss = obt - mu1*np.eye(N)
    if verbose:
        print(f"Final one-norm of CDF is {cdf_one_norm(obt_bliss, new_cores, leaves):.2e}")

    return obt_bliss, new_cores

def general_bliss_linprog(one_body, two_body, ob_syms, tb_syms=[], verbose=True, model="highs"):
    """
    Perform BLISS on electronic structure Hamiltonian for arbitrary number of one- and two-body symmetries

    ob_syms: array containing S1 tuples, where each ob_syms[i] corresponds to (sym, val) for sym the one-body operator and val its target eigenvalue block
        we denote these as ob_syms[i] = (O_i, o_i)
    tb_syms: array containing S2 tuples, same as ob_syms but for two-body symmetries
        we denote these as tb_syms[i] = (T_i, t_i)
    """
    if verbose:
        print(f"Starting general BLISS linprog routine with 1-norm {pauli_1_norm(one_body, two_body)}")

    S1 = len(ob_syms)
    S2 = len(tb_syms)

    ob_combs = []
    for i1 in range(S1):
        for i2 in range(i1+1):
            ob_combs.append((i1,i2))

    pure_ones = S1 #number of one-body operators of form a_i*(O_i - o_i)
    mixed_twos = len(ob_combs) #number of two-body operators of form b_ij*(0.5*(O_i-o_i)*(O_j-o_j) + h.c.)
    pure_twos = S2 #number of two-body operators of from 0.5*c_i*(T_i - t_i)


    if verbose == 2:
        print(f"Found {S1} one-body symmetries and {S2} two-body symmetries")

    N = one_body.shape[0]

    obt_len = N*N
    tbt_full_len = N**4
    tbt_sym_len = int((N*(N-1)/2)**2)

    # Variables: [a_1,a_2,...,a_S1,b_11,b_21,b_22,b_31,...,b_S1S1,c_1,...,c_S2, O_1..., O_2..., ..., O_S1, obt..., tbt1..., tbt2...]
    len_onenorm = obt_len + tbt_full_len + tbt_sym_len
    num_scalars = pure_ones + mixed_twos + pure_twos
    num_bliss_ob_vars = pure_ones * obt_len
    L = num_scalars + num_bliss_ob_vars
    M = L + len_onenorm
    mixed2_index = pure_ones
    pure2_index = mixed2_index + mixed_twos
    omats_indices = [pure2_index + pure_twos]
    for ii in range(pure_ones - 1):
        omats_indices.append(omats_indices[-1] + obt_len)
    obt_index = omats_indices[-1] + obt_len
    tbt1_index = obt_index + obt_len
    tbt2_index = tbt1_index + tbt_full_len

    if verbose == 2:
        print(f"Optimizing {pure_ones} one-body scalars, {mixed_twos} composite two-body scalars, and {pure_twos} two-body scalars")
        print(f"Using {pure_ones} one-body matrices for optimization, each one with {obt_len} elements")
    
    # Objective: Minimize sum(obt) + sum(tbt1) + sum(tbt2)
    c = np.zeros(M)
    c[obt_index:] = 1
    
    # create lambda vector for starting 1-norm and dictionary for Omat flattener
    lambda_vec = np.zeros(len_onenorm)
    obt_corr = one_body + ob_correction(two_body) 
    O_dict = {}
    
    idx = 0
    for ii in range(N):
        for jj in range(N):
            lambda_vec[idx] = obt_corr[ii,jj]
            O_dict[(ii,jj)] = idx
            idx += 1

    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                for ll in range(N):
                    lambda_vec[idx] = 0.25*two_body[ii,jj,kk,ll]
                    idx += 1

    for ii in range(N):
        for jj in range(N):
            for kk in range(ii):
                for ll in range(jj):
                    lambda_vec[idx] = 0.5*(two_body[ii,jj,kk,ll] - two_body[ii,ll,kk,jj])
                    idx += 1
    
    # Calculate terms Zp == sum_k O^{(p)}_{kk}
    sym_invars = np.zeros(pure_ones)
    for pp in range(pure_ones):
        sym_invars[pp] = np.einsum('kk->', ob_syms[pp][0])

    tb_invars = [np.einsum('ijkk->ij', tb_sym[zz][0]) for zz in range(pure_twos)]


    ## Now create A matrix
    A_mat = np.zeros((len_onenorm,L))
    # one-body components
    idx = 0
    for ii in range(N):
        for jj in range(N):
            # Part 1: h_ij terms
            # i) a_p * O_p terms
            for pp in range(pure_ones):
                A_mat[idx, pp] += ob_syms[pp][0][ii,jj]

            # ii) b_pq * ((O_p - o_p)*(O_q - o_q) + h.c.) terms
            for pq_idx, sub_pqs in enumerate(ob_combs):
                pp, qq = sub_pqs
                bpq_idx = mixed2_index + pq_idx
                A_mat[idx, bpq_idx] -= ob_syms[qq][1]*ob_syms[pp][0][ii,jj] + ob_syms[pp][1]*ob_syms[qq][0][ii,jj]

            # iii) o_p*Beta_p terms
            for pp in range(pure_ones):
                beta_p_idx = omats_indices[pp]
                A_mat[idx, beta_p_idx + idx] -= ob_syms[pp][1]

            # Part 2: sum_k g_ijkk terms
            # i) b_pq * (o_p_ij*Zq + o_q_ij*Zp)
            for pq_idx, sub_pqs in enumerate(ob_combs):
                pp, qq = sub_pqs
                bpq_idx = mixed2_index + pq_idx
                A_mat[idx, bpq_idx] += ob_syms[pp][0][ii,jj]*sym_invars[qq] + ob_syms[qq][0][ii,jj]*sym_invars[pp]

            # ii) c_z * t_z_ijkk
            for zz in range(pure_twos):
                A_mat[idx, pure2_index + zz] += tb_invars[zz][ii,jj]

            # iii) a) Beta_p_ij*Zp + b) O_p_ij*(sum_k Beta_p_kk)
            for pp in range(pure_ones):
                beta_p_idx = omats_indices[pp]
                # a)
                A_mat[idx, beta_p_idx + idx] += sym_invars[pp]

                # b)
                for i2 in range(N):
                    for j2 in range(N):
                        if i2 == j2:
                            kk_idx = O_dict[(i2,j2)]
                            A_mat[idx, beta_p_idx + kk_idx] += ob_syms[pp][0][ii,jj]

            idx += 1

    # full two-body components
    for ii in range(N):
        for jj in range(N):
            ij_idx = O_dict[(ii,jj)]
            for kk in range(N):
                for ll in range(N):
                    kl_idx = O_dict[(kk,ll)]

                    # Part 1: 1/4 * b_pq*(O_p_ij*O_q_kl + O_q_ij*O_p_kl)
                    for pq_idx, sub_pqs in enumerate(ob_combs):
                        pp, qq = sub_pqs
                        bpq_idx = mixed2_index + pq_idx
                        A_mat[idx, bpq_idx] += 0.25*(ob_syms[pp][0][ii,jj]*ob_syms[qq][0][kk,ll] + ob_syms[qq][0][ii,jj]*ob_syms[pp][0][kk,ll])

                    # Part 2: 1/4 * c_z * t_z_ijkl
                    for zz in range(pure_twos):
                        A_mat[idx, pure2_index + zz] += 0.25*(tb_syms[zz][0][ii,jj,kk,ll])

                    # Part 3: 1/4 * Beta_p_ij*O_p_kl + O_p_ij*Beta_p_kl
                    for pp in range(pure_ones):
                        beta_p_idx = omats_indices[pp]
                        A_mat[idx, beta_p_idx + ij_idx] += 0.25*(ob_syms[pp][0][kk,ll])
                        A_mat[idx, beta_p_idx + kl_idx] += 0.25*(ob_syms[pp][0][ii,jj])

                    idx += 1

    # symmetric two-body components
    # corresponds to same as before but with factor of 1/2 instead of 1/4 and considering ijkl -> ijkl - ilkj
    for ii in range(N):
        for jj in range(N):
            for kk in range(ii):
                for ll in range(jj):
                    ij_idx = O_dict[(ii,jj)]
                    kl_idx = O_dict[(kk,ll)]
                    il_idx = O_dict[(ii,ll)]
                    kj_idx = O_dict[(kk,jj)]

                    # Part 1: ijkl: b_pq*(O_p_ij*O_q_kl + O_q_ij*O_p_kl)
                    for pq_idx, sub_pqs in enumerate(ob_combs):
                        pp, qq = sub_pqs
                        bpq_idx = mixed2_index + pq_idx
                        A_mat[idx, bpq_idx] += 0.5*ob_syms[pp][0][ii,jj]*ob_syms[qq][0][kk,ll] + ob_syms[qq][0][ii,jj]*ob_syms[pp][0][kk,ll]
                        A_mat[idx, bpq_idx] -= 0.5*ob_syms[pp][0][ii,ll]*ob_syms[qq][0][kk,jj] + ob_syms[qq][0][ii,ll]*ob_syms[pp][0][kk,jj]

                    # Part 2: ijkl: c_z * t_z_ijkl
                    for zz in range(pure_twos):
                        A_mat[idx, pure2_index + zz] += 0.5*tb_syms[zz][0][ii,jj,kk,ll]
                        A_mat[idx, pure2_index + zz] -= 0.5*tb_syms[zz][0][ii,ll,kk,jj]

                    # Part 3: ijkl -> Beta_p_ij*O_p_kl + O_p_ij*Beta_p_kl
                    for pp in range(pure_ones):
                        beta_p_idx = omats_indices[pp]
                        A_mat[idx, beta_p_idx + ij_idx] += 0.5*ob_syms[pp][0][kk,ll]
                        A_mat[idx, beta_p_idx + kl_idx] += 0.5*ob_syms[pp][0][ii,jj]

                        A_mat[idx, beta_p_idx + il_idx] -= 0.5*ob_syms[pp][0][kk,jj]
                        A_mat[idx, beta_p_idx + kj_idx] -= 0.5*ob_syms[pp][0][ii,ll]

                    idx += 1

    A_ub = lil_matrix((2*len_onenorm, M), dtype=np.float64)
    A_ub[:len_onenorm, :L] = A_mat
    A_ub[len_onenorm:, :L] = -A_mat
    for mm in range(len_onenorm):
        A_ub[mm, L+mm] = -1
        A_ub[len_onenorm+mm, L+mm] = -1
    A_ub = csr_matrix(A_ub)

    b_ub = np.zeros(2*len_onenorm)
    b_ub[:len_onenorm] = lambda_vec
    b_ub[len_onenorm:] = -lambda_vec

    # Bounds:   - t1, t2, omat are free real numbers; obt, tbt1, tbt2 are positive real numbers
    bounds = [(None, None)] * L + [(0,None)] * len_onenorm 

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=model)

    if not res.success:
        raise RuntimeError("BLISS linprog failed: " + res.message)

    # Extract solution
    a_vec = res.x[:pure_ones]
    b_flat_vec = res.x[pure_ones:pure2_index]
    c_vec = res.x[pure2_index:omats_indices[0]]
    Omats = [res.x[omats_indices[pp]:omats_indices[pp]+obt_len].reshape((N,N)) for pp in range(pure_ones)]
    for pp in range(pure_ones):
        Omats[pp] = (Omats[pp] + Omats[pp].T) / 2

    if verbose == 2:
        print("Finished BLISS optimization, found parameters:")
        print(f"a_vec = {np.round(a_vec, decimals=3)}")
        for pq_idx, sub_pqs in enumerate(ob_combs):
            pp, qq = sub_pqs
            print(f"b_{pp}_{qq} = {b_flat_vec[pq_idx]:.2e}")
        print(f"c_vec = {np.round(c_vec, decimals=3)}")

        for pp in range(pure_ones):
            print(f"O_{pp} = ")
            print(np.round(Omats[pp], decimals=3))

    # Build symmetry shift operator
    # Part 1: two-body component
    # i) b_pq * (O_p*O_q + O_q*O_p)
    tbt_shift = np.zeros((N,N,N,N))
    for pq_idx, sub_pqs in enumerate(ob_combs):
        pp, qq = sub_pqs
        tbt_shift += b_flat_vec[pq_idx]*np.einsum("ij,kl->ijkl",ob_syms[pp][0],ob_syms[qq][0])
        tbt_shift += b_flat_vec[pq_idx]*np.einsum("ij,kl->ijkl",ob_syms[qq][0],ob_syms[pp][0])

    # ii) c_z * T_z
    for zz in range(pure_twos):
        tbt_shift += c_vec[zz] * tb_syms[zz][0]

    # iii) Beta_p*O_p + O_p*Beta_p
    for pp in range(pure_ones):
        tbt_shift += np.einsum("ij,kl->ijkl",Omats[pp],ob_syms[pp][0])
        tbt_shift += np.einsum("ij,kl->ijkl",ob_syms[pp][0],Omats[pp])

    # Part 2: one-body component
    # i) a_p * O_p
    obt_shift = np.zeros((N,N))
    for pp in range(pure_ones):
        obt_shift += a_vec[pp] * ob_syms[pp][0]

    # ii) b_pq * (o_q*O_p + o_p*O_q)
    for pq_idx, sub_pqs in enumerate(ob_combs):
        pp, qq = sub_pqs
        obt_shift -= b_flat_vec[pq_idx] * ob_syms[qq][1] * ob_syms[pp][0]
        obt_shift -= b_flat_vec[pq_idx] * ob_syms[pp][1] * ob_syms[qq][0]

    # iii) o_p*Beta_p
    for pp in range(pure_ones):
        obt_shift -= ob_syms[pp][1] * Omats[pp]

    """
    # Part 3: constant shift
    const_shift = 0.0
    for pp in range(pure_ones):
        const_shift -= a_vec[pp] * ob_syms[pp][1]

    for pq_idx, sub_pqs in enumerate(ob_combs):
        pp, qq = sub_pqs
        const_shift += b_flat_vec[pq_idx] * ob_syms[pp][1] * ob_syms[qq][1]

    for zz in range(pure_twos):
        const_shift -= 0.5 * c_vec[zz] * tb_syms[zz][1]
    """


    new_obt = one_body - obt_shift
    new_tbt = two_body - tbt_shift

    if verbose:
        print(f"Finished BLISS linprog routine with 1-norm {pauli_1_norm(new_obt, new_tbt)}")


    return new_obt, new_tbt