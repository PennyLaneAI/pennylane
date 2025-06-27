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
                A_mat[idx, t1_idx] = 1
                A_mat[idx, t2_idx] = 2*eta
            
            A_mat[idx, O_dict[(ii,jj)]] = eta
            idx += 1

    # full two-body components
    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                for ll in range(N):
                    if ii == jj and kk == ll:
                        A_mat[idx, t2_idx] += 0.25
                    if kk == ll:
                        A_mat[idx, O_dict[(ii,jj)]] += 0.125
                    if ii == jj:
                        A_mat[idx, O_dict[(kk,ll)]] += 0.125
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
                        A_mat[idx, O_dict[(ii,jj)]] += 0.25
                    if ii == jj:
                        A_mat[idx, O_dict[(kk,ll)]] += 0.25
                    if kk == jj:
                        A_mat[idx, O_dict[(ii,ll)]] -= 0.25
                    if ii == ll:
                        A_mat[idx, O_dict[(kk,jj)]] -= 0.25
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
    
    if verbose:
        print("Finished BLISS optimization, found parameters")
        print(f"t1={t1:.2e}, t2={t2:.2e}, O =\n")
        print(np.round(O, decimals=3))

    # Build symmetry shift operator S
    s2_tbt = t2 * Ne2_tbt
    for i in range(N):
        for j in range(N):
            for k in range(N):
                s2_tbt[i, j, k, k] += O[i, j] / 2
                s2_tbt[k, k, i, j] += O[i, j] / 2
    
    s1_obt = t1 * Ne_obt - eta * O
    const_shift = -t1*eta - t2*eta**2

    new_obt = one_body - s1_obt
    new_tbt = two_body - s2_tbt

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