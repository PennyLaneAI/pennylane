import numpy as np
import scipy

au_to_cm = 219475
hbar = 6.022*1.055e12 # (amu)*(angstrom^2/s)
c_light = 3*10**8 # m/s


def _pm_cost(q):
	nnuc, _, nmodes = q.shape

	xi_pm = 0.0
	for p in range(nmodes):
		for i in range(nnuc):
			q2 = 0.0
			for alpha in range(3):
				q2 += q[i,alpha,p]**2
			xi_pm += q2**2

	return -xi_pm

def _mat_transform(u, qmat):
	qloc = np.einsum('qp,iaq->iap', u, qmat)

	return qloc

def _params_to_unitary(params, nmodes):
        ugen = np.zeros((nmodes,nmodes))
	
        idx = 0
        for m1 in range(nmodes):
                for m2 in range(m1):
                        ugen[m1,m2] += params[idx]
                        ugen[m2,m1] -= params[idx]
                        idx += 1

        return scipy.linalg.expm(ugen)

def _params_cost(params, qmat, nmodes):
        uparams = _params_to_unitary(params, nmodes)
        qrot = _mat_transform(uparams, qmat)
        return _pm_cost(qrot)

def _q_normalizer(qmat):
	qnormalized = np.zeros_like(qmat)
	natoms,_,nmodes = qmat.shape
	norms_arr = np.zeros(nmodes)

	for m in range(nmodes):
		m_norm = np.sum(np.abs(qmat[:,:,m])**2)
		qnormalized[:,:,m] = qmat[:,:,m] / np.sqrt(m_norm)
		norms_arr[m] = np.sqrt(m_norm)

	return qnormalized, norms_arr

def _localization_unitary(qmat, verbose=False, rand_start=True):
	natoms,_,nmodes = qmat.shape
	num_params = int(nmodes*(nmodes-1)/2)

	if rand_start:
		params = 2*np.pi*np.random.rand(num_params)
	else:
		params = np.zeros(num_params)

	qnormalized, norms_arr = _q_normalizer(qmat)

	ini_cost = _pm_cost(qnormalized)
	if verbose:
		print(f"Initial cost is {ini_cost}")

	optimization_res = scipy.optimize.minimize(_params_cost, params, args=(qnormalized, nmodes))
	if optimization_res.success is False:
		print("WARNING: mode localization finished unsuccessfully, returning normal modes...")
		return _params_to_unitary(0*params, nmodes), qmat

	params_opt = optimization_res.x
	uloc = _params_to_unitary(params_opt, nmodes)

	qloc = _mat_transform(uloc, qmat)

	if verbose:
		print(f"Final cost is {optimization_res.fun}")
		print("Original displacements:")
		for m in range(nmodes):
			print(f"\nMode {m}:")
			for i in range(natoms):
				print(f"{qmat[i,0,m]}  {qmat[i,1,m]}  {qmat[i,2,m]}")

		print("\n\nLocalized displacements:")
		for m in range(nmodes):
			print(f"\nMode {m}:")
			for i in range(natoms):
				print(f"{qloc[i,0,m]}  {qloc[i,1,m]}  {qloc[i,2,m]}")

	return uloc, qloc

def _localize_modes(freqs, disp_vecs, verbose=False, order=True):
	nmodes = len(freqs)
	hess_normal = np.zeros((nmodes,nmodes))
	for m in range(nmodes):
                hess_normal[m,m] = freqs[m]**2

	natoms,_ = np.shape(disp_vecs[0])

	qmat = np.zeros((natoms, 3, nmodes))
	for m in range(nmodes):
		dvec = disp_vecs[m]
		for i in range(natoms):
			for alpha in range(3):
				qmat[i,alpha,m] = dvec[i,alpha]

	uloc, qloc = _localization_unitary(qmat, verbose=verbose)
	hess_loc = uloc.transpose() @ hess_normal @ uloc
	loc_freqs = np.sqrt(np.array([hess_loc[m,m] for m in range(nmodes)]))


	if order is True:
		loc_perm = np.argsort(loc_freqs)
		loc_freqs = loc_freqs[loc_perm]
		qloc = qloc[:,:,loc_perm]
		uloc = uloc[:,loc_perm]

	if verbose:
		print("Starting frequencies are (in cm^-1):")
		print(freqs * au_to_cm)
		print("Localized frequencies are (in cm^-1):")
		print(loc_freqs * au_to_cm)

	return loc_freqs, qloc, uloc


def localize_normal_modes(results, verbose=True, freq_separation=[2600]):
        '''
        Arguments: results dictionary obtained from harmonic_analysis
        separates frequencies at each point in freq_separation array

        Returns: new dictionary with information for localized modes
        '''
        freqs_in_cm = results['freq_wavenumber']
        freqs = freqs_in_cm / au_to_cm
        disp_vecs = results['norm_mode']
        nmodes = len(freqs)
        
        modes_arr = []
        freqs_arr = []
        disps_arr = []
        num_seps = len(freq_separation)
        min_modes = np.nonzero(freqs_in_cm <= freq_separation[0])[0]

        modes_arr.append(min_modes)
        freqs_arr.append(freqs[min_modes])
        disps_arr.append(disp_vecs[min_modes])

        for i_sep in range(num_seps-1):
                is_bigger = np.array(freq_separation[i_sep] <= freqs_in_cm)
                is_smaller = np.array(freq_separation[i_sep+1] >= freqs_in_cm)
                mid_modes = np.nonzero(is_bigger * is_smaller)[0]
                modes_arr.append(mid_modes)
                freqs_arr.append(freqs[mid_modes])
                disps_arr.append(disp_vecs[mid_modes])

        max_modes = np.nonzero(freqs_in_cm >= freq_separation[-1])[0]

        modes_arr.append(max_modes)
        freqs_arr.append(freqs[max_modes])
        disps_arr.append(disp_vecs[max_modes])

        if verbose:
                for idx,f_arr in enumerate(freqs_arr):
                        print(f"Set of frequencies {idx} is {f_arr*au_to_cm}")

        natoms = np.shape(disp_vecs[0])[0]
        
        loc_freqs_arr = []
        qlocs_arr = []
        ulocs_arr = []
        for idx in range(num_seps+1):
                num_freqs = len(freqs_arr[idx])
                if num_freqs > 1:
                        loc_freqs, qloc, uloc = _localize_modes(freqs_arr[idx], disps_arr[idx])
                elif num_freqs == 1:
                        loc_freqs = freqs_arr[idx]
                        qloc = np.zeros((natoms, 3, 1))
                        qloc[:,:,0] = disps_arr[idx][0]
                        uloc = np.zeros((1,1))
                        uloc[0,0] = 1
                else:
                        loc_freqs = []
                        uloc = []
                        qloc = []
                loc_freqs_arr.append(loc_freqs)
                qlocs_arr.append(qloc)
                ulocs_arr.append(uloc)

        uloc = np.zeros((nmodes,nmodes))
        for idx in range(num_seps+1):
                for i_enu,i_str in enumerate(modes_arr[idx]):
                        for j_enu,j_str in enumerate(modes_arr[idx]):
                                uloc[i_str, j_str] = ulocs_arr[idx][i_enu,j_enu]

        loc_results = {}
        loc_freqs = []
        for idx in range(num_seps+1):
                loc_freqs.extend(loc_freqs_arr[idx])
        loc_freqs = np.array(loc_freqs)
        loc_results['freq_wavenumber'] = loc_freqs * au_to_cm
        new_disp = []
        for idx in range(num_seps+1):
                for m in range(len(loc_freqs_arr[idx])):
                        m_disp = qlocs_arr[idx][:,:,m]
                        new_disp.append(m_disp)
        loc_results['norm_mode'] = new_disp

        if verbose:
                print(f"\nFinished localizing modes (separated at each frequency point in {freq_separation})")
                print("Starting frequencies are (in cm^-1):")
                print(freqs * au_to_cm)
                print("Localized frequencies are (in cm^-1):")
                print(loc_freqs * au_to_cm)

        return loc_results, uloc, np.array(new_disp)

