import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import itertools


def _obtain_r2(ytrue, yfit):
	ymean = np.sum(ytrue) / len(ytrue)
	ssres = np.sum((ytrue - yfit)**2) 
	sstot = np.sum((ytrue - ymean)**2)

	return 1-ssres/sstot

def _remove_harmonic(freqs, pes_onebody):
	nmodes, quad_order = np.shape(pes_onebody)
	gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(quad_order)

	harmonic_pes = np.zeros((nmodes, quad_order))
	anh_pes = np.zeros((nmodes, quad_order))

	for ii in range(nmodes):
		ho_const = freqs[ii] / 2
		harmonic_pes[ii,:] = ho_const * (gauss_grid**2)
		anh_pes[ii,:] = pes_onebody[ii,:] - harmonic_pes[ii,:]

	return nmodes, quad_order, anh_pes, harmonic_pes

def _fit_onebody(anh_pes, deg, verbose=True, min_deg = 3):
        if deg < min_deg:
                raise Exception(f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!")

        nmodes, quad_order = np.shape(anh_pes)
        gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)
        fs = np.zeros((nmodes, deg - min_deg + 1))
	
        predicted_1D = np.zeros_like(anh_pes)

        for i1 in range(nmodes):
                poly1D = PolynomialFeatures(degree=(min_deg,deg), include_bias=False)
                poly1D_features = poly1D.fit_transform(gauss_grid.reshape(-1, 1))
                poly1D_reg_model = LinearRegression()
                poly1D_reg_model.fit(poly1D_features, anh_pes[i1,:])
                fs[i1,:] = poly1D_reg_model.coef_
                predicted_1D[i1,:] = poly1D_reg_model.predict(poly1D_features)

        if verbose:
                for i1 in range(nmodes):
                        for my_deg in range(deg-min_deg+1):
                                print(f"Fit coefficient for q{i1+1}^{my_deg+min_deg} = {fs[i1,my_deg]}")

        return fs, predicted_1D

def _twobody_degs(deg, min_deg=3):
	fit_degs = []
	deg_idx = 0
	for feat_deg in range(min_deg,deg+1):
		max_deg = feat_deg - 1
		for deg_dist in range(1,max_deg+1):
			q1deg = max_deg-deg_dist+1
			q2deg = deg_dist
			fit_degs.append((q1deg,q2deg))

	return fit_degs

def _fit_twobody(pes_twobody, deg, min_deg=3):
	nmodes,_,quad_order,_ = np.shape(pes_twobody)
	gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

	if deg < min_deg:
		raise Exception(f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!")
	
	fit_degs = _twobody_degs(deg, min_deg)
	num_fs = len(fit_degs)
	fs = np.zeros((nmodes,nmodes,num_fs))

	predicted_2D = np.zeros_like(pes_twobody)

	grid_2D = np.array(np.meshgrid(gauss_grid,gauss_grid))
	q1 = grid_2D[0,::].flatten()
	q2 = grid_2D[1,::].flatten()
	idx_2D = np.array(np.meshgrid(range(quad_order),range(quad_order)))
	idx1 = idx_2D[0,::].flatten()
	idx2 = idx_2D[1,::].flatten()
	num_2D = len(q1)

	features = np.zeros((num_2D, num_fs))
	for deg_idx,Qs in enumerate(fit_degs):
		q1deg,q2deg = Qs
		features[:,deg_idx] = q1**(q1deg) * q2**(q2deg)
		
	for i1 in range(nmodes):
		for i2 in range(i1):
			poly2D = PolynomialFeatures(degree=(min_deg,deg), include_bias=False, interaction_only=True)
			Y = []
			for idx in range(num_2D):
				idx_q1 = idx1[idx]
				idx_q2 = idx2[idx]
				Y.append(pes_twobody[i1,i2,idx_q1,idx_q2])
			poly2D_reg_model = LinearRegression()
			poly2D_reg_model.fit(features, Y)
			fs[i1,i2,:] = poly2D_reg_model.coef_
			predicted = poly2D_reg_model.predict(features)
			for idx in range(num_2D):
				idx_q1 = idx1[idx]
				idx_q2 = idx2[idx]
				predicted_2D[i1,i2,idx_q1,idx_q2] = predicted[idx]


	return fs, predicted_2D

def _generate_bin_occupations(max_occ, nbins):
    # Generate all combinations placing max_occ balls in nbins
    combinations = list(itertools.product(range(max_occ+1), repeat=nbins))

    # Filter valid combinations
    valid_combinations = [combo for combo in combinations if sum(combo) == max_occ]

    return valid_combinations

def _threebody_degs(deg, min_deg=3):
	fit_degs = []
	deg_idx = 0
	for feat_deg in range(min_deg,deg+1):
		max_deg = feat_deg - 3
		if max_deg < 0:
			continue
		possible_occupations = _generate_bin_occupations(max_deg, 3)
		for occ in possible_occupations:
			q1deg = 1 + occ[0]
			q2deg = 1 + occ[1]
			q3deg = 1 + occ[2]
			fit_degs.append((q1deg,q2deg,q3deg))

	return fit_degs

def _fit_threebody(pes_threebody, deg, verbose=False, min_deg=3):
	nmodes,_,_,quad_order,_,_ = np.shape(pes_threebody)
	gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

	if deg < min_deg:
		raise Exception(f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!")
	
	predicted_3D = np.zeros_like(pes_threebody)
	fit_degs = _threebody_degs(deg)
	num_fs = len(fit_degs)
	fs = np.zeros((nmodes,nmodes,nmodes,num_fs))

	grid_3D = np.array(np.meshgrid(gauss_grid,gauss_grid,gauss_grid))
	q1 = grid_3D[0,::].flatten()
	q2 = grid_3D[1,::].flatten()
	q3 = grid_3D[2,::].flatten()
	idx_3D = np.array(np.meshgrid(range(quad_order),range(quad_order),range(quad_order)))
	idx1 = idx_3D[0,::].flatten()
	idx2 = idx_3D[1,::].flatten()
	idx3 = idx_3D[2,::].flatten()
	num_3D = len(q1)


	features = np.zeros((num_3D, num_fs))
	for deg_idx,Qs in enumerate(fit_degs):
		q1deg,q2deg,q3deg = Qs
		features[:,deg_idx] = q1**(q1deg) * q2**(q2deg) * q3**(q3deg)
	
	for i1 in range(nmodes):
		for i2 in range(i1):
			for i3 in range(i2):
				poly3D = PolynomialFeatures(degree=(min_deg,deg), include_bias=False, interaction_only=True)
				Y = []
				for idx in range(num_3D):
					idx_q1 = idx1[idx]
					idx_q2 = idx2[idx]
					idx_q3 = idx3[idx]
					Y.append(pes_threebody[i1,i2,i3,idx_q1,idx_q2,idx_q3])

				poly3D_reg_model = LinearRegression()
				poly3D_reg_model.fit(features, Y)
				fs[i1,i2,i3,:] = poly3D_reg_model.coef_
				predicted = poly3D_reg_model.predict(features)
				for idx in range(num_3D):
					idx_q1 = idx1[idx]
					idx_q2 = idx2[idx]
					idx_q3 = idx3[idx]
					predicted_3D[i1,i2,i3,idx_q1,idx_q2,idx_q3] = predicted[idx]

	if verbose:
		for i1 in range(nmodes):
			for i2 in range(i1):
				for i3 in range(i2):
					for fit_num in range(num_fs):
						q1deg, q2deg, q3deg = fit_degs[fit_num]
						print(f"Fit coefficient for q{i1+1}^{q1deg} q{i2+1}^{q2deg} q{i3+1}^{q3deg} = {fs[i1,i2,i3,fit_num]}")

	return fs, predicted_3D


def taylor_integrals(pes, deg=4, min_deg=3):
        r"""Returns the coefficients for real-space Hamiltonian.
        Args:
          pes: PES object.
          deg:
          min_deg:
        """

        print("Starting one-mode fitting...")
        nmodes, quad_order, anh_pes, harmonic_pes = _remove_harmonic(pes.freqs, pes.pes_onebody)
        coeff_1D,predicted_1D = _fit_onebody(anh_pes, deg, min_deg = min_deg)
        predicted_1D += harmonic_pes
        coeff_arr = [coeff_1D]
        predicted_arr = [predicted_1D]
        print("coeff1d: ", coeff_1D, "predicted: ", predicted_1D)

        if pes.pes_twobody is not None:
                print("Starting two-mode fitting...")
                coeff_2D,predicted_2D = _fit_twobody(pes.pes_twobody, deg, min_deg = min_deg)
                coeff_arr.append(coeff_2D)
                predicted_arr.append(predicted_2D)
                print("coeff2d: ", coeff_2D, "predicted: ", predicted_2D)

        if pes.pes_threebody is not None:
                print("Starting three-mode fitting...")
                coeff_3D,predicted_3D = _fit_threebody(pes.pes_threebody, deg, min_deg = min_deg)
                coeff_arr.append(coeff_3D)
                predicted_arr.append(predicted_3D)
                print("coeff3d: ", coeff_2D, "predicted: ", predicted_2D)

        return coeff_arr


def taylor_integrals_dipole(pes, deg=4, min_deg=1):

	print("Starting one-mode fitting...")
	nmodes,quad_order,_ = pes.dipole_onebody.shape

	f_x_1D, predicted_x_1D = _fit_onebody(pes.dipole_onebody[:,:,0], deg, min_deg = min_deg)
	f_x_arr = [f_x_1D]
	predicted_x_arr = [predicted_x_1D]

	f_y_1D, predicted_y_1D = _fit_onebody(pes.dipole_onebody[:,:,1], deg, min_deg = min_deg)
	f_y_arr = [f_y_1D]
	predicted_y_arr = [predicted_y_1D]

	f_z_1D, predicted_z_1D = _fit_onebody(pes.dipole_onebody[:,:,2], deg, min_deg = min_deg)
	f_z_arr = [f_z_1D]
	predicted_z_arr = [predicted_z_1D]

	if pes.dipole_twobody is not None:
		print("Starting two-mode fitting...")
		f_x_2D, predicted_x_2D = _fit_twobody(pes.dipole_twobody[:,:,:,:,0], deg, min_deg = min_deg)
		f_x_arr.append(f_x_2D)
		predicted_x_arr.append(predicted_x_2D)

		f_y_2D, predicted_y_2D = _fit_twobody(pes.dipole_twobody[:,:,:,:,1], deg, min_deg = min_deg)
		f_y_arr.append(f_y_2D)
		predicted_y_arr.append(predicted_y_2D)

		f_z_2D, predicted_z_2D = _fit_twobody(pes.dipole_twobody[:,:,:,:,2], deg, min_deg = min_deg)
		f_z_arr.append(f_z_2D)
		predicted_z_arr.append(predicted_z_2D)

	if pes.dipole_threebody is not None:
		print("Starting three-mode fitting...")
		f_x_3D, predicted_x_3D = _fit_threebody(pes.dipole_threebody[:,:,:,:,:,:,0], deg, min_deg = min_deg)
		f_x_arr.append(f_x_3D)
		predicted_x_arr.append(predicted_x_3D)

		f_y_3D, predicted_y_3D = _fit_threebody(pes.dipole_threebody[:,:,:,:,:,:,1], deg, min_deg = min_deg)
		f_y_arr.append(f_y_3D)
		predicted_y_arr.append(predicted_y_3D)

		f_z_3D, predicted_z_3D = _fit_threebody(pes.dipole_threebody[:,:,:,:,:,:,2], deg, min_deg = min_deg)
		f_z_arr.append(f_z_3D)
		predicted_z_arr.append(predicted_z_3D)

	return f_x_arr, f_y_arr, f_z_arr

