import numpy as np
import pennylane as qml
#
# U = np.array([[1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(6)], [np.sqrt(2/29), np.sqrt(3/29), -2 * np.sqrt(6/29)], [-5/np.sqrt(58), 7/np.sqrt(87), 1/np.sqrt(174)]])
#
#
# print(np.linalg.norm(U[1]))

# inv_sqrt_3 = 1 / np.sqrt(3)
# gellMann_8_coeffs = np.array([inv_sqrt_3, -1, inv_sqrt_3, 1, inv_sqrt_3, -1]) / 2
# gellMann_8_obs = [qml.GellMann(0, i) for i in [1, 2, 4, 5, 6, 7]]

# obs = qml.expval(qml.Hamiltonian(gellMann_8_coeffs, gellMann_8_obs))
#
# obs.matrix()
#
# ham = qml.Hamiltonian()

# from scipy.stats import unitary_group
# X = qml.PauliX.compute_matrix()
# Y = qml.PauliX.compute_matrix()
# Z = qml.PauliX.compute_matrix()
# I = np.eye(2)



#gellMann_8_obs = [qml.Hermitian(sum([np.random.rand()*m for m in [X, Y, Z, I]]), wires=0)]
# for obs in gellMann_8_obs:
#     diagm = obs.diagonalizing_gates()[0].matrix()
#     diagm_adj = np.conj(diagm).T
#     obsm = obs.matrix()
#
#     print(np.round(diagm@obsm@diagm_adj, 5), "\n")
#     print(np.round(diagm_adj @ diagm, 5), "\n===============================================================\n")
had = qml.THadamard.compute_matrix()
ahad = np.conj(had).T
G8 = qml.GellMann.compute_matrix(8)
G3 = qml.GellMann.compute_matrix(3)


# print(np.round(had@G8@ahad, 5))
#
# print()
# print(np.round(had@G3@ahad, 5))

inv_sqrt_3 = 1 / np.sqrt(3)
inv_sqrt_3_i = inv_sqrt_3 * 1j
#
gellMann_3_equivalent = (
            np.array(
                [[0, 1+inv_sqrt_3_i, 1-inv_sqrt_3_i],
                 [1-inv_sqrt_3_i, 0, 1 + inv_sqrt_3_i],
                 [1+inv_sqrt_3_i, 1-inv_sqrt_3_i, 0]]
            )
            / 2
        )
gellMann_8_equivalent = (
                np.array(
                    [[0, (inv_sqrt_3 - 1j), (inv_sqrt_3 + 1j)],
                     [inv_sqrt_3 + 1j, 0, inv_sqrt_3 - 1j],
                     [inv_sqrt_3 - 1j, inv_sqrt_3 + 1j, 0]]
                )
                / 2
        )

dg = qml.THermitian(gellMann_8_equivalent, 0).diagonalizing_gates()[0].matrix()
# dga = np.conj(dg).T
# print(np.round(dg@gellMann_8_equivalent@dga, 5))
#
# print(np.abs((dg@had@(np.array([1/2,1/3,1/6])**(1/2))))**2)
#print(qml.GellMann(0, 1).diagonalizing_gates()[0].matrix())

# print(np.round(dg@had, 4))
obs = np.diag([1, 2, 3])
print(np.round(had@obs@ahad, 4))

obs = np.diag([-2, -1, 1])

non_commuting_obs_two = np.array(
            [
                [-2/3, -2/3 + inv_sqrt_3_i, -2/3 - inv_sqrt_3_i],
                [-2/3 - inv_sqrt_3_i, -2/3, -2/3 + inv_sqrt_3_i],
                [-2/3 + inv_sqrt_3_i, -2/3 - inv_sqrt_3_i, -2/3],
            ]
        )

print(np.round(had@obs@ahad, 4))

print(np.allclose(had@obs@ahad, non_commuting_obs_two))
# print(np.allclose(had@G8@ahad, gellMann_8_equivalent))
# #print(had@G8@ahad)
# print(np.allclose(ahad, dg))
import jax
print(jax.numpy.array([jax.numpy.nan, 1., 2.]))



