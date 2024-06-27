# import jax
import pennylane as qml
import numpy as np
# import scipy
#
# n_wires = 2
# num_qscripts = 2
# qscripts = []
# for i in range(num_qscripts):
#     unitary = scipy.stats.unitary_group(dim=3**n_wires, seed=(42 + i)).rvs()
#     op = qml.QutritUnitary(unitary, wires=range(n_wires))
#     qs = qml.tape.QuantumScript([op], [qml.expval(qml.GellMann(0, 3))])
#     qscripts.append(qs)
#
# dev = qml.devices.DefaultQutritMixed()
# program, execution_config = dev.preprocess()
# new_batch, post_processing_fn = program(qscripts)
# results = dev.execute(new_batch, execution_config=execution_config)
# print(post_processing_fn(results))
#
# @jax.jit
# def f(x):
#     qs = qml.tape.QuantumScript([qml.TRX(x, 0)], [qml.expval(qml.GellMann(0, 3))])
#     program, execution_config = dev.preprocess()
#     new_batch, post_processing_fn = program([qs])
#     results = dev.execute(new_batch, execution_config=execution_config)
#     return post_processing_fn(results)[0]
#
# jax.config.update("jax_enable_x64", True)
# print(f(jax.numpy.array(1.2)))
# print(jax.grad(f)(jax.numpy.array(1.2)))

# 1/2, 1/3, 1/6

# 1/2, 1/3, 1/6
#
# 1/2+1/3-2/6
#
# 2/6-3/6
#
#
# gellMann_8_coeffs = np.array([1/np.sqrt(3), -1, 1/np.sqrt(3), 1, 1/np.sqrt(3), -1]) / 2
# gellMann_8_obs = [qml.GellMann(0, i) for i in [1, 2, 4, 5, 6, 7]]
# H1 = qml.Hamiltonian(gellMann_8_coeffs, gellMann_8_obs).matrix()
#
# G8 = qml.GellMann.compute_matrix(8)
# Had = qml.THadamard.compute_matrix()
# aHad = np.conj(Had).T
#
# H2 = np.round(aHad@G8@Had, 5)
#
#
# # print(np.allclose(H1, H2))
# # print(H1)
# # print(H2)
# # print(np.round(H2*np.sqrt(3), 5))
# obs = aHad@G8@Had
#
# diag_gates = qml.THermitian(obs, 0).diagonalizing_gates()#[0].matrix()
# print(len(diag_gates))
# diag_gates = diag_gates[0].matrix()
#
# print(np.round((np.conj(diag_gates).T)@obs@diag_gates, 5))


def setup_state(nr_wires):
    """Sets up a basic state used for testing."""
    setup_unitary = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(3), 1 / np.sqrt(6)],
            [np.sqrt(2 / 29), np.sqrt(3 / 29), -2 * np.sqrt(6 / 29)],
            [-5 / np.sqrt(58), 7 / np.sqrt(87), 1 / np.sqrt(174)],
        ]
    ).T
    qml.QutritUnitary(setup_unitary, wires=0)
    qml.QutritUnitary(setup_unitary, wires=1)
    if nr_wires == 3:
        qml.TAdd(wires=(0, 2))


dev = qml.device(
        "default.qutrit.mixed",
        wires=2,
        damping_measurement_gammas=(0.2, 0.1, 0.4),
        trit_flip_measurement_probs=(0.1, 0.2, 0.5),
    )
# Create matricies for the observables with diagonalizing matrix :math:`THadamard^\dag`
inv_sqrt_3_i = 1j / np.sqrt(3)
non_commuting_obs_one = np.array(
    [
        [0, -1 + inv_sqrt_3_i, -1 - inv_sqrt_3_i],
        [-1 - inv_sqrt_3_i, 0, -1 + inv_sqrt_3_i],
        [-1 + inv_sqrt_3_i, -1 - inv_sqrt_3_i, 0],
    ]
)
non_commuting_obs_one /= 2

@qml.qnode(dev)
def circuit():
    setup_state(2)

    qml.THadamard(wires=0)
    qml.THadamard(wires=1, subspace=(0, 1))

    return qml.expval(qml.THermitian(non_commuting_obs_one, 0))





print(my_test())






