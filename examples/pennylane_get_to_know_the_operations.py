r"""
.. _get_to_know_the_operations:


Get to know the Operations
==========================

In this tutorial, we use the quantum circuits we learned in the previous
tutorial to get aquainted with some of the quantum operations we can use
in PennyLane. For a full list of quantum operations, see :mod:`supported operations <pennylane.ops>` 
and :mod:`supported expectations <pennylane.expval>`.

"""

# let's first import the essentials
import pennylane as qml
from pennylane import numpy as np


##############################################################################
#
# Discrete Variable Quantum Operations
# --------------------------------------
#
# 1. RX, PauliZ
# ^^^^^^^^^^^^^^
#
# Let’s look at the example we saw in the first tutorial in deatil.
#
#   `RX <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html#pennylane.ops.qubit.RX>`__
#   function in PennyLane applies :math:`e^{-i\phi\frac{\sigma_x}{2}}`
#   phase operation on the specified single qubit.
#
#   `expval.PauliZ <https://pennylane.readthedocs.io/en/latest/code/expval/qubit.html#pennylane.expval.qubit.PauliZ>`__
#   calculates the expectation value of the Pauli-Z operator on the given
#   quantum state.

dev1 = qml.device('default.qubit', wires = 1)
    
@qml.qnode(dev1)
def circuit(param):
    qml.RX(param, wires = 0)
    return qml.expval.PauliZ(0)

##############################################################################

print(circuit(np.pi))
##############################################################################
#
# Let's explicitly look at what is happening behind the scence when we call
# the ``circuit`` function with argument :math:`\pi`:
#
# .. math::  RX(\pi)=\begin{pmatrix} \cos(\frac{\pi}{2}) &  -i\sin(\frac{\pi}{2}) \\ -i\sin(\frac{\pi}{2}) & \cos(\frac{\pi}{2}) \end{pmatrix}
#
# .. math::  RX(\pi)|0\rangle=\begin{pmatrix} \cos(\frac{\pi}{2}) &  -i\sin(\frac{\pi}{2}) \\ -i\sin(\frac{\pi}{2}) & \cos(\frac{\pi}{2}) \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ -i\end{pmatrix} 
#
# .. math:: \langle\begin{pmatrix} 0 & i\end{pmatrix}|\hat{\sigma}_z|\begin{pmatrix} 0 \\ -i\end{pmatrix}\rangle = -1
#
# 2. H, CNOT, Multiple Expectation Values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#    `Hadamard <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html#pennylane.ops.qubit.Hadamard>`__
#    operator takes in one qubit. It maps :math:`|0\rangle` to
#    :math:`|+\rangle` and :math:`|1\rangle` to :math:`|-\rangle`.
#
#    `CNOT <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html#pennylane.ops.qubit.CNOT>`__
#    is the two-qubit Controlled-Not operator.
#
# PennyLane supports the return of multiple expectation values; up to
# one per wire.

dev2 = qml.device('default.qubit', wires = 2)
    
@qml.qnode(dev2)
def entangle_local():
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0,1])
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1)

##############################################################################
#
# This produces the completely entangled Bell state
# :math:`|\Phi^{+}\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)`.

print(entangle_local())
##############################################################################
#
# In ``entangle_local`` function, expectation values of Pauli-Z are
# measured locally, i.e. this circuit is evaluating :math:`\langle\sigma_z\rangle _0 = \langle\sigma_z \otimes \hat{I} \rangle`,
# :math:`\langle\sigma_z\rangle _1 = \langle\hat{I}\otimes\sigma_z\rangle`
# and **not** :math:`\langle\sigma_z \otimes \sigma_z \rangle _{01}`
#
# As a result, the reduced states of :math:`|\Phi^{+}\rangle` on each
# subsystem are completely mixed and local expectation values average to
# zero.
#
# In order to measure :math:`\langle\sigma_z \otimes \sigma_z \rangle _{01}`, we can use the
# Hermitian operator shown below.
#
# 3. Hermitian Operator
# ^^^^^^^^^^^^^^^^^^^^^
#
#    `expval.Hermitian <https://pennylane.readthedocs.io/en/latest/code/expval/qubit.html#pennylane.expval.qubit.Hermitian>`__
#    lets us calculate the expectation value of any customized Hermitian
#    operator.

dev3 = qml.device('default.qubit', wires=2)
    
@qml.qnode(dev3)
def entangle_global(A):
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0,1])
    return qml.expval.Hermitian(A, wires = [0,1])
##############################################################################

# Pauli Z operator
sigma_Z = np.array([[1,0], [0, -1]])
    
# Define the Hermitian matrix of tensor product of Pauli Z operators
sigma_ZZ = np.kron(sigma_Z, sigma_Z)
    
# call the function with this operator
print(entangle_global(sigma_ZZ))

##############################################################################
#
# This makes sense as :math:`\langle \Phi^{+}|\sigma_z \otimes \sigma_z|\Phi^{+}\rangle = 1`.
# Both spins are either up or down.
#
# .. note::
#
#     A better practice will be to explicitly mention ``A=None`` in the
#     arguments of ``entangle_global`` function to inform PennyLane **not** to 
#     use this matrix as a differential argument. See the section on ``Keyword arguments`` in
#     the :ref:`Advanced usage <advanced_usage>` tutorial.
#
# Continuous Variable Quantum Operations
# ----------------------------------------
#
# 1. Displacement Operator, MeanPhoton
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#    `Displacement <https://pennylane.readthedocs.io/en/latest/code/ops/cv.html#pennylane.ops.cv.Displacement>`__
#    operator for the Gaussian state in the phase space. It applies
#    :math:`exp(\alpha\hat{a}^{\dagger}-\alpha^{*}\hat{a}^{\dagger})` to
#    the Gaussian state, where :math:`\alpha = ae^{i\phi}` is a complex
#    number.
#
#    `expval.MeanPhoton <https://pennylane.readthedocs.io/en/latest/code/expval/cv.html#pennylane.expval.cv.MeanPhoton>`__
#    returns the expectation value of the Photon number operator
#    :math:`\langle \hat{n} \rangle`.

dev4 = qml.device('default.gaussian', wires = 1)
    
@qml.qnode(dev4)
def displace_func(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval.MeanPhoton(0)

##############################################################################

print(displace_func(1, 0))

##############################################################################
#
# This makes sense as :math:`D(1,0)| 0 \rangle=| 1 \rangle`. Hence
# we measure one Photon in the system.
#
# 2. Heisenberg Uncertainty for squeezed states
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#    `Squeezing <https://pennylane.readthedocs.io/en/latest/code/ops/cv.html#pennylane.ops.cv.Squeezing>`__
#    operator implements phase space squeezing.
#
#    `expval.X <https://pennylane.readthedocs.io/en/latest/code/expval/cv.html#pennylane.expval.cv.X>`__
#    and
#    `expval.P <https://pennylane.readthedocs.io/en/latest/code/expval/cv.html#pennylane.expval.cv.P>`__
#    calculate the expectation value of position and momentum in the phase
#    space, respectively.
#
#    `expval.PolyXP <https://pennylane.readthedocs.io/en/latest/code/expval/cv.html#pennylane.expval.cv.PolyXP>`__
#    can be used to calculate higher order X/P expectation values. It
#    requires a matrix as input to determine what expectations to
#    calculate and returns a sum of all the activated terms. For example:
#    for the following matrix
#
#    .. math:: A=\begin{pmatrix} 0 & 1 & 0 \\ 1 & 1 & 0\\ 0 & 0 & 1 \end{pmatrix}
#
#    PolyXP will return
#    :math:`\langle\hat{x}\rangle + \langle\hat{x}^2 \rangle + \langle\hat{p}^2\rangle`
#
# .. note::
#
#     In PennyLane, it is assumed that :math:`\hbar = 2`
#

# We can create multiple QNodes on the same device 
dev5 = qml.device('default.gaussian', wires = 1)
    
@qml.qnode(dev5)
def before_squeezing_X(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval.X(0)
    
@qml.qnode(dev5)
def before_squeezing_X2(mag_alpha, phase_alpha, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval.PolyXP(q, wires = 0)
    
@qml.qnode(dev5)
def before_squeezing_P(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval.P(0)
    
@qml.qnode(dev5)
def before_squeezing_P2(mag_alpha, phase_alpha, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval.PolyXP(q, wires = 0)

##############################################################################
#
# .. note::
#
#     We explicitly mention ``q=None`` in the arguments to inform PennyLane
#     **not** to use it as a differential argument. See the section on ``Keyword arguments`` in
#     the :ref:`Advanced usage <advanced_usage>` tutorial.
#

# let's make the corresponding matrix for PolyXP to get <X^2> 
q_X2 = np.array([[0,0,0], [0,1,0], [0,0,0]])
print(q_X2)

##############################################################################
mean_X = before_squeezing_X(1, 0) 
print(mean_X)

##############################################################################

# explicitly state q when calling the function
mean_X2 = before_squeezing_X2(1, 0, q = q_X2)
print(mean_X2)

##############################################################################
#
# Now we can calculate the satndard deviation using
# :math:`\sqrt{\langle\hat{x}^2\rangle - \langle\hat{x}\rangle^2}`

# calculate the standard deviation in Position
std_X = np.sqrt((mean_X2-(mean_X)**2))
print(std_X)

##############################################################################

# let's make the corresponding matrices for PolyXP to get <P^2>
q_P2 = np.array([[0,0,0], [0,0,0], [0,0,1]])
print(q_P2)

##############################################################################
    
mean_P = before_squeezing_P(1, 0) 
print(mean_P)

##############################################################################
    
# explicitly state q when calling the function
mean_P2 = before_squeezing_P2(1, 0, q = q_P2)
print(mean_P2)

##############################################################################
# Now we can calculate the satndard deviation using
# :math:`\sqrt{\langle\hat{p}^2\rangle - \langle\hat{p}\rangle^2}`

# calculate the standard deviation in Momentum
std_P = np.sqrt((mean_P2-(mean_P)**2))
print(std_P)

##############################################################################
#
# This makes sense as for a Gaussian state, position and momentum can be
# measured with equal uncertainty; :math:`\sigma_x\sigma_p =1`
#
# Now, let’s squeeze the displaced state we have seen so far with
# *squeezing magnitude* 1 and *squeezing phase* 0 :

dev6 = qml.device('default.gaussian', wires = 1)
    
@qml.qnode(dev6)
def after_squeezing_X(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval.X(0)
    
@qml.qnode(dev6)
def after_squeezing_X2(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval.PolyXP(q, wires = 0)
    
@qml.qnode(dev6)
def after_squeezing_P(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval.P(0)
    
@qml.qnode(dev6)
def after_squeezing_P2(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval.PolyXP(q, wires = 0)

##############################################################################

mean_X_S = after_squeezing_X(1, 0, 1, 0, q = q_X2) 
print(mean_X_S)

##############################################################################
    
mean_X2_S = after_squeezing_X2(1, 0, 1, 0, q = q_X2)
print(mean_X2_S)

##############################################################################

mean_P_S = after_squeezing_P(1, 0, 1, 0, q = q_P2) 
print(mean_P_S)

##############################################################################
    
mean_P2_S = after_squeezing_P2(1, 0, 1, 0, q = q_P2)
print(mean_P2_S)

##############################################################################

# calculate the standard deviation in position after squeezing
std_X_S = np.sqrt((mean_X2_S-(mean_X_S)**2))
print(std_X_S)

##############################################################################
    
# calculate the standard deviation in momentum after squeezing
std_P_S = np.sqrt((mean_P2_S-(mean_P_S)**2))
print(std_P_S)

##############################################################################
#
# Hence, **after** squeezing the uncertainty in Position has decreased and
# uncertainty in momentum has increased as shown below:
#
# .. figure:: ../../examples/figures/squeeze1.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);
#

# check that the Heisenberg Principle still holds; std_p*std_x =1
print(std_P_S * std_X_S)

##############################################################################
#
# 3. Beam Splitter
# ^^^^^^^^^^^^^^^^
#
#    `BeamSplitter <https://pennylane.readthedocs.io/en/latest/code/ops/cv.html#pennylane.ops.cv.Beamsplitter>`__
#    operator acts on two input modes with characterized reflection
#    :math:`r=e^{i\phi}\sin(\theta)` and transmission
#    :math:`t=\cos(\theta).`
#

dev7 = qml.device('default.gaussian', wires = 2)
    
# without Beamsplitter
@qml.qnode(dev7)
def func_NO_BS(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval.MeanPhoton(0), qml.expval.MeanPhoton(1)
    
# with Beamsplitter
@qml.qnode(dev7)
def func_BS(mag_alpha, phase_alpha, theta, phi):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Beamsplitter(theta, phi, wires = [0,1])
	return qml.expval.MeanPhoton(0), qml.expval.MeanPhoton(1)

##############################################################################

# let's try a BS with r=1,t=0
print("Before BS:",func_NO_BS(1, 0))
print("After BS:",func_BS(1, 0, np.pi/2, 0))

##############################################################################
#
# This makes sense as after the Displacement operator acts on the first
# wire, we have one photon in this wire. This is reflected to the second
# wire when both are incident on this Beam Splitter with :math:`r=1`,
# :math:`t=0`. The figure below illustrates this.
#
# .. figure:: ../../examples/figures/BSexample.png
#     :align: center
#     :width: 60%
#     :target: javascript:void(0);


