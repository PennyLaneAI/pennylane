r"""
.. _get_to_know_the_operations:


Get to know the Operations
==========================

In this tutorial, we use the quantum circuits we learned in the previous
tutorial to get aquainted with some of the quantum operations we can use
in PennyLane. For a full list of quantum operations, see :mod:`Quantum operations <pennylane.ops>` 
and :mod:`Measurements <pennylane.measure>`.

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
#   The :mod:`RX <pennylane.ops.qubit.RX>`
#   operator in PennyLane applies :math:`e^{-i\phi\frac{\sigma_x}{2}}`
#   phase operation on the specified single qubit.
#
#   The :mod:`PauliZ <pennylane.ops.qubit.PauliZ>` operator can be used as a gate and an
#   observable. 
#

dev1 = qml.device('default.qubit', wires = 1)
    
@qml.qnode(dev1)
def circuit(param):
    qml.RX(param, wires = 0)
    return qml.expval(qml.PauliZ(0))

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
# 2. H, CNOT
# ^^^^^^^^^^^^^
#
#    The :mod:`Hadamard <pennylane.ops.qubit.Hadamard>`
#    maps :math:`|0\rangle` to :math:`|+\rangle` and :math:`|1\rangle` to :math:`|-\rangle`.
#    It can be used as an operator or an observable to measure in the :math:`|\pm\rangle` basis.
#
#    :mod:`CNOT <pennylane.ops.qubit.CNOT>`
#    is the two-qubit Controlled-Not operator.
#
# PennyLane supports the return of multiple expectation values; up to
# one observable per wire.

dev2 = qml.device('default.qubit', wires = 2)
    
@qml.qnode(dev2)
def entangle_local():
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0,1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

##############################################################################
#
# Here, ``entangle_local`` function produces the completely entangled Bell state
# :math:`|\Phi^{+}\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)`. The expectation values of
# Pauli-Z are measured locally, i.e. this circuit is evaluating
# :math:`\langle\sigma_z\rangle _0 = \langle\sigma_z \otimes \hat{I} \rangle`,
# :math:`\langle\sigma_z\rangle _1 = \langle\hat{I}\otimes\sigma_z\rangle`
# and **not** :math:`\langle\sigma_z \otimes \sigma_z \rangle _{01}`
#
# As a result, the reduced states of :math:`|\Phi^{+}\rangle` on each
# subsystem are completely mixed and local expectation values average to
# zero.

print(entangle_local())
##############################################################################
#
# In order to measure :math:`\langle\sigma_z \otimes \sigma_z \rangle _{01}`, we can use the
# Hermitian operator shown below.
#
# 3. Hermitian Operator
# ^^^^^^^^^^^^^^^^^^^^^^
#
#    The :mod:`Hermitian <pennylane.ops.qubit.Hermitian>` operator
#    lets us calculate the expectation value of any custom-defined Hermitian
#    observable.

dev3 = qml.device('default.qubit', wires=2)
    
@qml.qnode(dev3)
def entangle_global(A):
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0,1])
    return qml.expval(qml.Hermitian(A, wires = [0,1]))
##############################################################################

# Pauli Z operator
sigma_Z = np.array([[1,0], [0, -1]])
    
# Define the Hermitian matrix as the tensor product of Pauli-Z operators
sigma_ZZ = np.kron(sigma_Z, sigma_Z)
    
# call the function with this operator
print(entangle_global(sigma_ZZ))

##############################################################################
#
# This outcome is expected, as :math:`\langle \Phi^{+}|\sigma_z \otimes \sigma_z|\Phi^{+}\rangle = 1`.
# Both spins are either up or down.
#
# .. note::
#
#     A better practice will be to explicitly mention ``A=None`` in the
#     arguments of ``entangle_global`` function to inform PennyLane **not** to 
#     use this matrix as a differential argument. See the section on ``Keyword arguments`` in
#     the :ref:`Advanced usage <advanced_features>` tutorial.
#
# Continuous Variable Quantum Operations
# ----------------------------------------
#
# 1. Displacement Operator, NumberOperator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#    The :mod:`Displacement <pennylane.ops.cv.Displacement>`
#    operator applies
#    :math:`exp(\alpha\hat{a}^{\dagger}-\alpha^{*}\hat{a}^{\dagger})` to
#    the given Gaussian state in the phase space, where :math:`\alpha = ae^{i\phi}` is a complex
#    number.
#
#    The :mod:`NumberOperator <pennylane.ops.cv.NumberOperator>` operator
#    returns the expectation value of the Photon number operator
#    :math:`\langle \hat{n} \rangle`.

dev4 = qml.device('default.gaussian', wires = 1)
    
@qml.qnode(dev4)
def displace_func(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval(qml.NumberOperator(0))

##############################################################################

print(displace_func(1, 0))

##############################################################################
#
# This is the expected outcome, as :math:`D(1,0)| 0 \rangle=| 1 \rangle`. Hence,
# we measure one Photon in the system.
#
# 2. Heisenberg Uncertainty for squeezed states
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#    The :mod:`Squeezing <pennylane.ops.cv.Squeezing>`
#    operator implements phase space squeezing.
#
#    The :mod:`X <pennylane.ops.cv.X>`
#    and :mod:`P <pennylane.ops.cv.P>` operators
#    return the expectation value of position and momentum in the phase
#    space, respectively.
#
#    The :mod:`PolyXP <pennylane.ops.cv.PolyXP>` operator
#    can be used to obtain higher order X/P expectation values. It
#    requires a matrix as input to determine what expectations to
#    calculate and returns a **sum** of all the activated terms. For example:
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
	return qml.expval(qml.X(0))
    
@qml.qnode(dev5)
def before_squeezing_X2(mag_alpha, phase_alpha, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval(qml.PolyXP(q, wires = 0))
    
@qml.qnode(dev5)
def before_squeezing_P(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval(qml.P(0))
    
@qml.qnode(dev5)
def before_squeezing_P2(mag_alpha, phase_alpha, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval(qml.PolyXP(q, wires = 0))

##############################################################################
#
# .. note::
#
#     We explicitly mention ``q=None`` in the arguments to inform PennyLane
#     **not** to use it as a differential argument. See the section on ``Keyword arguments`` in
#     the :ref:`Advanced usage <advanced_features>` tutorial.
#

# let's make the corresponding matrix for PolyXP to get <X^2> 
q_X2 = np.array([[0,0,0], [0,1,0], [0,0,0]])
print(q_X2)

##############################################################################
mean_X = before_squeezing_X(1, 0) 

# explicitly state q when calling the function
mean_X2 = before_squeezing_X2(1, 0, q = q_X2)

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
    
# explicitly state q when calling the function
mean_P2 = before_squeezing_P2(1, 0, q = q_P2)

##############################################################################
# Now we can calculate the satndard deviation using
# :math:`\sqrt{\langle\hat{p}^2\rangle - \langle\hat{p}\rangle^2}`

# calculate the standard deviation in Momentum
std_P = np.sqrt((mean_P2-(mean_P)**2))
print(std_P)

##############################################################################
#
# This outcome is expected, as for a Gaussian state, position and momentum can be
# measured with equal uncertainty; :math:`\sigma_x\sigma_p =1`
#
# Now, let’s squeeze the displaced state we have seen so far with
# *squeezing magnitude* 1 and *squeezing phase* 0 :

dev6 = qml.device('default.gaussian', wires = 1)
    
@qml.qnode(dev6)
def after_squeezing_X(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval(qml.X(0))
    
@qml.qnode(dev6)
def after_squeezing_X2(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval(qml.PolyXP(q, wires = 0))
    
@qml.qnode(dev6)
def after_squeezing_P(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval(qml.P(0))
    
@qml.qnode(dev6)
def after_squeezing_P2(mag_alpha, phase_alpha, mag_z, phase_z, q = None):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Squeezing(mag_z, phase_z, wires = 0)
	return qml.expval(qml.PolyXP(q, wires = 0))

##############################################################################

mean_X_Sq = after_squeezing_X(1, 0, 1, 0, q = q_X2) 
    
mean_X2_Sq = after_squeezing_X2(1, 0, 1, 0, q = q_X2)

##############################################################################

mean_P_Sq = after_squeezing_P(1, 0, 1, 0, q = q_P2) 
    
mean_P2_Sq = after_squeezing_P2(1, 0, 1, 0, q = q_P2)

##############################################################################

# calculate the standard deviation in position after squeezing
std_X_Sq = np.sqrt((mean_X2_Sq-(mean_X_Sq)**2))
print(std_X_Sq)

##############################################################################
    
# calculate the standard deviation in momentum after squeezing
std_P_Sq = np.sqrt((mean_P2_Sq-(mean_P_Sq)**2))
print(std_P_Sq)

##############################################################################
#
# Hence, **after** squeezing the uncertainty in position has decreased and the
# uncertainty in momentum has increased as shown below:
#
# .. figure:: ../../examples/figures/squeeze1.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0);
#

# check that the Heisenberg Principle still holds; std_p*std_x =1
print(std_P_Sq * std_X_Sq)

##############################################################################
# Note: We can also directly use the function :mod:`var <pennylane.measure.var>` for this example. 
#
# 3. Beam Splitter
# ^^^^^^^^^^^^^^^^
#
#    The :mod:`BeamSplitter <pennylane.ops.cv.Beamsplitter>`
#    operator acts on two input modes with characterized reflection
#    :math:`r=e^{i\phi}\sin(\theta)` and transmission
#    :math:`t=\cos(\theta).`
#

dev7 = qml.device('default.gaussian', wires = 2)
    
# without Beamsplitter
@qml.qnode(dev7)
def func_NO_BS(mag_alpha, phase_alpha):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))
    
# with Beamsplitter
@qml.qnode(dev7)
def func_BS(mag_alpha, phase_alpha, theta, phi):
	qml.Displacement(mag_alpha, phase_alpha, wires = 0)
	qml.Beamsplitter(theta, phi, wires = [0,1])
	return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

##############################################################################

# let's try a Beamsplitter with r=1,t=0
print("Before BS:",func_NO_BS(1, 0))
print("After BS:",func_BS(1, 0, np.pi/2, 0))

##############################################################################
#
# This outcome is expected, as after the Displacement operator acts on the first
# wire, we have one photon in this wire. This is reflected to the second
# wire when both are incident on the Beam Splitter with :math:`r=1`,
# :math:`t=0`. The figure below illustrates this.
#
# .. figure:: ../../examples/figures/BSexample.png
#     :align: center
#     :width: 60%
#     :target: javascript:void(0);


