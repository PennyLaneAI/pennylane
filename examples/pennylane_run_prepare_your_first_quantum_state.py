r"""
.. _prepare_your_first_quantum_state:

Making Quantum States
=====================

In this tutorial, we use the knowledge of quantum circuits and operations we learned
in the previous tutorials and get aquainted with different kinds of
quantum states we can prepare in PennyLane. For a full list of quantum
functions available for state preparation, see :mod:`pennylane.ops.qubit` and :mod:`pennylane.ops.cv`.

"""

# first we import the essentials
import pennylane as qml
from pennylane import numpy as np

##############################################################################
#
# Qubit State Preparation
# -------------------------
#
# Let’s first look at the features PennyLane provides for state
# preparation on the ``default.qubit`` device.
#
#    The :mod:`BasisState <pennylane.ops.qubit.BasisState>`
#    function prepares the device in a single computational basis state.
#    For example, if we are working with two qubits and want to initialize
#    the device in the state
#    :math:`|10\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}`,
#    we can use the *BasisState* function. We will need to pass this state as
#    an array, i.e ``np.array([1,0])``. If we are working with three
#    qubits and want to initialize the device in the state
#    :math:`|111\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}`,
#    we will need to pass this desired state in the form of an array, i.e
#    ``np.array([1,1,1])``.
#
# Let's look at an example.

dev1 = qml.device('default.qubit', wires = 2)
    
@qml.qnode(dev1)
def make_State1(n = None):
	qml.BasisState(n, wires = [0,1])
	qml.CNOT(wires=[0, 1])
	return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

##############################################################################

nstr = np.array([1,0])
print(make_State1(n = nstr))

##############################################################################
#
# This output is expected, as applying the CNOT gate transforms
# :math:`|10\rangle` to :math:`|11\rangle` state, which results in the two Pauli-Z expectation values
# being :math:`-1` and :math:`-1`.
#
#    The :mod:`QubitStateVector <pennylane.ops.qubit.QubitStateVector>`
#    function prepares the device in the given ket vector
#    in the computational basis. This acts similar to the BasisState function,
#    except:
#
#    -  It can be used to prepare more complicated wavefunctions
#
#    -  we need to provide the whole ket vector explicitly. For example,
#       to prepare the state :math:`\mid10\rangle`, we explicitly give the
#       ket state :math:`\begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}`,
#       i.e ``np.array([0,0,1,0])``.

dev2 = qml.device('default.qubit', wires = 2)
    
@qml.qnode(dev2)
def make_State2(s = None):
	qml.QubitStateVector(s, wires = [0,1])
	return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

##############################################################################
#
# Let’s use this function to make the entangled state
# :math:`|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle-i|11\rangle)`

state = np.array([1+0.j, 0.j, 0.j, 0-1.j])*np.sqrt(2)
print(state)

##############################################################################

print(make_State2(s = state))

##############################################################################
#
# As we make local :math:`\langle\hat{\sigma}_z\rangle` measurements on
# the reduced states of :math:`|\psi\rangle`, each subsystem is completely
# mixed and local expectation values average to zero.
#
# Gaussian State Preparation
# ------------------------------
#
# Let’s look at the features PennyLane provides for state preparation on
# the ``default.gaussian`` device.
#
#    The :mod:`CoherentState <pennylane.ops.cv.CoherentState>`
#    function prepares a coherent state with the given displacement
#    magnitude :math:`| \alpha |` and angle :math:`\phi`.

dev3 = qml.device('default.gaussian', wires = 1)
    
@qml.qnode(dev3)
def make_State3(alpha_mag, phi):
	qml.CoherentState(alpha_mag, phi, wires = 0)
	return qml.expval(qml.NumberOperator(0))

##############################################################################

print(make_State3(2, 0))

##############################################################################
#
# This is expected, as :math:`\langle\hat{n}\rangle = |\alpha|^{2}`
#
#    The :mod:`DisplacedSqueezedState <pennylane.ops.cv.DisplacedSqueezedState>`
#    function prepares a displaced squeezed state in the phase space by
#    applying a displacement operator followed by a squeezing operator -
#    :math:`D(\alpha)S(z)|0\rangle` - like we saw in an example in
#    :ref:`previous <get_to_know_the_operations>` tutorial.
#
#    The :mod:`GaussianState <pennylane.ops.cv.GaussianState>`
#    function prepares each qumode into a Gaussian state with the given
#    parameters. It requires two input arguments:
#
#    -  a concatenated vector of the mean position and momentum values for
#       all modes. For example, to prepare two Gaussian modes, we need to
#       input the vector
#       :math:`[\langle x\rangle_0,\langle x\rangle_1,\langle p\rangle_0,\langle p\rangle_1]`
#
#    -  the covariance matrix

dev4 = qml.device('default.gaussian', wires = 2)
    
@qml.qnode(dev4)
def make_State4(r, V):
	qml.GaussianState(r, V, wires = [0,1])
	return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

##############################################################################

mean_array = np.array([2, 2, 0, 0])
Cov_mat = np.eye(4)
print(make_State4(mean_array, Cov_mat))

##############################################################################
#
# **NOTE:** We used :math:`\langle x\rangle=2` and
# :math:`\langle p\rangle=0`. As ``NumberOperator`` returns the expectation
# value of the number operator :math:`\langle \hat{n} \rangle`, we have to
# use the x-representation of Coherent states to calculate the mean
# position and momentum:
#
# .. math:: \langle x\rangle = \sqrt{\frac{2\hbar}{mw}} \Re(\alpha)
#
# .. math:: \langle p\rangle = \sqrt{2\hbar mw} \Im(\alpha)
# .. note::
#     In PennyLane, it is assumed that :math:`\hbar = 2` and :math:`mw=1`.
