import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
#print(tf.executing_eagerly())

import pennylane as qml
from pennylane import numpy as np

dev2 = qml.device('default.qubit', wires=3)
@qml.qnode(dev2, interface='tfe')
def circuit2(p1, p2):
        qml.Rot(p1[0],p1[1],p1[2], wires =1)
        qml.Rot(p2[0],p2[1],p2[2], wires =2)
        return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2) 

def cost(var):
#	let coupling matrix be J=[1, -1]
#   circuit1 function returns a numpy array of pauliz exp values
	spins = circuit2(var[0],var[1])
#	the expectation value of pauliZ is plus 1 for spin up and -1 for spin down
	energy = -(1*spins[0]*spins[1]) - (-1*spins[1]*spins[2])
	return energy

t1 = tfe.Variable(tf.random_uniform([3],0,np.pi, dtype = tf.float64))
t2 = tfe.Variable(tf.random_uniform([3],0,np.pi, dtype = tf.float64))
var_init = [t1,t2]
cost_init = cost(var_init)
print(cost_init)

#optimize using tensorflow optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
steps = 100

var = var_init
var_tf = [var]
cost_tf = [cost_init]

for i in range(steps):
    with tf.GradientTape() as tape:
        loss = cost([t1,t2])
        grads = tape.gradient(loss, [t1, t2])

    opt.apply_gradients(zip(grads, [t1, t2]), global_step=tf.train.get_or_create_global_step())
    if (i + 1) % 5 == 0:
            var_tf.append([t1,t2])
            cost_tf.append (cost([t1,t2]))
            print('Energy after step {:5d}: {: .7f} | Angles: {}'.format(i+1,cost([t1,t2]), [t1,t2]))


print(t1, t2)

print(cost([t1, t2]))