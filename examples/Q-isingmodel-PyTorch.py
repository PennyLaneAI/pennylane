import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np

dev3 = qml.device('default.qubit', wires=3)
@qml.qnode(dev3, interface='torch')
def circuit3(p1, p2):
    qml.Rot(p1[0],p1[1],p1[2], wires =1)
    qml.Rot(p2[0],p2[1],p2[2], wires =2)
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2) 

def cost(var1,var2):
#	let coupling matrix be J=[1, -1]
#   circuit1 function returns a numpy array of pauliz exp values
	spins = circuit3(var1,var2)
#	the expectation value of pauliZ is plus 1 for spin up and -1 for spin down
	energy = -(1*spins[0]*spins[1]) - (-1*spins[1]*spins[2])
	return energy


#test it for [1,-1,-1] spin configuration. Total energy for this ising model should be

#H = -1(J1*s1*s2 + J2*s2*s3) = -1 (1*1*-1  +  -1*-1*-1) = 2 

t1=torch.tensor([0, np.pi, 0])
t2=torch.tensor([0,np.pi ,0])

cost_check= cost(t1,t2)
print(cost_check)

#optimizing using PyTorch optimizer 

p1 = Variable((np.pi*torch.rand(3,dtype=torch.float64)), requires_grad=True)
p2 = Variable((np.pi*torch.rand(3,dtype=torch.float64)), requires_grad=True)
#print(p1,p2)
var_init = [p1,p2]
cost_init = cost(p1,p2)
opt = torch.optim.Adam(var_init, lr = 0.1)

steps = 100

def closure():
    opt.zero_grad()
    loss = cost(p1, p2)
    loss.backward()
    return loss


var_pt = [var_init]
cost_pt = [cost_init]

for i in range(steps):
    opt.step(closure)
    if (i + 1) % 5 == 0:
        p1n, p2n = opt.param_groups[0]['params']
        costn = cost(p1n, p2n)
        var_pt.append([p1n,p2n])
        cost_pt.append(costn)
        print('Energy after step {:5d}: {: .7f} | Angles: {}'.format(i+1,costn, [p1n, p2n]))


p1_final, p2_final = opt.param_groups[0]['params']

print(p1_final, p2_final)

print(cost(p1_final, p2_final))