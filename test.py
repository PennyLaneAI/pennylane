import pennylane as qml
from pennylane.ftqc import measure_x, measure_y
from pennylane.ftqc import diagonalize_mcms



dev = qml.device("lightning.qubit", wires=3)

# @diagonalize_mcms
# @qml.set_shots(1)
# @qml.qnode(dev, mcm_method="one-shot")
# def circ():
#     qml.Rot(1.2, 2.3, 0.2, 0)
#     qml.Rot(0.62, 0.12, 1.7, 1)
#     m1 = measure_x(0)
#     m2 = measure_y(1)
#     qml.cond(m2^m1, qml.X)(2)
#     return qml.sample(m1), qml.sample(m2)#qml.expval(qml.Z(2))#qml.expval(qml.Z(2))#, qml.sample(m1), qml.sample(m2)

# circ()

@diagonalize_mcms
@qml.set_shots(1)
@qml.qnode(dev, mcm_method="one-shot")
def circ():
    qml.Rot(1.2, 2.3, 0.2, 0)
    qml.Rot(0.62, 0.12, 1.7, 1)
    m1 = measure_x(0)
    m2 = measure_y(1)
    print("++++")
    breakpoint()
    qml.cond(m1^m2, qml.X)(2)
    print("++++")
    return  qml.sample(m2)#, qml.sample(m2)#qml.expval(qml.Z(2))#qml.expval(qml.Z(2))#, qml.sample(m1), qml.sample(m2)

print(circ())
#qs = qml.tape.make_qscript(circ)()

#print(qs.measurements)