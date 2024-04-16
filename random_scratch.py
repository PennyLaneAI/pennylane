import pennylane as qml
from pennylane.debugging import pl_breakpoint
# from pennylane.operation import enable_new_opmath
# enable_new_opmath()


def main():
    dev = qml.device("default.qubit")
    
    @qml.qnode(dev)
    def circ():
        # pl_breakpoint()
        
        qml.X(0)

        for i in range(3):
            qml.Hadamard(i)

        return qml.expval(qml.Z(0))
    
    res = circ()
    print(res)
    return


if __name__ == "__main__":
    main()
