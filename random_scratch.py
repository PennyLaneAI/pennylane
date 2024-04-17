import pennylane as qml
# from pennylane.operation import enable_new_opmath
# enable_new_opmath()


def main():
    dev = qml.device("default.qubit")
    
    @qml.qnode(dev)
    def circ():
        qml.pl_breakpoint()

        qml.X(0)

        for i in range(3):
            qml.Hadamard(i)
        
        qml.Y(1)

        qml.pl_breakpoint()

        qml.Z(0)

        return qml.expval(qml.Z(0))
    
    res = circ()
    print(res)
    return


if __name__ == "__main__":
    main()
