import pennylane as qml
from pennylane.templates.subroutines.qramv1 import select_bucket_brigade_bus_qram 

bitstrings = ["010", "111", "110", "000"]  # 2^2 entries, m=3
dev = qml.device("default.qubit")

@qml.qnode(dev)
def bb_quantum():
    # No select (k=0). qram_wires are the 2 LSB address bits.
    qram_wires   = [0, 1]        # |i> for 4 leaves
    target_wires = [2, 3, 4]     # m=3
    bus          = 5             # single bus at the top

    # For n_k=2 → (2^2 - 1) = 3 internal nodes in level order:
    # (0,0) root; (1,0) left child; (1,1) right child
    dir_wires    = [6, 7, 8]
    portL_wires  = [9, 10, 11]
    portR_wires  = [12, 13, 14]

    # prepare an address, e.g., |10> (index 2)
    qml.BasisEmbedding(2, wires=qram_wires)

    select_bucket_brigade_bus_qram(
        bitstrings,
        select_wires=[],          # k=0
        qram_wires=qram_wires,    # n_k=2
        target_wires=target_wires,
        bus_wire=bus,
        dir_wires=dir_wires,
        portL_wires=portL_wires,
        portR_wires=portR_wires,
        mode="quantum",           # fully bb qram, no select wires
    )
    return qml.probs(wires=target_wires)

print("Quantum bucket-brigade probs:", bb_quantum())
