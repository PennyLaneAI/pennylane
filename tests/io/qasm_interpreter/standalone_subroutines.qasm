qubit q0;
qubit anc;

def second(qubit q) {
    h q;
    y q;
}

def first(qubit q0) -> bit {
    second(q0);
}

first(q0);
