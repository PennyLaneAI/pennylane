qubit[1] q;

def f3(qubit q2) {
    h q2;
}

def f2(qubit q2) {
    x q2;
    f3(q2);
}

def f1(qubit q1) {
    y q1;
    f2(q1);
}

f1(q[0]);