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

qubit[2] p;

def l3(qubit[2] p2) {
    h p2[1];
}

def l2(qubit[2] p2) {
    x p2[0];
    l3(p2);
}

def l1(qubit[2] p1) {
    y p1[1];
    l2(p1);
}

l1(p);
