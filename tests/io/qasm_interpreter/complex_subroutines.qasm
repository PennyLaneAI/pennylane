qubit q0;
qubit anc;
qubit[2] p;

def second(qubit q) {
    h q;
    y q;
    return "0";
}

def first(qubit q0) -> bit {
  bit b = "0";
  b = second(q0);
  return b;
}

bit c = first(q0);

def third(qubit q, float a) {
    h q;
    rx(a) q;
}

def fourth(qubit[2] p) {
    x p[0];
    y p[1];
}

third(q0, 0.1);
fourth(p);
