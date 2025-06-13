qubit q0;
qubit anc;

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

third(q0, 0.1);