qubit q0;
qubit anc;

def random(qubit q0) -> bit {
  bit b = "0";
  h q0;
  measure q0 -> b;
  return b;
}

bit c = random(q0);
