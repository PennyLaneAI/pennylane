qubit q0;
qubit anc;

def random(qubit q) -> bit {
  bit b = "0";
  h q;
  measure q -> b;
  return b;
}

bit c = random(q0);
bit d = "0";
measure q0 -> d;
bit e = measure q0;
measure anc;