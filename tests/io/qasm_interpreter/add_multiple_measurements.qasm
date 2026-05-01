qubit q0;
qubit q1;

def random(qubit q0) -> bit {
  bit b = "0";
  reset q0;
  h q0;
  measure q0 -> b;
  return b;
}

bit c = random(q0);
bit d = random(q1);

int res = c + d;
