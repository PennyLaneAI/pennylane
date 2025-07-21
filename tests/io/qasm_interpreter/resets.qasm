qubit[10] q;

def reset_and_measure(qubit[10] q) -> bit {
  bit b = "0000000000";
  reset q;
  measure q -> b;
  return b;
}

bit a = reset_and_measure(q);
