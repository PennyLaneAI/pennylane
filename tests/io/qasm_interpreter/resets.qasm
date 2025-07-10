qubit q;
reset q;

def reset_and_measure(qubit p) -> bit {
  bit b = "0";
  reset p;
  measure p -> b;
  return b;
}

bit a = reset_and_measure(q);
