qubit q0;
qubit ancilla;
bit bits = "1";

def random(qubit anc, qubit q) -> bit[2] {
  bit b = "0";
  h anc;
  measure anc -> b;
  return b != 0;
}

// this works
if (bits) {
    bits = random(ancilla, q0);
}

// example loop... not supported!
while(bits) {
    bits = random(ancilla, q0);
}
