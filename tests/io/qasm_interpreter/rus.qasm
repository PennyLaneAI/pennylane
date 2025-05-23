/*
 * Repeat-until-success circuit for Rz(theta),
 * cos(theta-pi)=3/5, from Nielsen and Chuang, Chapter 4.
 */
include "stdgates.inc";

/*
 * Applies identity if out is 01, 10, or 11 and a Z-rotation by
 * theta + pi where cos(theta)=3/5 if out is 00.
 * The 00 outcome occurs with probability 5/8.
 */
def segment(qubit[2] anc, qubit psi) -> bit[2] {
  bit[2] b;
  reset anc;
  h anc;
  ccx anc[0], anc[1], psi;
  s psi;
  ccx anc[0], anc[1], psi;
  z psi;
  h anc;
  measure anc -> b;
  return b;
}

qubit input_qubit;
qubit[2] ancilla;
bit[2] flags = "11";
bit output_qubit;

reset input_qubit;
h input_qubit;

// braces are optional in this case
while(int[2](flags) != 0) {
  flags = segment(ancilla, input_qubit);
}
rz(pi - arccos(3 / 5)) input_qubit;
h input_qubit;
output_qubit = measure input_qubit;  // should get zero
