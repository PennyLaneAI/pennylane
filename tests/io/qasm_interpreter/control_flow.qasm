/*
 * A demo of the control flow features available in QASM 3.0.
 * Marcus Edwards 13/05/2025
 */

qubit[1] q0;
qubit[2] ancilla;
bit[2] bits = "10";
bit o;

def random(qubit[2] anc, qubit q) -> bit[2] {
  bit[2] b;
  reset anc;
  h anc;
  measure anc -> b;
  return b;
}

// example loop
while(int[2](bits) != 0) {
  bits = random(ancilla, q0);
}

bool target = false;

// example of branching
while(o == false) {
    if (target == bits[0]) {
        if (bits[1] == target) {
            break;
        }
        y q0;
        measure q0 -> o;
    } else {
        continue;
    }
}

int[32] acc = 0;

// loop over every even integer from 0 to 20 using a range, and call a
// subroutine with that value.
for int i in [0:2:20]
   random(ancilla, q0);

// high precision typed loop variable
for uint[64] i in [4294967296:4294967306] {
   rx(i, q0);
}

// Loop over an array of floats.
array[float[64], 4] my_floats = {1.2, -3.4, 0.5, 9.8};
for float[64] f in my_floats {
   ry(f, q0);
}

// Loop over a register of bits.
bit[5] register = "011011";
for bit b in register {}
let alias = register[1:3];
for bit b in alias {
    measure q0 -> b;
}

int i = 15;

switch (i) {
    case 1 {
        x q0;
    }
    case 2 {
        y q0;
    }
    case -1 {
        z q0;
    }
    default {
        inv @ x q0;
    }
}

end;

z q0;
y q0;
x q0;