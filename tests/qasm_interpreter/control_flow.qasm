/*
 * A demo of the control flow features available in QASM 3.0.
 * Marcus Edwards 13/05/2025
 */

qubit input;
qubit[2] ancilla;
bit[2] bits = "10";
bit output;

def random(qubit[2] anc, qubit q) -> bit[2] {
  bit[2] b;
  reset anc;
  h anc;
  measure anc -> b;
  return b;
}

// example loop
while(int[2](bits) != 0) {
  bits = random(ancilla, input);
}

bool target = false;

// example of branching
if (target == bits[0]) {
    if (bits[1] == target) {
        break;
    }
    y input;
} else {
    continue;
}

int[32] acc = 0;

// loop over a discrete set of values
for int[32] i in {1, 5, 10} {
    acc += i;
}

// loop over every even integer from 0 to 20 using a range, and call a
// subroutine with that value.
for int i in [0:2:20]
   random(ancilla, input);

// high precision typed loop variable
for uint[64] i in [4294967296:4294967306] {
   rx(i, input);
}

// Loop over an array of floats.
array[float[64], 4] my_floats = {1.2, -3.4, 0.5, 9.8};
for float[64] f in my_floats {
   ry(f, input);
}

// Loop over a register of bits.
bit[5] register;
for bit b in register {}
let alias = register[1:3];
for bit b in alias {
    measure input -> b;
}

int i = 15;

switch (i) {
    case 1, 3, 5 {
        x input;
    }
    case 2, 4, 6 {
        y input;
    }
    case -1 {
        z input;
    }
    default {
        inv @ x input;
    }
}

end;

z input;
y input;
x input;