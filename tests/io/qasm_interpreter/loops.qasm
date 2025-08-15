qubit q0;
bit[2] bits = "10";

// while loop
while(int[2](bits) != 0) {
  z q0;
  bits = 0;
}

// loop over every even integer from 0 to 20 using a range, and call a
// subroutine with that value.
for int i in [0:2:20]
   x q0;

// high precision typed loop variable
for uint[64] i in [4294967296:4294967306] {
   rx(i) q0;
}

// Loop over an array of floats.
array[float[64], 4] my_floats = {1.2, -3.4, 0.5, 9.8};
for float[64] f in my_floats {
   ry(f) q0;
}

for float[64] f in {1.2, -3.4, 0.5, 9.8} {
   ry(f) q0;
}

// Loop over a register of bits.
bit[6] register = "011011";
for bit b in register {
    rz(0.1) q0;
}

let alias = register[0:5];
for bit b in alias {
    y q0;
}

for bit b in alias {
    y q0;
    break;
}

for bit b in alias {
    z q0;
    continue;
    y q0;
}

for uint[64] i in [4294967296:4294967306] {
   rx(i) q0;
   continue;
   ry(i) q0;
}

while(int[2](bits) != 1) {
  z q0;
  bits = 1;
  continue;
  x q0;
}
