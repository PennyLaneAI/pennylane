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