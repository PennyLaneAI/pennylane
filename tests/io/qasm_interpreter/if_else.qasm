bool target = false;
qubit q0;
bit[2] bits = "00";

if (target == bits[0]) {
    if (bits[1] == target) {
        x q0;
    }
    y q0;
} else {
    z q0;
}

if (target != bits[0]) {
    z q0;
}
