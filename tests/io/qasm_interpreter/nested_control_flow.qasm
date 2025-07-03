bool target = false;
qubit q0;
bit[2] bits = "00";

if (target == bits[0]) {
    if (bits[1] == target) {
        for int i in [0:2:10]
            x q0;
    }
    switch (target) {
        case false {
            while(int[2](bits) != 1) {
              z q0;
              bits = 1;
              continue;
              x q0;
            }
        }
        case true {
            for float[64] f in {1.2, -3.4, 0.5, 9.8} {
               ry(f) q0;
            }
        }
        default {
            rx(0.1) q0;
        }
    }
} else {
    z q0;
}

for int i in [0:5:20] {
    x q0;
    for int i in [0:1:2]
        y q0;
}
