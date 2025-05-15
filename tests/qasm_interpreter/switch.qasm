int i = 1;
qubit[1] q0;

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
        rx(0.1) q0;
    }
}