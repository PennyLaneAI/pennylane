qubit q0;

switch (1) {
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

switch (2) {
    case 1 {
        x q0;
    }
    case 2 {
        y q0;
    }
    case -1 {
        z q0;
    }
}

switch (0) {
    case 1 {
        x q0;
    }
    default {
        rx(0.1) q0;
    }
}
