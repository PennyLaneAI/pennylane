int i = 1;
qubit q0;

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

i = 2;

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
}

i = 0;

switch (i) {
    case 1 {
        x q0;
    }
    default {
        rx(0.1) q0;
    }
}

i = (i + 1) * 2;

switch (i) {
    case 2 {
        y q0;
    }
    default {
        rx(0.1) q0;
    }
}
