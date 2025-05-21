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

// int stop = 20;

// For loop with changing iterable
// for int i in [0:2:stop] {
//    s q0;
//    stop = stop - 2;
// }
