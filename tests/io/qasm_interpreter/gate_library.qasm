qubit q0;
qubit q1;

ch q0, q1;
cx q1, q0;
cy q0, q1;
cz q1, q0;
swap q0, q1;

cp(0.4) q0, q1;
cphase(0.4) q0, q1;
crx(0.2) q0, q1;
cry(0.1) q0, q1;
crz(0.3) q1, q0;

qubit[1] q2;

ccx q0, q2, q1;
cswap q1, q2, q0;

rx(0.9) q0;
ry(0.8) q1;
rz(1.1) q2;
p(8) q0;
phase(2.0) q1;
u1(3.3) q0;
u2(1.0, 2.0) q1;
u3(1.0, 2.0, 3.0) q2;

id q0;
h q2;
x q1;
y q2;
z q0;
s q2;
t q1;
sx q0;

ctrl @ id q0, q1;
inv @ h q2;
pow(2) @ t q1;
