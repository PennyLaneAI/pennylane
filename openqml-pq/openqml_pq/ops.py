class Gate(GateSpec): # pylint: disable=too-few-public-methods
    """Implements the quantum gates and observables.
    """
    def __init__(self, name, n_sys, n_par, cls=None, par_domain='R'): # pylint: disable=too-many-arguments
        super().__init__(name, n_sys, n_par, grad_method='F', par_domain=par_domain)
        self.cls = cls  #: class: pq subclass corresponding to the gate

    def execute(self, par, reg, sim):
        """Applies a single gate or measurement on the current system state.

        Args:
          par (Sequence[float]): gate parameters
          reg   (Sequence[int]): subsystems to which the gate is applied
          sim (~openqml.plugin.PluginAPI): simulator instance keeping track of the system state and measurement results
        """
        G = self.cls(*par)
        reg = tuple(reg)
        G | reg


class Observable(Gate): # pylint: disable=too-few-public-methods
    """Implements hermitian observables.

    We assume that all the observables in the circuit are consequtive, and commute.
    Since we are only interested in the expectation values, there is no need to project the state after the measurement.
    See :ref:`measurements`.
    """
    def execute(self, par, reg, sim):
        """Estimates the expectation value of the observable in the current system state.

        Args:
          par (Sequence[float]): gate parameters
          reg   (Sequence[int]): subsystems to which the gate is applied
          sim (~openqml.plugin.PluginAPI): simulator instance keeping track of the system state and measurement results
        """
        backend = type(sim.eng.backend).__name__

        if backend == 'IBMBackend' and hasattr(sim.eng, 'already_a_measurement_performed') and sim.eng.already_a_measurement_performed ==True :
            raise NotImplementedError("Only a single measurement is possible on the IBMBackend.")
        else:
            sim.eng.already_a_measurement_performed = True

        if backend == 'Simulator':
            sim.eng.flush(deallocate_qubits=False)
            if self.cls == pq.ops.X or self.cls == pq.ops.Y or self.cls == pq.ops.Z:
                expectation_value = sim.eng.backend.get_expectation_value(pq.ops.QubitOperator(str(self.cls)+'0'), reg)
                variance = 1 - expectation_value**2
            elif self.cls == AllZClass:
                expectation_value = [ sim.eng.backend.get_expectation_value(pq.ops.QubitOperator(str(pq.ops.Z)+'0'), [qubit]) for qubit in sim.reg]
                variance = [1 - e**2 for e in expectation_value]
            else:
                raise NotImplementedError("Estimation of expectation values not yet implemented for the observable {} in backend {}.".format(self.cls, backend))
        elif backend == 'ClassicalSimulator':
            sim.eng.flush(deallocate_qubits=False)
            if self.cls == pq.ops.Z:
                expectation_value = sim.eng.backend.read_bit(reg[0])
                variance = 0
            else:
                raise NotImplementedError("Estimation of expectation values not yet implemented for the observable {} in backend {}.".format(self.cls), backend)
        elif backend == 'IBMBackend':
            pq.ops.R(0) | sim.reg[0]# todo:remove this once https://github.com/ProjectQ-Framework/ProjectQ/issues/259 is resolved
            pq.ops.All(pq.ops.Measure) | sim.reg
            sim.eng.flush()
            if self.cls == pq.ops.Z:
                probabilities = sim.eng.backend.get_probabilities(reg)
                #print("IBM probabilities="+str(probabilities))
                if '1' in probabilities:
                    expectation_value = 2*probabilities['1']-1
                else:
                    expectation_value = -(2*probabilities['0']-1)
                variance = 1 - expectation_value**2
            elif self.cls == AllZClass:
                probabilities = sim.eng.backend.get_probabilities(sim.reg)
                #print("IBM all probabilities="+str(probabilities))
                expectation_value = [ ((2*sum(p for (state,p) in probabilities.items() if state[i] == '1')-1)-(2*sum(p for (state,p) in probabilities.items() if state[i] == '0')-1)) for i in range(len(sim.reg)) ]
                variance = [1 - e**2 for e in expectation_value]
            else:
                raise NotImplementedError("Estimation of expectation values not yet implemented for the observable {} in backend {}.".format(self.cls, backend))
        else:
            raise NotImplementedError("Estimation of expectation values not yet implemented for the {} backend.".format(backend))

        log.info('observable: ev: %s, var: %s', expectation_value, variance)
        return expectation_value, variance


class CNOTClass(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the CNOT gate.

    Contrary to other gates, ProjectQ does not have a class for the CNOT gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        #return pq.ops.C(pq.ops.XGate())
        return pq.ops.C(pq.ops.NOT)


class CZClass(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the CNOT gate.

    Contrary to other gates, ProjectQ does not have a class for the CZ gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        return pq.ops.C(pq.ops.ZGate())

class ToffoliClass(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the Toffoli gate.

    Contrary to other gates, ProjectQ does not have a class for the Toffoli gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        return pq.ops.C(pq.ops.ZGate(), 2)

class AllZClass(pq.ops.BasicGate): # pylint: disable=too-few-public-methods
    """Class for the AllZ gate.

    Contrary to other gates, ProjectQ does not have a class for the AllZ gate,
    as it is implemented as a meta-gate.
    For consistency we define this class, whose constructor is made to retun
    a gate with the correct properties by overwriting __new__().
    """
    def __new__(*par): # pylint: disable=no-method-argument
        return pq.ops.Tensor(pq.ops.ZGate())


#gates
H = Gate('H', 1, 0, pq.ops.HGate)
X = Gate('X', 1, 0, pq.ops.XGate)
Y = Gate('Y', 1, 0, pq.ops.YGate)
Z = Gate('Z', 1, 0, pq.ops.ZGate)
S = Gate('S', 1, 0, pq.ops.SGate)
T = Gate('T', 1, 0, pq.ops.TGate)
SqrtX = Gate('SqrtX', 1, 0, pq.ops.SqrtXGate)
Swap = Gate('Swap', 2, 0, pq.ops.SwapGate)
SqrtSwap = Gate('SqrtSwap', 2, 0, pq.ops.SqrtSwapGate)
#Entangle = Gate('Entangle', n, 0, pq.ops.EntangleGate) #This gate acts on all qubits
#Ph = Gate('Ph', 0, 1, pq.ops.Ph) #This gate acts on all qubits or non, depending on how one looks at it...
Rx = Gate('Rx', 1, 1, pq.ops.Rx) #(angle) RotationX gate class
Ry = Gate('Ry', 1, 1, pq.ops.Ry) #(angle) RotationY gate class
Rz = Gate('Rz', 1, 1, pq.ops.Rz) #(angle) RotationZ gate class
R = Gate('R', 1, 1, pq.ops.R) #(angle) Phase-shift gate (equivalent to Rz up to a global phase)
#pq.ops.AllGate , which is the same as pq.ops.Tensor, is a meta gate that acts on all qubits
#pq.ops.QFTGate #This gate acts on all qubits
#pq.ops.QubitOperator #A sum of terms acting on qubits, e.g., 0.5 * ‘X0 X5’ + 0.3 * ‘Z1 Z2’
CRz = Gate('CRz', 2, 1, pq.ops.CRz) #(angle) Shortcut for C(Rz(angle), n=1).
CNOT = Gate('CNOT', 2, 0, CNOTClass)
CZ = Gate('CZ', 2, 0, CZClass)
#Toffoli = Gate('Toffoli', 3, 0, ToffoliClass)
#pq.ops.TimeEvolution #Gate for time evolution under a Hamiltonian (QubitOperator object).
AllZ = Gate('AllZ', 1, 0, AllZClass) #todo: 1 should be replaced by a way to specify "all"

# measurements
MeasureX = Observable('X', 1, 0, pq.ops.X)
MeasureY = Observable('Y', 1, 0, pq.ops.Y)
MeasureZ = Observable('Z', 1, 0, pq.ops.Z)
MeasureAllZ = Observable('AllZ', 1, 0, AllZClass) #todo: 1 should be replaced by a way to specify "all"


classical_demo = [
    Command(X,  [0], []),
    Command(Swap, [0, 1], []),
]

demo = [
    Command(Rx,  [0], [ParRef(0)]),
    Command(Rx,  [1], [ParRef(1)]),
    Command(CNOT, [0, 1], []),
]

# circuit templates
_circuit_list = [
    Circuit(classical_demo, 'classical_demo'),
    Circuit(classical_demo+[Command(MeasureZ, [0])], 'classical_demo_ev'),
    Circuit(demo, 'demo'),
    Circuit(demo+[Command(MeasureZ, [0])], 'demo_ev0', out=[0]),
    Circuit(demo+[Command(MeasureAllZ, [0])], 'demo_ev', out=[0]),
]
