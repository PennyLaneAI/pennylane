import numpy as np
import re
from copy import deepcopy
from sympy.parsing.sympy_parser import parse_expr 
from sympy import var, symbols
import sympy 
import pickle

from partition import get_partitions


filename = f'./n4o4anth.op'

linbrk = '='*64
param_lines = []
ham_section_lines = []

with open(filename) as file: 
    lines = [line.rstrip() for line in file]

lines_scrubbed = []
for line in lines:
    if len(line) == 0:
        continue 
    elif line[0] in ['-', '#']:
        continue
    else:
        lines_scrubbed.append(line)

var_def = True
for line in lines_scrubbed:
    if 'HAMILTONIAN-SECTION' in line:
        var_def=False
    if var_def:
        param_lines.append(line)
    else: 
        ham_section_lines.append(line)    

def parse_O(O):

    q_terms = []
    elec_idxs = []

    O_parsed = {}
    KEO = False 
    for Oi in O:
        print(Oi)
        Oi = re.sub(' +', ' ', Oi)
        Oi = Oi.split(' ')
        Oi_dof_idx = int(Oi[0].strip())
        Oi_op_form = Oi[1].strip()
        
        Oi_op_form = Oi_op_form.replace('^', '**')
        if Oi_dof_idx == 1: #Electronic DOF 
            #handle w/ states 
            elec_idxs = [int(idx) for idx in re.findall('\d+', Oi_op_form)]

        else: #Vibrational DOF
            if Oi_op_form == 'KE':
                KEO = True
            q_expr = Oi_op_form.split('**') 
            if len(q_expr) == 1: #is linear 
                q_terms.append(Oi_dof_idx)
            else:
                exponent = q_expr[1]
                for _ in range(int(exponent)):
                    q_terms.append(int(Oi_dof_idx))
    O_parsed['KEO'] = KEO
    O_parsed['VIB'] = q_terms
    O_parsed['ELEC'] = elec_idxs
    return O_parsed

def parse_hamiltonian_section(lines, verbose=True):
    """
    Given list of lines comprising the HAMILTONIAN-SECTION, 
    returns the operator structure.
    """

    hamiltonian_terms = []

    if verbose:
        print(f'\n{linbrk}\nParsing HAMILTONIAN-SECTION\n{linbrk}')

    #reads the DOF labels and corresponding indices

    dof_idxs = {}
    for line_idx, line in enumerate(lines): 
        if 'modes' in line:
            print(line)
            dof_idxs_defs = line
            dof_idxs_defs = dof_idxs_defs.split('|')[1:]
            for dof in dof_idxs_defs:
                dof_idxs[dof.strip()] = len(dof_idxs)+1
            dof_idxs_line = line_idx

    n_modes = len(dof_idxs) - 1
    max_elec_idx = 0
    #reads the operator terms for the DOFs:
    for line in lines[dof_idxs_line+1:]:
        if 'end-hamiltonian-section' not in line and 'end-operator' not in line:
            term = line.split('|')
            print(term)
            coeff = term[0].strip()
            op = parse_O([O.strip() for O in term[1:]])
            op['COEFF'] = parse_expr(coeff)
            if verbose:
                print(op)

            if len(op['ELEC']) > 0:
                if max(op['ELEC']) > max_elec_idx:
                    max_elec_idx = max(op['ELEC'])

            hamiltonian_terms.append(op)
        else:
            break
    n_states = max_elec_idx
    if verbose:
        print(f'\n{linbrk}\nFinished parsing HAMILTONIAN-SECTION\nStates: {n_states}\nModes: {n_modes}\nFound {len(hamiltonian_terms)} terms.\n{linbrk}')
    return hamiltonian_terms, n_states, n_modes


def parse_parameter_section(lines, verbose=True):
    
    param_defs = {}
    all_units = [] #make sure everything is done in the same units, otherwise need to implement conversions. 
    
    if verbose:
        print(f'Parsing PARAMETER-SECTION\n{linbrk}')
    
    for line_idx, line in enumerate(lines):
        print(line)
        if "parameter-section" in line.lower() and 'end-' not in line.lower():
            read_idx_0 = line_idx + 1
        elif "end-parameter-section" in line.lower(): 
            read_idx_1 = line_idx
    
    for line in lines[read_idx_0:read_idx_1]:
        param_label, val_w_units = line.split('=')
        val, units = [e.strip() for e in val_w_units.split(',')]
        
        param_label = parse_expr(param_label)
        val = parse_expr(val)
        all_units.append(units)
        param_defs[param_label] = val

    for p in param_defs:
        print(f'{p} = {param_defs[p]}')

    if verbose:
        print(f'{linbrk}\nFinished parsing PARAMETER-SECTION\nFound {len(param_defs)} parameters.')

        if not len(set(all_units)) == 1:
            raise ValueError('Not all units are the same! Conversions are necessary!')
        else:
            print(f'All units in {all_units[0]}.')
        print(linbrk)

    return param_defs 

def substitute_params(terms, param_defs):

    substituted_terms = []
    for term in terms:
        param = list(term['COEFF'].free_symbols)[0]
        paramval = param_defs[param]
        coeff = term['COEFF'].subs({param: paramval})
        subbed_term = deepcopy(term)
        subbed_term['COEFF'] = coeff
        substituted_terms.append(subbed_term)
    return substituted_terms


def build_arrays(terms, n_states, n_modes):
    """
    Build the arrays of coefficients for the vibronic Hamiltonian.
    """    
    def get_degrees(terms):
        degrees = []
        for term in terms:
            if len(term['VIB']) not in degrees:
                degrees.append(len(term['VIB']))
        return degrees

    degs = sorted(get_degrees(terms))
    #initiate empty arrays 
    coupling_arrays = {}

    for deg in degs:
        dim = (n_states, n_states) + (n_modes,)*deg 
        coupling_arrays[deg] = np.zeros((dim))
    omega = np.zeros(n_modes)

    for term in terms:
        if term['KEO']:
            omega[term['VIB'][0]- 2] += term['COEFF'] #get omegas first

    for term in terms:
        if not term['KEO']: #potential 
            deg = len(term['VIB'])
            q_idxs = [idx - 2 for idx in term['VIB']]
            el_idxs = [idx - 1 for idx in term['ELEC']]
            if len(el_idxs) == 0: #Acts as identity on electronic space
                if deg == 2: #quadratic 
                    if not (term['COEFF'] == omega[term['VIB'][0] - 2] /2): #skip harmonic part, doesnt get added to betas!
                        for el_idx in range(n_states):
                            idx = tuple([el_idx,el_idx] + q_idxs)
                            coupling_arrays[deg][idx] += term['COEFF'] 
                    else:
                        print('harmonic, skipping addition to beta array')
                else: #no need for harmonic check 

                    for el_idx in range(n_states):
                        idx = tuple([el_idx,el_idx] + q_idxs)
                        coupling_arrays[deg][idx] += term['COEFF'] 
                    

                        #if term['VIB'][0] == term['VIB'][1]: #remove contribution of omega/2 



            else:
                coupling_arrays[deg][tuple(el_idxs + q_idxs)] += term['COEFF'] 
                #consider h.c. 
                if len(set(el_idxs)) == 2:
                    coupling_arrays[deg][tuple(list(reversed(el_idxs)) + q_idxs)] += term['COEFF']


    return omega, coupling_arrays 

def build_symbolic(terms, n_states, n_modes, ignore_keo=True):
    """
    Build the matrix elements as sympy polynomials of the vibrational coordinates. 
    """

    matrix_elements = {}
    for i in range(n_states):
        for j in range(i, n_states):            
            matrix_elements[(i,j)] = 0.0

    for term in terms: 
        if term['KEO'] and ignore_keo:
            continue 

        monomial = term['COEFF']
        for q in term['VIB']:
            monomial = monomial * var(f'q{q - 2}')

        if len(term['ELEC']) == 2:
            matrix_elements[tuple([s - 1 for s in term['ELEC']])] += monomial
        
        else: #acts as identity in electronic: add to every diagonal matrix element. 
            for i in range(n_states):
                matrix_elements[(i,i)] += monomial
    
    return matrix_elements


def collect_monomials(matrix_elements):
    #Get a list of all operator quantities appearing across all matrix elements. 



    return 1


def get_counts(symbolic_matrix_elements):

    n_states = max([max(ij) for ij in symbolic_matrix_elements]) + 1

    counts = []
    all_monomials = []
    partitions = get_partitions(n_states=n_states, matrix_elements=symbolic_matrix_elements)
    functional_dependencies = {}

    for part in partitions:
        print(f'Partition: {part}')
        for (i,j) in part:
            if i <= j:
                functional_dependencies[i,j] = []
                h_ij = symbolic_matrix_elements[i,j]
                h_ij_monomials = sympy.expand(h_ij).as_ordered_terms()
                for monomial in h_ij_monomials:
                    functional_part = 1
                    for factor in monomial.as_ordered_factors():
                        if 'q' in str(factor): #found q dependence
                            functional_part *= factor 
                    functional_dependencies[i,j].append(functional_part) 
                    if functional_part not in all_monomials:
                        all_monomials.append(functional_part)

    for idx, part in enumerate(partitions):
        for monomial in all_monomials:
            if all([monomial not in functional_dependencies[(i,j)] for (i,j) in part if i <= j]):
                print(f'Partition {idx} does not depend on term {monomial}')
        
    
    return counts

hamiltonian_terms, n_states, n_modes = parse_hamiltonian_section(ham_section_lines)
param_assignments = parse_parameter_section(param_lines)
hamiltonian_terms_vald = substitute_params(hamiltonian_terms, param_assignments)
omega, couplings = build_arrays(hamiltonian_terms_vald, n_states, n_modes)
matrix_elements = build_symbolic(hamiltonian_terms, n_states, n_modes)


print('\nDiabatic potential matrix elements:')
for h_ij in matrix_elements:
    print(f'\n{h_ij}:\n{matrix_elements[h_ij]}\n')

print('\nOmegas: {}'.format(omega))

output = open(f'no4a_sf.pkl', 'wb')
pickle.dump([omega,couplings], output, -1)

betas = couplings[2]

output.close()
