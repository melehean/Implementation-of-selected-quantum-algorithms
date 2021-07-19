import pennylane as qml
import matplotlib.pyplot as plt 
import functools
import pyscf
from pennylane import qchem
from pennylane import numpy as np

ATOM_1 = 'H'
ATOM_2 = 'H'
GEOMETRY_FILE = 'h2.xyz'
ELECTRON_NUMBER = 2

def changeGeometryFile(distance):
    with open(GEOMETRY_FILE, "r") as file:
        data = file.readlines()

    for index, line in enumerate(data):
        if line.startswith(ATOM_1):
                coordinates = line.split(' ')
                coordinates[-1] = str(distance) + '\n'
                newLine = ' '
                newLine = newLine.join(coordinates)
                data[index] = newLine
                break

    with open(GEOMETRY_FILE, "w") as file:
        file.writelines(data)

def getMoleculeHamiltonianCircuit(distance, mapping='jordan_wigner'):
    changeGeometryFile(distance)
    symbols, coordinates = qchem.read_structure(GEOMETRY_FILE)

    hamiltonian, qubitsNumber = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=0,
        mult=1,
        basis='sto-3g',
        active_electrons=2,
        active_orbitals=2,
        mapping=mapping
    )

    return hamiltonian, qubitsNumber

def getExactEnergy(distance):
    molecule = pyscf.M(
    atom = 'H 0 0 0; H 0 0 ' + str(distance),  # in Angstrom
    basis = 'sto-3g')
    
    hartreeFockMethod = molecule.RHF().run() # restricted Hartree-Fock
    exactSolver = pyscf.fci.FCI(hartreeFockMethod)
    return exactSolver.kernel()[0]

def default(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

def basic(params, wires):
    qml.BasisState(np.array([0, 0, 0, 0], requires_grad=False), wires=wires)
    for i in wires:
        qml.RX(params[i][0], wires=i)
        qml.RY(params[i][1], wires=i)
        qml.RZ(params[i][2], wires=i)

def getCostFunction(hamiltonian, device, qubitsNumber, ansatzName="default"):
    np.random.seed(0)
    if ansatzName == "default":
        params = np.random.normal(0, np.pi, (qubitsNumber, 3))  
        return qml.ExpvalCost(default, hamiltonian, device), params
    if ansatzName == "basic":
        params = np.random.normal(0, np.pi, (qubitsNumber, 3))  
        return qml.ExpvalCost(basic, hamiltonian, device), params
    if ansatzName == 'usscd':
        initial_state = qml.qchem.hf_state(ELECTRON_NUMBER, qubitsNumber)
        singles, doubles = qml.qchem.excitations(ELECTRON_NUMBER, qubitsNumber)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        ansatz = functools.partial(
            qml.templates.UCCSD, init_state=initial_state, s_wires=s_wires, d_wires=d_wires
        )
        params = np.random.normal(0, np.pi, len(s_wires) + len(d_wires))
        return qml.ExpvalCost(ansatz, hamiltonian, device), params

def getOptimizer(optimizerName="GradientDescent", stepSize=0.01):
    if optimizerName == "GradientDescent":
        return qml.GradientDescentOptimizer(stepsize=stepSize)
    if optimizerName == "Adam":
        return qml.AdamOptimizer(stepsize=stepSize)
    if optimizerName == "NesterovMomentum":
        return qml.NesterovMomentumOptimizer(stepsize=stepSize)
    if optimizerName == "Adagrad":
        return qml.AdagradOptimizer(stepsize=stepSize)

if __name__ == '__main__':
    deviceName = "default.qubit" #qiskit.aer
    ansatzName = "usscd"


    optimizer = getOptimizer(optimizerName="GradientDescent", stepSize=0.4)

    distances = np.linspace(0.1, 3.0, 30)
    energies = []
    exactEnergies = []
    accuracies = []

    distance = 0.75
    hamiltonian, qubitsNumber = getMoleculeHamiltonianCircuit(distance, 'jordan_wigner')
    print("Number of qubits = ", qubitsNumber)
    print()
    print("Hamiltonian for distance ", distance)
    print(hamiltonian)

    for distance in distances:
        hamiltonian, qubitsNumber = getMoleculeHamiltonianCircuit(distance, 'jordan_wigner')
        device = qml.device(deviceName, wires=qubitsNumber)
        costFunction, params = getCostFunction(hamiltonian, device, qubitsNumber, ansatzName)

        max_iterations = 100

        for n in range(max_iterations):
            params, prev_energy = optimizer.step_and_cost(costFunction, params)
            energy = costFunction(params)

        exactEnergy = getExactEnergy(distance)
        accuracy = exactEnergy-energy

        print('Final value of the ground-state energy = {:.8f} Ha for distance {:.4f}'.format(energy,distance))
        print('Exact FCI value of the ground-state energy = {:.8f} Ha for distance {:.4f}'.format(exactEnergy,distance))
        print('Accuracy = {:.8f}'.format(np.abs(accuracy)))
        energies.append(energy)
        exactEnergies.append(exactEnergy)
        accuracies.append(accuracy)
    
    plt.plot(distances, energies)
    plt.show()
    plt.plot(distances, exactEnergies)
    plt.show()
    plt.plot(distances, accuracies)
    plt.show()
