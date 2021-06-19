from numpy import pi, arccos
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def random_phase_generator(circuit, name_prefix=''):
    quantum_register = QuantumRegister(2, name_prefix + '_random_generator')
    classic_register = ClassicalRegister(2, name_prefix + '_phase')

    circuit.add_register(quantum_register)
    circuit.add_register(classic_register)

    circuit.rx(arccos(-1 / 3), quantum_register[0])
    circuit.crx(pi / 2, quantum_register[0], quantum_register[1])

    circuit.measure(quantum_register, classic_register)

    return quantum_register


def entanglement(circuit):
    quantum_register = QuantumRegister(2, 'entanglement')

    circuit.add_register(quantum_register)

    circuit.h(quantum_register[0])
    circuit.cx(quantum_register[0], quantum_register[1])

    return quantum_register


def draw(circuit):
    # plot_bloch_multivector(circuit).show()
    circuit.draw(output='mpl').show()


if __name__ == '__main__':
    circuit = QuantumCircuit()
    alice_quantum_register = random_phase_generator(circuit, 'alice')
    entanglement_register = entanglement(circuit)
    bob_quantum_register = random_phase_generator(circuit, 'bob')

    draw(circuit)
