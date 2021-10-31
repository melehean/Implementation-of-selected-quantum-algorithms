from numpy import pi, arccos
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute


def random_phase_generator(circuit, name_prefix=''):
    quantum_register = QuantumRegister(2, name_prefix + '_random_generator')
    classic_register = ClassicalRegister(2, name_prefix + '_phase')

    circuit.add_register(quantum_register)
    circuit.add_register(classic_register)

    circuit.rx(arccos(-1 / 3), quantum_register[0])
    circuit.crx(pi / 2, quantum_register[0], quantum_register[1])
    circuit.x(quantum_register[0])

    circuit.measure(quantum_register, classic_register)

    return quantum_register


def entanglement(circuit):
    quantum_register = QuantumRegister(2, 'entanglement')
    circuit.add_register(quantum_register)

    circuit.h(quantum_register[0])
    circuit.cx(quantum_register[0], quantum_register[1])

    return quantum_register


def phase_rotation(circuit, random_register, entanglement_register, start_angle=0):
    if start_angle:
        circuit.rx(start_angle, entanglement_register)
    circuit.crx(pi / 8, random_register[0], entanglement_register)
    circuit.crx(pi / 4, random_register[1], entanglement_register)


def measure_entanglement(circuit, entanglement_register, name_prefix=''):
    classic_register = ClassicalRegister(1, name_prefix + '_measure')
    circuit.add_register(classic_register)

    circuit.measure(entanglement_register, classic_register)

    return classic_register


def attack(circuit, entanglement_register):
    return measure_entanglement(circuit, entanglement_register, 'eve')


def draw(circuit):
    circuit.draw(output='mpl').show()


def run(circuit):
    n = 10000

    circuit_execution = execute(circuit, Aer.get_backend('aer_simulator'), shots=n, memory=True)
    circuit_results = circuit_execution.result()

    output_key = []
    a3b3 = []
    a3b1 = []
    a1b3 = []
    a1b1 = []
    for single_result in circuit_results.get_memory():
        values = single_result.split()
        bob_measure, alice_measure, bob_phase, alice_phase = values[0], values[1], values[-2], values[-1]
        bob_measure, alice_measure = int(bob_measure), int(alice_measure)
        measurement_equality = alice_measure == bob_measure
        if (alice_phase == '00' and bob_phase == '01') or (alice_phase == '01' and bob_phase == '00'):
            # assert alice_measure == bob_measure, "Measurements is not correlated"
            output_key.append(alice_measure)
        elif alice_phase == '00':
            if bob_phase == '10':
                a3b3.append(measurement_equality)
            elif bob_phase == '00':
                a3b1.append(measurement_equality)
        elif alice_phase == '10':
            if bob_phase == '00':
                a1b3.append(measurement_equality)
            elif bob_phase == '10':
                a1b1.append(measurement_equality)
    e_a3b3 = 2 * sum(a3b3) / len(a3b3) - 1
    e_a3b1 = 2 * sum(a3b1) / len(a3b1) - 1
    e_a1b3 = 2 * sum(a1b3) / len(a1b3) - 1
    e_a1b1 = 2 * sum(a1b1) / len(a1b1) - 1
    print([e_a3b3, e_a3b1, e_a1b3, e_a1b1])
    security_check = e_a3b3 + e_a3b1 + e_a1b3 - e_a1b1
    assert security_check >= 2, "Channel is not secure: " + str(security_check)

    print("Initial length: ", n, "; output key length: ", len(output_key), "; security check: ", security_check)


if __name__ == '__main__':
    circuit = QuantumCircuit()
    alice_quantum_register = random_phase_generator(circuit, 'alice')
    entanglement_register = entanglement(circuit)
    bob_quantum_register = random_phase_generator(circuit, 'bob')

    circuit.barrier(entanglement_register)

    phase_rotation(circuit, alice_quantum_register, entanglement_register[0])

    circuit.barrier()

    # attack(circuit, entanglement_register[1])

    # circuit.barrier()

    phase_rotation(circuit, bob_quantum_register, entanglement_register[1], -pi / 8)

    alice_measure = measure_entanglement(circuit, entanglement_register[0], 'alice')
    bob_measure = measure_entanglement(circuit, entanglement_register[1], 'bob')

    draw(circuit)
    run(circuit)
