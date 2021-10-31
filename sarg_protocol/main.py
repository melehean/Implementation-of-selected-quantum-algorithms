from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute


def add_qubit(circuit, n, name_prefix=''):
    quantum_register = QuantumRegister(n, name_prefix)

    circuit.add_register(quantum_register)

    return quantum_register


def random_value_generator(circuit, n, name_prefix=''):
    quantum_register = QuantumRegister(n, name_prefix + '_val')
    classic_register = ClassicalRegister(n, name_prefix + '_c_val')

    circuit.add_register(quantum_register)
    circuit.add_register(classic_register)

    circuit.h(quantum_register)

    circuit.measure(quantum_register, classic_register)

    return quantum_register


def encode_secret(circuit, n, quantum_secret_register, quantum_basis_register, name_prefix=''):
    quantum_register = QuantumRegister(n, name_prefix + '_encoded')
    circuit.add_register(quantum_register)

    circuit.cx(quantum_secret_register, quantum_register)
    circuit.cry(pi / 2, quantum_basis_register, quantum_register)

    return quantum_register


def decode_secret(circuit, quantum_value_register, quantum_basis_register):
    circuit.cry(- pi / 2, quantum_basis_register, quantum_value_register)


def transfer(circuit, from_register, to_register):
    circuit.swap(from_register, to_register)


def measure(circuit, n, register, name_prefix=''):
    classic_register = ClassicalRegister(n, name_prefix + '_measure')
    circuit.add_register(classic_register)

    circuit.measure(register, classic_register)

    return classic_register


def draw(circuit):
    circuit.draw(output='mpl').show()


def run(circuit, n):
    import random
    shots = 100

    circuit_execution = execute(circuit, Aer.get_backend('aer_simulator'), shots=shots, memory=True)
    circuit_results = circuit_execution.result()

    success = 0

    for single_result in circuit_results.get_memory():
        values = single_result.split()
        alice_secret_v, alice_basis_v, bob_receive_v, bob_basis_v = values[-1], values[-2], values[0], values[1]
        for i in range(0, n):
            alice_secret = int(alice_secret_v[i])
            alice_basis = int(alice_basis_v[i])
            bob_receive = int(bob_receive_v[i])
            bob_basis = int(bob_basis_v[i])

            alice_public_state = [0, 0]

            alice_public_state[alice_basis] = alice_secret
            alice_public_state[1 - alice_basis] = random.getrandbits(1)

            if alice_public_state[bob_basis] != bob_receive:
                assert alice_secret == alice_public_state[1 - bob_basis]
                success += 1
    print('success: ', success)


if __name__ == '__main__':
    n = 1
    circuit = QuantumCircuit()
    alice_quantum_secret_register = random_value_generator(circuit, n, 'alice_secret')
    alice_quantum_basis_register = random_value_generator(circuit, n, 'alice_basis')
    alice_encoded_register = encode_secret(circuit, n, alice_quantum_secret_register, alice_quantum_basis_register,
                                           'alice')

    circuit.barrier()

    bob_quantum_basis_register = random_value_generator(circuit, n, 'bob_basis')
    bob_receive_register = add_qubit(circuit, n, 'bob_receive')

    transfer(circuit, alice_encoded_register, bob_receive_register)

    decode_secret(circuit, bob_receive_register, bob_quantum_basis_register)

    circuit.barrier()

    bob_measure = measure(circuit, n, bob_receive_register, 'bob')

    draw(circuit)
    run(circuit, n)
