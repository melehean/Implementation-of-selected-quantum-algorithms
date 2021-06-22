import math
from fractions import Fraction

import matplotlib.pyplot
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram


def shor(N, a=2, counting_qubits_amount=None):
    # log(N) qubits per result, U-gate size
    # x=2^amount_counting_qubits-1 exp of last U-gate; a^x
    if counting_qubits_amount is None:
        counting_qubits_amount = math.ceil(math.log2(N))
    circuit = qpu_part(N, a, counting_qubits_amount)

    factors = None
    while factors is None:
        print("try")
        measurement = run(circuit, N)
        factors = cpu_part(measurement, N, a, counting_qubits_amount)
    print(factors)


def qpu_part(N, a, counting_qubits_amount):
    # Circuit initialization
    circuit = QuantumCircuit()

    result_qubits_amount = math.ceil(math.log2(N))
    count_register = QuantumRegister(counting_qubits_amount, 'count')
    result_register = QuantumRegister(result_qubits_amount, 'result')
    read_register = ClassicalRegister(result_qubits_amount, 'read')
    circuit.add_register(count_register)
    circuit.add_register(result_register)
    circuit.add_register(read_register)

    # Superposition
    circuit.h(count_register)

    # Result register initialization (write 1)
    circuit.x(result_register[0])

    # Exponentiation/Multiplication
    circuit.barrier()
    for i in range(counting_qubits_amount):
        power = 2 ** i
        for j in range(power):
            for k in range(counting_qubits_amount - 1, 0, -1):
                circuit.cswap(count_register[i], result_register[k], result_register[k-1])

    # QFT
    circuit.barrier()
    for i in range(counting_qubits_amount - 1, -1, -1):
        circuit.h(count_register[i])
        phase = math.pi
        for j in range(i - 1, -1, -1):
            phase = phase / 2
            circuit.cp(phase, count_register[i], count_register[j])
        circuit.barrier(count_register)
    for i in range(counting_qubits_amount//2):
        circuit.swap(count_register[i], count_register[counting_qubits_amount - i - 1])

    # Measure
    circuit.barrier()
    circuit.measure(count_register, read_register)

    draw(circuit)
    return circuit


def cpu_part(measurement, N, a, counting_qubits_amount):
    phase = measurement/(2**counting_qubits_amount)
    r = Fraction(phase).limit_denominator(15).denominator
    factors = [math.gcd(a ** (r // 2) - 1, N), math.gcd(a ** (r // 2) + 1, N)]
    if factors[0] != 1 and factors[1] != 1:
        return factors
    return None


def draw(circuit):
    circuit.draw(output='mpl', plot_barriers=False)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()


def run(circuit, N):
    circuit_execution = execute(circuit, Aer.get_backend('aer_simulator'), shots=1)
    circuit_results = circuit_execution.result()
    print(list(circuit_results.get_counts().items()))
    for i in range(len(list(circuit_results.get_counts().items()))):
        print(int(list(circuit_results.get_counts().items())[i][0], 2))
    # plot_histogram(circuit_results.get_counts())
    # matplotlib.pyplot.show()
    return int(list(circuit_results.get_counts().items())[0][0], 2)


if __name__ == '__main__':
    shor(15)  # 15 63 255 !!511 1023
