import math

import matplotlib.pyplot
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram


def grover(problem, clauses, variables):

    # Circuit initialization
    circuit = QuantumCircuit()

    variables_register = QuantumRegister(variables, 'variables')
    clauses_register = QuantumRegister(clauses, 'clauses')
    read_register = ClassicalRegister(variables, 'read')
    circuit.add_register(variables_register)
    circuit.add_register(clauses_register)
    circuit.add_register(read_register)

    # Superposition
    circuit.h(variables_register)
    for _ in range(int(math.pi*math.sqrt(2**variables)/4)):
        # Oracle
        i = 0
        for clause in problem:
            for variable in clause:
                if variable < 0:
                    circuit.x(variables_register[abs(variable) - 1])
                circuit.x(variables_register[abs(variable) - 1])

            circuit.mct([variables_register[abs(j) - 1] for j in clause], clauses_register[i])
            circuit.x(clauses_register[i])

            for variable in clause:
                if variable < 0:
                    circuit.x(variables_register[abs(variable) - 1])
                circuit.x(variables_register[abs(variable) - 1])
            i += 1
            circuit.barrier()

        circuit.h(clauses_register[-1])
        circuit.mct(clauses_register[:-1], clauses_register[-1])
        circuit.h(clauses_register[-1])
        circuit.barrier()

        # Uncomputing
        i = clauses - 1
        for clause in reversed(problem):
            for variable in clause:
                circuit.x(variables_register[abs(variable) - 1])
                if variable < 0:
                    circuit.x(variables_register[abs(variable) - 1])

            circuit.x(clauses_register[i])
            circuit.mct([variables_register[abs(j) - 1] for j in clause], clauses_register[i])

            for variable in clause:
                circuit.x(variables_register[abs(variable) - 1])
                if variable < 0:
                    circuit.x(variables_register[abs(variable) - 1])
            i -= 1
            circuit.barrier()

        # Amplification
        circuit.h(variables_register)
        circuit.x(variables_register)
        circuit.h(variables_register[-1])
        circuit.mct(variables_register[:-1], variables_register[-1])
        circuit.h(variables_register[-1])
        circuit.x(variables_register)
        circuit.h(variables_register)
        circuit.barrier()

    # Measure
    circuit.measure(variables_register, read_register)

    draw(circuit)
    run(circuit)
    return circuit


def draw(circuit):
    circuit.draw(output='mpl', plot_barriers=True)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()


def run(circuit):
    circuit_execution = execute(circuit, Aer.get_backend('aer_simulator'), shots=10000)
    circuit_results = circuit_execution.result()
    print(list(circuit_results.get_counts().items()))
    plot_histogram(circuit_results.get_counts())
    matplotlib.pyplot.show()


if __name__ == '__main__':
    # (a or b) and (not a or c) and (not b or not c) and (a or c)
    problem = [[1, 2], [-1, 3], [-2, -3], [1, 3]]
    clauses_number = len(problem)
    variables_number = 3
    grover(problem.copy(), clauses_number, variables_number)