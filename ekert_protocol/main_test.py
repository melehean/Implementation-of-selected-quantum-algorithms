import unittest

from qiskit import Aer, execute, QuantumCircuit

from main import random_phase_generator, entanglement


class EkertTest(unittest.TestCase):
    def test_random_phase_generator(self):
        shots = 10 ** 6
        decimal_places = 2

        circuit = QuantumCircuit()
        random_phase_generator(circuit, 'test')

        circuit_execution = execute(circuit, Aer.get_backend('aer_simulator'), shots=shots)
        circuit_results = circuit_execution.result()
        counts = circuit_results.get_counts(circuit)

        print(counts)

        self.assertAlmostEqual(counts['00'] / shots, 1 / 3, decimal_places)
        self.assertAlmostEqual(counts['01'] / shots, 1 / 3, decimal_places)
        self.assertAlmostEqual(counts['10'] / shots, 1 / 3, decimal_places)

    def test_entanglement(self):
        shots = 10 ** 6
        decimal_places = 2

        circuit = QuantumCircuit()
        entanglement(circuit)
        circuit.measure_all()

        circuit_execution = execute(circuit, Aer.get_backend('aer_simulator'), shots=shots)
        circuit_results = circuit_execution.result()
        counts = circuit_results.get_counts(circuit)

        print(counts)

        self.assertAlmostEqual(counts['00'] / shots, 1 / 2, decimal_places)
        self.assertAlmostEqual(counts['11'] / shots, 1 / 2, decimal_places)


if __name__ == '__main__':
    unittest.main()
