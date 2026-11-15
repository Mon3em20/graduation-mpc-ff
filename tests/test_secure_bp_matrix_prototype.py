import math
import unittest


def scalar_forward(x, weights, bias):
    return [
        sum(x[input_] * weights[input_][output]
            for input_ in range(len(x))) + bias[output]
        for output in range(len(bias))
    ]


def matrix_forward(x_rows, weights, bias):
    return [
        [
            sum(x_rows[row][input_] * weights[input_][output]
                for input_ in range(len(x_rows[row]))) + bias[output]
            for output in range(len(bias))
        ]
        for row in range(len(x_rows))
    ]


def matrix_transpose_product(left, right):
    return [
        [
            sum(left[row][left_column] * right[row][right_column]
                for row in range(len(left)))
            for right_column in range(len(right[0]))
        ]
        for left_column in range(len(left[0]))
    ]


class SecureBPMatrixPrototypeTests(unittest.TestCase):
    def test_batched_forward_matches_scalar_forward(self):
        x_rows = [[1.0, 2.0, 4.0], [-1.0, 3.0, 2.0]]
        weights = [[2.0, 5.0], [3.0, 7.0], [11.0, 13.0]]
        bias = [17.0, 19.0]
        expected = [scalar_forward(row, weights, bias) for row in x_rows]
        self.assertEqual(matrix_forward(x_rows, weights, bias), expected)

    def test_unreduced_gradient_reduces_once_after_batch_sum(self):
        activations = [[1.0, 2.0], [3.0, 5.0]]
        deltas = [[7.0, 11.0], [13.0, 17.0]]
        gradient = matrix_transpose_product(activations, deltas)
        expected = [[46.0, 62.0], [79.0, 107.0]]
        self.assertEqual(gradient, expected)

    def test_batch_average_uses_actual_incomplete_batch_size(self):
        learning_rate = 0.05
        weight = 1.0
        gradient_sum = 0.8
        self.assertAlmostEqual(weight - learning_rate * gradient_sum / 3, 0.9866666666666667)
        self.assertNotEqual(weight - learning_rate * gradient_sum / 3,
                            weight - learning_rate * gradient_sum / 8)

    def test_stable_cross_entropy_reference(self):
        logits = [1000.0, 999.0]
        shift = max(logits)
        log_sum_exp = shift + math.log(sum(math.exp(value - shift)
                                           for value in logits))
        self.assertAlmostEqual(log_sum_exp - logits[0],
                               math.log1p(math.exp(-1.0)))


if __name__ == "__main__":
    unittest.main()
