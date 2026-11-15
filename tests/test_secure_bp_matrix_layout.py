import unittest


def torch_linear_flattened(rows):
    return [value for row in rows for value in row]


def mp_spdz_weight_order(torch_rows):
    return [
        torch_rows[output][input_]
        for input_ in range(len(torch_rows[0]))
        for output in range(len(torch_rows))
    ]


class SecureBPMatrixLayoutTests(unittest.TestCase):
    def test_nonsymmetric_linear_weight_transpose(self):
        torch_rows = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
        self.assertEqual(torch_linear_flattened(torch_rows),
                         [1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        self.assertEqual(mp_spdz_weight_order(torch_rows),
                         [1.0, 10.0, 2.0, 20.0, 3.0, 30.0])

    def test_transposed_order_preserves_linear_output(self):
        torch_rows = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
        vector = [4.0, 5.0, 6.0]
        expected = [sum(weight * value for weight, value in zip(row, vector))
                    for row in torch_rows]
        transposed = mp_spdz_weight_order(torch_rows)
        actual = [
            sum(transposed[input_ * 2 + output] * vector[input_]
                for input_ in range(3))
            for output in range(2)
        ]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
