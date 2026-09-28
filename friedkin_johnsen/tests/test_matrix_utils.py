import pytest
import sys
import numpy as np
from ..matrix_utils import (make_adjacency_matrix_stochastic,
                            normalize_prejudice_matrix,
                            normalize_susceptibility_diagonal)


class TestNormalizeAdjacencyMatrix:
    @pytest.mark.parametrize("matrix, expected_result", [
        ([[1, 1], [1, 1]], [[0.5, 0.5], [0.5, 0.5]]),
        ([[1, 3], [4, 1]], [[0.25, 0.75], [0.8, 0.2]]),
        ([[0.9, 0.1], [0, 0.01]], [[0.9, 0.1], [0, 1]]),

        ([[0, 1, 0, 0], [0.001, 0.001, 0.002, 0.002],
            [5, 20, 20, 5], [np.pi, np.pi, np.pi, np.pi]],
            [[0, 1, 0, 0], [1/6, 1/6, 1/3, 1/3],
                [0.1, 0.4, 0.4, 0.1], [1/4, 1/4, 1/4, 1/4]])
    ])
    def test_correct_normalization(self, matrix, expected_result):
        matrix = np.array(matrix, dtype=float)
        make_adjacency_matrix_stochastic(matrix)
        assert np.allclose(matrix, expected_result)

    def test_none_case(self):
        assert make_adjacency_matrix_stochastic(None) is None

    @pytest.mark.parametrize("matrix", [
        [[0, 0], [0, 0]],
        [[1, 2], [0, 0]],
        [[0, 0, 0, 0], [1/6, 1/6, 1/3, 1/3],
         [0.1, 0.4, 0.4, 0.1], [1/4, 1/4, 1/4, 1/4]]
    ])
    def test_raises_zero_row(self, matrix):
        with pytest.raises(ValueError,
                           match="Adjacency matrix is impossible "
                           "to render stochastic"):
            make_adjacency_matrix_stochastic(np.array(matrix, dtype=float))

    @pytest.mark.parametrize("matrix", [
        [[1, 0], [0, 1]],
        [["1", "2"], ["3", "4"]],
        [[None, 2], [3, 4]]
    ])
    def test_raises_not_float_dtype(self, matrix):
        with pytest.raises(ValueError,
                           match="Expected np.array of floating "
                           "point numbers"):
            make_adjacency_matrix_stochastic(np.array(matrix))

    @pytest.mark.parametrize("matrix", [
        [[1, 0, 0], [0, 0, 1]],
        [[1], [2]],
        [[1], [0], [0]], [[0], [0], [1]]
    ])
    def test_raises_matrix_not_square(self, matrix):
        with pytest.raises(ValueError, match="Expected square matrix"):
            make_adjacency_matrix_stochastic(np.array(matrix, dtype=float))

    @pytest.mark.parametrize("matrix, expected_result", [
        ([[-1, 1], [1, -1]], [[0.5, 0.5], [0.5, 0.5]]),
        ([[0.9, 0.1], [0, -0.01]], [[0.9, 0.1], [0, 1]])
    ])
    def test_flip_negative_values(self, matrix, expected_result):
        matrix = np.array(matrix, dtype=float)
        with pytest.warns(UserWarning, match=r"Some negative values were "
                          "detected in the input.*"):
            make_adjacency_matrix_stochastic(matrix)
        assert np.allclose(matrix, expected_result)


class TestNormalizePrejudiceMatrix:
    @pytest.mark.parametrize("matrix, expected_result", [
        ([[0, 1], [0, 1]], [[0, 1], [0, 1]]),
        ([[0, 0.5], [0.75, 1]], [[0, 0.5], [0.75, 1]]),

        ([[1, 3], [4, 1]], [[0, 2/3], [1, 0]]),
        ([[0, 1], [5, 3]], [[0, 1/5], [1, 3/5]]),
        ([[0, 6], [9, 3]], [[0, 2/3], [1, 1/3]]),

        ([[0, 0.5], [0.25, 0.3]], [[0, 1], [0.5, 0.6]]),
        ([[0.001, 0], [0.0002, 0.0005]], [[1, 0], [0.2, 0.5]])
    ])
    def test_correct_normalization(self, matrix, expected_result):
        matrix = np.array(matrix, dtype=float)
        normalize_prejudice_matrix(matrix)
        assert np.allclose(matrix, expected_result)

    @pytest.mark.parametrize("matrix, expected_result", [
            ([[0, -1], [0, -1]], [[1, 0], [1, 0]]),
            ([[0, -1], [0, 1]], [[0.5, 0], [0.5, 1]]),
            ([[0, -0.5], [-0.75, -1]], [[1, 0.5], [0.25, 0]]),

            ([[1, 3], [-4, 1]], [[5/7, 1], [0, 5/7]]),
            ([[0, 1], [5, -3]], [[3/8, 4/8], [1, 0]]),
            ([[0, -6], [9, 3]], [[6/15, 0], [1, 9/15]]),

            ([[0, 0.5], [0.25, -0.3]], [[0.3/0.8, 1], [0.55/0.8, 0]]),

            ([[0.001, 0], [0.0002, -0.0005]],
             [[1, 0.0005/0.0015], [0.0007/0.0015, 0]])
        ])
    def test_correct_normalizazion_negative_values(self,
                                                   matrix,
                                                   expected_result):
        matrix = np.array(matrix, dtype=float)
        normalize_prejudice_matrix(matrix)
        assert np.allclose(matrix, expected_result)

    def test_none_case(self):
        assert normalize_prejudice_matrix(None) is None

    @pytest.mark.parametrize("matrix", [
        [[1, 0], [0, 1]],
        [["1", "2"], ["3", "4"]],
        [[None, 2], [3, 4]]
    ])
    def test_raises_not_float_dtype(self, matrix):
        with pytest.raises(ValueError,
                           match="Expected np.array of floating "
                           "point numbers"):
            normalize_prejudice_matrix(np.array(matrix))


class TestNormalizeSusceptibilityMatrix:
    @pytest.mark.parametrize("matrix, expected_result", [
        ([1, 1], [1, 1]),
        ([0, 0.5, 0.75, 1], [0, 0.5, 0.75, 1]),

        ([1, 3, 4, 1], [0, 2/3, 1, 0]),
        ([0, 1, 5, 3], [0, 1/5, 1, 3/5]),
        ([0, 6, 9, 3], [0, 2/3, 1, 1/3]),

        ([0, 0.5, 0.25, 0.3], [0, 1, 0.5, 0.6]),
        ([0.001, 0, 0.0002, 0.0005], [1, 0, 0.2, 0.5])
    ])
    def test_correct_normalization(self, matrix, expected_result):
        matrix = np.diag(matrix).astype(float)
        expected_result = np.diag(expected_result).astype(float)

        normalize_susceptibility_diagonal(matrix)
        assert np.allclose(matrix, expected_result)

    @pytest.mark.parametrize("matrix, expected_result", [
            ([0, -1, 0, -1], [1, 0, 1, 0]),
            ([0, -1, 0, 1], [0.5, 0, 0.5, 1]),
            ([0, -0.5, -0.75, -1], [1, 0.5, 0.25, 0]),

            ([1, 3, -4, 1], [5/7, 1, 0, 5/7]),
            ([0, 1, 5, -3], [3/8, 4/8, 1, 0]),
            ([0, -6, 9, 3], [6/15, 0, 1, 9/15]),

            ([0, 0.5, 0.25, -0.3], [0.3/0.8, 1, 0.55/0.8, 0]),
            ([0.001, 0, 0.0002, -0.0005], [1, 0.0005/0.0015, 0.0007/0.0015, 0])
        ])
    def test_correct_normalizazion_negative_values(self,
                                                   matrix,
                                                   expected_result):
        matrix = np.diag(matrix).astype(float)
        expected_result = np.diag(expected_result).astype(float)

        normalize_susceptibility_diagonal(matrix)
        assert np.allclose(matrix, expected_result)

    def test_none_case(self):
        assert normalize_susceptibility_diagonal(None) is None

    @pytest.mark.parametrize("matrix", [
        [[1, 0], [0, 1]],
        [["1", "2"], ["3", "4"]],
        [[None, 2], [3, 4]]
    ])
    def test_raises_not_float_dtype(self, matrix):
        with pytest.raises(ValueError,
                           match="Expected np.array of floating "
                           "point numbers"):
            normalize_susceptibility_diagonal(np.array(matrix))

    @pytest.mark.parametrize("matrix", [
        [[1, 0, 0], [0, 0, 1]],
        [[1], [2]],
        [[1], [0], [0]], [[0], [0], [1]]
    ])
    def test_raises_matrix_not_square(self, matrix):
        with pytest.raises(ValueError, match="Expected square matrix"):
            normalize_susceptibility_diagonal(np.array(matrix, dtype=float))

    @pytest.mark.parametrize("matrix", [
        [[1, 1], [0, 1]],
        [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    ])
    def test_raises_not_diagonal(self, matrix):
        with pytest.raises(ValueError, match="Expected a diagonal matrix."):
            normalize_susceptibility_diagonal(np.array(matrix, dtype=float))


if __name__ == "__main__":
    sys.exit(pytest.main(
        ["-s", "-v",  "--cov=your_package", "--cov-report=term-missing"
         ]))
