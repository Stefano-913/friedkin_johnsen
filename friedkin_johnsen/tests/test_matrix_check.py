import pytest
import sys
import numpy as np
from ..matrix_check import (matrix_is_2D,
                            is_square_matrix,
                            get_number_of_users,
                            is_matrix_diag,
                            matrices_are_compatible,
                            has_matrix_zero_row)


class TestMatrixIs2D:
    # Since all the following queries have the same decorator containing
    # the basic input checks, the exceptions raised in it will be tested
    # in matrix_is_2D(), which does basically nothing but call said decorator.

    @pytest.mark.parametrize("matrix", [
        np.array([[1], [2]]),
        np.array([[1, 2], [3, 4]])
    ])
    def test_matrix_is_2D(self, matrix):
        matrix_is_2D(matrix)

    def test_none_case(self):
        assert matrix_is_2D(None) is True

    @pytest.mark.parametrize("matrix", [
        "[[1],[2]]",
        [[1], [2]],
        (1, 2),
        123
    ])
    def test_wrong_type(self, matrix):
        with pytest.raises(TypeError,
                           match=r"Input is of.*type, "
                           "expected a np.ndarray matrix."):
            matrix_is_2D(matrix)

    @pytest.mark.parametrize("matrix", [
        [],
        [1],

        [[[]]],
        [[[1]]],
        [[[1], [2]], [[3], [4]]]
    ])
    def test_wrong_dimension(self, matrix):
        with pytest.raises(ValueError,
                           match=r"Input is of.*dimension, "
                           "expected a 2D matrix."):
            matrix_is_2D(np.array(matrix))


class TestIsSquareMatrix:
    @pytest.mark.parametrize("matrix, result", [
        ([[1]], True),
        ([[1, 2], [3, 4]], True),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], True),

        ([[1, 2, 3], [4, 5, 6]], False),
        ([[1, 2], [3, 4], [5, 6]], False)
    ])
    def test_is_square_matrix(self, matrix, result):
        assert is_square_matrix(np.array(matrix)) == result

    def test_none_case(self):
        assert is_square_matrix(None) is True

    def test_empty_matrix(self):
        assert is_square_matrix(np.empty((0, 0))) is True


class TestGetNumberOfUsers:
    @pytest.mark.parametrize("matrix, result", [
        ([[1]], 1),
        ([[1], [2], [3]], 3),
        ([[1], [2], [3], [4], [5]], 5),

        ([[1, 1], [2, 2]], 2),
        ([[1, 1, 1], [2, 2, 2], [3, 3, 3]], 3),

        ([[1, 1, 1], [2, 2, 2]], 2),
        ([[1, 1], [2, 2], [3, 3]], 3)
    ])
    def test_get_number_of_users(self, matrix, result):
        assert get_number_of_users(np.array(matrix)) == result

    def test_none_case(self):
        assert get_number_of_users(None) is None

    def test_empty_matrix(self):
        assert get_number_of_users(np.empty((0, 0))) == 0


class TestIsMatrixDiag:
    @pytest.mark.parametrize("matrix, result", [
        ([[1]], True),
        ([[1, 0], [0, 1]], True),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True),

        ([[1], [0]], False),
        ([[1], [0], [0]], False),
        ([[1, 2], [3, 4], [5, 6]], False),
        ([[1, 0], [0, 0], [0, 1]], False)
    ])
    def test_is_matrix_diag(self, matrix, result):
        assert is_matrix_diag(np.array(matrix)) == result

    def test_none_case(self):
        assert is_matrix_diag(None) is True

    def test_empty_matrix(self):
        assert is_matrix_diag(np.empty((0, 0))) is True

# -----


class TestAreMatricesCompatible:
    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        pytest.param(None, None, None, id="all_none"),

        pytest.param([[1]], None, None, id="only_adjacency"),
        pytest.param(None, [[1]], None, id="only_prejudice"),
        pytest.param(None, None, [[1]], id="only_susceptibility"),

        pytest.param(None, [[1]], [[1]], id="missing_adjacency"),
        pytest.param([[1]], None, [[1]], id="missing_prejudice"),
        pytest.param([[1]], [[1]], None, id="missing_susceptibility")
        ])
    def test_none_matrices(self,
                           adjacency,
                           prejudice,
                           susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        assert matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [

        pytest.param([[1, 2], [3, 4]], None, None,
                     id="only_adjacency_dim2"),
        pytest.param(None, [[1], [2]], None,
                     id="only_prejudice_dim2"),
        pytest.param(None, None, [[1, 0], [0, 1]],
                     id="only_susceptibility_dim2"),

        pytest.param(None, [[1], [2]], [[1, 0], [0, 1]],
                     id="missing_adjacency_dim2"),
        pytest.param([[1, 2], [3, 4]], None, [[1, 0], [0, 1]],
                     id="missing_prejudice_dim2"),
        pytest.param([[1, 2], [3, 4]], [[1], [2]], None,
                     id="missing_susceptibility_dim2"),

        pytest.param([[1, 2, 3], [4, 5, 6], [7, 8, 9]], None, None,
                     id="only_adjacency_dim3"),
        pytest.param(None, [[1], [2], [3]], None,
                     id="only_prejudice_dim3"),
        pytest.param(None, None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     id="only_susceptibility_dim3"),

        pytest.param(None,
                     [[1], [2], [3]],
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     id="missing_adjacency_dim3"),

        pytest.param([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     None,
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     id="missing_prejudice_dim3"),

        pytest.param([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[1], [2], [3]],
                     None,
                     id="missing_susceptibility_dim3"),
    ])
    def test_none_multi_dimension_matrices(self,
                                           adjacency,
                                           prejudice,
                                           susceptibility):

        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        assert matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        pytest.param([[1, 2], [3, 4]],
                     [[1], [2]],
                     [[1, 0], [0, 1]],
                     id="2d_integers"),

        pytest.param([[0.2, 3], [6, -4]],
                     [[3], [0.4]],
                     [[5.3, 0], [0, np.pi]],
                     id="2d_negative_and_float_values"),

        pytest.param([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[1], [2], [3]],
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     id="3d_integers"),

        pytest.param([[0.31, 2, -3], [0, 0, 1], [0.4, -np.sqrt(5), 3]],
                     [[5], [0.6], [-1]],
                     [[1, 0, 0], [0, 5.9, 0], [0, 0, 1]],
                     id="3d_negative_and_float_values")
        ])
    def test_matrices_are_compatible(self,
                                     adjacency,
                                     prejudice,
                                     susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        assert matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        ("not a matrix", None, None),
        (None, [[1, 2], [3, 4]], None),
        (None, None, 123)
    ])
    def test_wrong_type(self, adjacency, prejudice, susceptibility):
        with pytest.raises(TypeError,
                           match=r".*is of.*type, expected a "
                           "np.ndarray matrix."):
            matrices_are_compatible(adjacency, prejudice, susceptibility)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        ([[1, 2], [3, 4]], [[1], [2], [3]], None),
        ([[1, 2], [3, 4]], None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (None, [[1], [2]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ])
    def test_incompatible_no_of_users(self,
                                      adjacency,
                                      prejudice,
                                      susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        with pytest.raises(ValueError,
                           match=r"Input matrices have an "
                           "incompatible numbers of users."):
            matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        ([[1, 2]], None, None),
        ([[1, 2, 3], [4, 5, 6]], [[1], [2]], None)
    ])
    def test_adjacency_not_square(self, adjacency, prejudice, susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        with pytest.raises(ValueError,
                           match=r"Inserted adjacency matrix "
                           "isn't a square matrix."):
            matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        (None, None, [[1, 2]]),
        (None, None, [[1, 2, 3], [4, 5, 6]])
    ])
    def test_susceptibility_not_square(self,
                                       adjacency,
                                       prejudice,
                                       susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        with pytest.raises(ValueError,
                           match=r"Inserted susceptibility matrix "
                           "isn't a square matrix."):
            matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        (None, None, [[1, 0], [1, 1]]),
        (None, None, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    ])
    def test_susceptibility_not_diagonal(self,
                                         adjacency,
                                         prejudice,
                                         susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        with pytest.raises(ValueError,
                           match=r"Inserted susceptibility matrix "
                           "isn't a diagonal matrix."):
            matrices_are_compatible(*matrices_list)

    @pytest.mark.parametrize("adjacency, prejudice, susceptibility", [
        ([[1, 2], [3, 4]], [1, 2, 3], None),
        (None, [[[1], [2]]], [[1, 0], [0, 1]])
    ])
    def test_prejudice_wrong_dimension(self,
                                       adjacency,
                                       prejudice,
                                       susceptibility):
        matrices_list = [adjacency, prejudice, susceptibility]
        for i in range(len(matrices_list)):
            if matrices_list[i] is not None:
                matrices_list[i] = np.array(matrices_list[i])
        with pytest.raises(ValueError,
                           match=r"Input is of.*dimension, "
                           "expected a 2D matrix."):
            assert matrices_are_compatible(*matrices_list)


class TestHasMatrixZeroRow:
    @pytest.mark.parametrize("matrix, has_zero_row", [
        ([[0, 0], [0, 0]], True),
        ([[0, 0], [1, 1]], True),
        ([[1, 0], [0, 0]], True),
        ([[0, 1], [1, 0]], False),

        ([[1, 1], [1, 0], [0, 1]], False),
        ([[1, 1], [0, 0], [1, 1]], True),
        ([[1, 1], [1, 1], [0, 0]], True),

        ([[1, 1], [1, 1], [1, 1], [0, 0]], True)
        ])
    def test_has_matrix_zero_row(self, matrix, has_zero_row):
        assert has_matrix_zero_row(np.array(matrix)) == has_zero_row

    @pytest.mark.parametrize("vector, has_zero_row", [
        ([[0], [0]], True),
        ([[0], [1], [1]], True),
        ([[1], [0], [1]], True),
        ([[1], [1], [1]], False)
        ])
    def test_with_column_vector(self, vector, has_zero_row):
        assert has_matrix_zero_row(np.array(vector)) == has_zero_row

    @pytest.mark.parametrize("vector ,has_zero_row", [
        ([0], True),
        ([0, 0], True),
        ([0, 0, 0], True),

        ([1], False),
        ([1, 0], False),
        ([0, 1], False),
        ([1, 1], False)
        ])
    def test_with_row_vector(self, vector, has_zero_row):
        assert has_matrix_zero_row(np.array(vector)) == has_zero_row

    @pytest.mark.parametrize("matrix , tolerance ,has_zero_row", [
        ([[1e-11, 0], [1, 1]], 1e-10, True),
        ([[1e-11, 0], [1, 1]], 1e-12, False),
        ([[1e-11, 0], [1, 1]], 1e-11, False),

        ([[1e-2, 0], [1, 1]], 1e-2, False),
        ([[-1e-2, 0], [1, 1]], 1e-2, False),
        ([[-1e-2, 0], [1, 1]], 1e-3, False),
        ([[-1e-2, 0], [1, 1]], 1e-1, True)
    ])
    def test_with_tolerance(self, matrix, tolerance, has_zero_row):
        assert has_matrix_zero_row(np.array(matrix),
                                   tolerance=tolerance) == has_zero_row

    @pytest.mark.parametrize("vector , tolerance , has_zero_row", [
        ([[1e-11], [1], [1]], 1e-10, True),
        ([[1e-11], [1], [1]], 1e-12, False),
        ([[1e-11], [1], [1]], 1e-11, False),

        ([[-1e-2], [-1e-2], [1e-2]], 1e-2, False),
        ([[-1e-2], [1e-2], [-1e-2]], 1e-3, False),
        ([[-1e-2], [-1e-2], [-1e-2]], 1e-1, True)
    ])
    def test_with_tolerance_column_vector(self,
                                          vector,
                                          tolerance,
                                          has_zero_row):
        assert has_matrix_zero_row(np.array(vector),
                                   tolerance=tolerance) == has_zero_row

    @pytest.mark.parametrize("vector , tolerance , has_zero_row", [
        ([1e-11, 0, 0], 1e-10, True),
        ([1e-11, 0, 0], 1e-12, False),
        ([1e-11, 0, 0], 1e-11, False),

        ([1e-2, 0, 0], 1e-2, False),
        ([-1e-2, 1e-2, -1e-2], 1e-2, False),
        ([-1e-2, -1e-2, 1e-2], 1e-3, False),
        ([-1e-2, 0, 0], 1e-1, True)
    ])
    def test_with_tolerance_row_vector(self, vector, tolerance, has_zero_row):
        assert has_matrix_zero_row(np.array(vector),
                                   tolerance=tolerance) == has_zero_row

    def test_error_if_wrong_dim(self):
        with pytest.raises(ValueError, match=r".*has unexpected shape."
                           r"*, incompatible with a matrix or a vector."):
            has_matrix_zero_row(np.array([[[0, 0], [0, 0]]]))
        with pytest.raises(ValueError, match=r".*has unexpected shape."
                           r"*, incompatible with a matrix or a vector."):
            has_matrix_zero_row(np.array([[[0, 0], [0, 0]], [[1, 2], [3, 4]]]))

    @pytest.mark.parametrize("matrix, tolerance, has_zero_row", [
        ([1e-11, 0, 0], -1e-10, True),
        ([1e-11, 0, 0], -1e-12, False),
        ([1e-11, 0, 0], -1e-11, False),
    ])
    def test_negative_tolerance_is_accepted(self,
                                            matrix,
                                            tolerance,
                                            has_zero_row):
        assert has_matrix_zero_row(np.array(matrix),
                                   tolerance=tolerance) == has_zero_row

    @pytest.mark.parametrize("tolerance_type", [None, [], object(), "str",
                                                3+4j, (), {}])
    def test_raises_if_param_not_float(self, tolerance_type):
        with pytest.raises(TypeError):
            has_matrix_zero_row(matrix=np.array([0, 0]),
                                tolerance=tolerance_type)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v",
                          "--cov=your_package",
                          "--cov-report=term-missing"]))
