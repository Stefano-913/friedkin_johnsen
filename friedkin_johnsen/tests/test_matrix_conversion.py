import pytest
import sys
import numpy as np
from ..matrix_conversion import to_array
from ..matrix_conversion import to_2D_matrix

# --------


class TestToArray:

    @pytest.mark.parametrize("input_array_like, expected_array", [
         ([], np.array([], dtype=float)),
         ((), np.array([], dtype=float)),
         (np.array([]), np.array([], dtype=float)),
         ("", np.array([], dtype=float)),

         (1, np.array(1, dtype=float)),
         ([1], np.array([1], dtype=float)),
         ("-1", np.array(-1, dtype=float)),
         ((np.pi), np.array(np.pi, dtype=float)),
         (True, np.array(1, dtype=float)),
        ])
    def test_0_dimensional(self, input_array_like, expected_array):
        assert np.allclose(to_array(input_array_like), expected_array)

    @pytest.mark.parametrize("input_array_like, expected_array", [
         ([1, 2, 3], [1, 2, 3]),
         ((0, 0, 0), [0, 0, 0]),
         (["1", "2", "3"], [1, 2, 3]),
         ("(1, 2, 3)", [1, 2, 3]),

         (np.array([-4, +2, 5.5], dtype=float), [-4, 2, 5.5])
        ])
    def test_1_dimensional(self, input_array_like, expected_array):
        assert np.allclose(to_array(input_array_like),
                           np.array(expected_array, dtype=float))

    @pytest.mark.parametrize("input_array_like, expected_array", [
         ([[1], [2]], [[1], [2]]),
         ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
         ([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]),

         ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),

         (np.array([[1.2, 2], [np.pi, 4]]), [[1.2, 2], [np.pi, 4]]),
         ("[[1, 2],[3, 4]]", [[1, 2], [3, 4]])
        ])
    def test_multi_dimensional(self, input_array_like, expected_array):
        assert np.allclose(to_array(input_array_like),
                           np.array(expected_array, dtype=float))

    @pytest.mark.parametrize("input_array_like, expected_array", [
         ([[]], [[]]),
         ((), []),
         ([[[]]], [[[]]]),
         ([[[], []], [[], []]], [[[], []], [[], []]]),
         ("[()]", [()]),
         ("[[[[]]]]", [[[[]]]]),
        ])
    def test_empty_structures(self, input_array_like, expected_array):
        assert np.allclose(to_array(input_array_like),
                           np.array(expected_array, dtype=float))

    @pytest.mark.parametrize("input_array_like, expected_array", [
         ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
         (([1, 2], [3, 4]), [[1, 2], [3, 4]]),
         (((1, 2), (3, 4)), [[1, 2], [3, 4]]),
         ((((1, 2), (3, 4))), [[1, 2], [3, 4]])
        ])
    def test_various_input_formats(self, input_array_like, expected_array):
        assert np.allclose(to_array(input_array_like),
                           np.array(expected_array, dtype=float))

    @pytest.mark.parametrize("string_input", [
         "invalid_string",
         "2+5",
         "np.pi",
         "np.inf",
         "[1,2,3-4]",
         "np.sin(1)",
         "[np.sin(1), np.cos(1)]",
         ])
    def test_error_string_to_interpret(self, string_input):
        with pytest.raises(ValueError,
                           match=r"String '.*' contains invalid "
                           "expression that might need evaluation.\n.*"):
            to_array(string_input)

    @pytest.mark.parametrize("string_input", [
         "0)",
         "[2, 3}",
         "[1, 2, 3",
         "(1",
         ])
    def test_error_syntax_string_(self, string_input):
        with pytest.raises(SyntaxError,
                           match=r"String '.*' contains a syntax error "
                           "that invalidates the input.\n.*"):
            to_array(string_input)

    @pytest.mark.parametrize("not_array_like_input", [
        [1, 2, [3, 4]],
        [1, 2, [[], []]],
        [1, 2, (3, 4)]
        ])
    def test_error_not_convertible(self, not_array_like_input):
        with pytest.raises(ValueError, match=r".*cannot "
                           r"be converted to a matrix.\n.*"):
            to_array(not_array_like_input)

    @pytest.mark.parametrize("not_floating_point_input", [
         object(),
         {1, 2, 3},
         {1: "a", 2: "b"}
         ])
    def test_error_not_convertible_to_float(self, not_floating_point_input):
        with pytest.raises(TypeError, match=r".*cannot "
                           r"be converted to floating-point.\n.*"):
            to_array(not_floating_point_input)

    def test_error_complex_numbers(self):
        with pytest.raises(TypeError):
            to_array([1, 2, 3+4j])
        with pytest.raises(TypeError):
            to_array("[1, 2, 3+4j]")
        with pytest.raises(TypeError):
            to_array(np.array([1, 2, 3], dtype=complex))

    def test_error_nan_or_inf(self):
        with pytest.raises(ValueError,
                           match=r".*contains NaN or infinite values."):
            to_array([np.nan])

        with pytest.raises(ValueError,
                           match=r".*contains NaN or infinite values."):
            to_array([1, 2, 3, np.nan])

        with pytest.raises(ValueError,
                           match=r".*contains NaN or infinite values."):
            to_array([np.inf])

        with pytest.raises(ValueError,
                           match=r".*contains NaN or infinite values."):
            # None converts to 'nan' during astype(float)
            to_array(None)

# --------


class TestTo2DMatrix:

    @pytest.mark.parametrize("input_array_like", [
        (np.array([])),
        (np.array([1])),
        (np.array([1, 2, 3]))
    ])
    def test_if_vector(self, input_array_like):
        array = to_2D_matrix(input_array_like)
        assert array.ndim == 2
        assert array.shape[1] == 1

    @pytest.mark.parametrize("input_array_like, expected_shape", [
        (np.array([[]]), (1, 0)),
        (np.array([[], [], []]), (3, 0)),
        (np.array([[1], [2], [3]]), (3, 1)),
        (np.array([[1, 2, 3]]), (1, 3)),
        (np.array([[1, 2, 3], [4, 5, 6]]), (2, 3))
    ])
    def test_if_2D_matrix(self, input_array_like, expected_shape):
        array = to_2D_matrix(input_array_like)
        assert array.ndim == 2
        assert array.shape == expected_shape

    @pytest.mark.parametrize("input_array_like", [
        (np.array([[[]]])),
        (np.array([[[1, 2], [3, 4]]])),
        (np.array([[[0, 0], [0, 0]], [[1, 2], [3, 4]]]))
    ])
    def test_error_if_not_2_dimensional(self, input_array_like):
        with pytest.raises(ValueError,
                           match=r".* expected a matrix of dimension "
                           "either one or two.*"):
            to_2D_matrix(input_array_like)


if __name__ == "__main__":
    sys.exit(pytest.main(
        ["-v", "--cov=your_package", "--cov-report=term-missing"]))
