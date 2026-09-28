import numpy as np
import warnings
from .matrix_check import (validate_2d_matrix,
                           get_number_of_users,
                           has_matrix_zero_row,
                           is_matrix_diag,
                           is_square_matrix)


@validate_2d_matrix(return_if_matrix_is_none=None)
def make_adjacency_matrix_stochastic(
        matrix: np.ndarray[float] | None = None
        ) -> None:
    """ Normalizes in place the rows of the input floating-point matrix as
        to render it stochastic.

        The matrix is expected to have at least a non-zero entry in all of its
        rows to permit the row normalization; also every entry will have its
        sign changed to positive before the normalization.

        Parameters
        ----------
        matrix : np.array
            Floating-point matrix to be rendered stochastic

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            - If the given matrix isn't 2D;
            - If the given matrix isn't square;
            - If the dtype of the matrix values isn't floating point;
            - If any of the matrix rows are comprised entirely of zero values.

        Warns
        -----
        UserWarning
            If the matrix contains negative values, which will be converted to
            positive before normalization.
    """

    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("Expected np.array of floating point numbers, "
                         "the dtype of the provided matrix is "
                         f"{matrix.dtype} instead.")

    if not is_square_matrix(matrix):
        raise ValueError("Expected square matrix, the size "
                         f"of the provided matrix is {matrix.size} instead.")

    if has_matrix_zero_row(matrix):
        raise ValueError("Adjacency matrix is impossible to render "
                         "stochastic, as one (or more) of its rows is "
                         "comprised entirely of zero values.")

    if matrix.min() < 0:
        np.absolute(matrix, out=matrix)
        warnings.warn("Some negative values were detected in the input "
                      "adjacency matrix, they have been automatically "
                      "converted to positive.", UserWarning)

    matrix /= matrix.sum(axis=1, keepdims=True)

    return None


@validate_2d_matrix(return_if_matrix_is_none=None)
def normalize_prejudice_matrix(
        matrix: np.ndarray[float] | None = None
        ) -> None:
    """ Normalizes in place the values of a floating-point prejudice
        matrix to [0, 1].

        The entries of the matrix are rescaled so that the [0,1] interval
        is used to the fullest.
        If all entries are equal, the values are set to all ones.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix of floating-point values

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            - If the given matrix isn't 2D;
            - If the dtype of the matrix values isn't floating point.
    """
    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("Expected np.array of floating point numbers, "
                         "the dtype of the provided matrix is "
                         f"{matrix.dtype} instead.")

    max_value = matrix.max()
    min_value = matrix.min()

    if not np.allclose(min_value, max_value):
        matrix[:] = matrix - min_value / (max_value-min_value)
    else:
        n_users = get_number_of_users(matrix)
        matrix[:] = np.full((n_users, n_users), 1)

    return None


@validate_2d_matrix(return_if_matrix_is_none=None)
def normalize_susceptibility_diagonal(
        matrix: np.ndarray[float] | None = None
        ) -> None:
    """ Normalizes in place the diagonal values of a floating-point
        susceptibility matrix to [0, 1].

        The diagonal entries of the matrix are rescaled so that the [0,1]
        interval is used to the fullest.
        If all diagonal entries are equal, the diagonal is set to all ones.

        Parameters
        ----------
        matrix : np.ndarray
            Square diagonal matrix of floating-point values

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            - If the given matrix isn't 2D;
            - If the given matrix isn't diagonal;
            - If the dtype of the matrix values isn't floating point.
    """
    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("Expected np.array of floating point numbers, "
                         "the dtype of the provided matrix is "
                         f"{matrix.dtype} instead.")

    if not is_square_matrix(matrix):
        raise ValueError("Expected square matrix, the size "
                         f"of the provided matrix is {matrix.size} instead.")

    if not is_matrix_diag(matrix):
        raise ValueError("Expected a diagonal matrix.")

    diag_values = matrix.diagonal()
    max_value = diag_values.max()
    min_value = diag_values.min()

    if not np.allclose(min_value, max_value):
        np.fill_diagonal(matrix,
                         matrix.diagonal() - min_value / (max_value-min_value))
    else:
        n_users = get_number_of_users(matrix)
        np.fill_diagonal(matrix, np.full((n_users), 1))

    return None
