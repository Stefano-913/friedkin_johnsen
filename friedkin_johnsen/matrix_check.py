import numpy as np
from itertools import combinations
from functools import wraps
from .error_messages import truncate_item_for_error


def validate_2d_matrix(return_if_matrix_is_none=None):
    """ Decorator that groups what would otherwise be repetitive code
        across numerous functions below.
        In particular, handles None matrices as input and checks the
        matrices' type and dimension.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(matrix: np.ndarray | None = None, *args, **kwargs):
            if matrix is None:
                return return_if_matrix_is_none

            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Input is of {type(matrix)} type, "
                                "expected a np.ndarray matrix.")

            if matrix.ndim != 2:
                raise ValueError(f"Input is of {matrix.ndim} dimension, "
                                 "expected a 2D matrix.")
            return func(matrix, *args, **kwargs)
        return wrapper
    return decorator


@validate_2d_matrix(return_if_matrix_is_none=True)
def matrix_is_2D(matrix: np.ndarray | None = None) -> bool:
    """ Checks that the input matrix is 2D.
        Can also accept 'None' as argument, in which case the test will pass.

        Parameters
        ----------
        matrix : np.ndarray | None
            Matrix the dimension of which is to be evaluated.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            If the given matrix isn't 2D.
    """
    return True


@validate_2d_matrix(return_if_matrix_is_none=True)
def is_square_matrix(matrix: np.ndarray | None = None) -> bool:
    """ Checks if the input 2D matrix is of square shape (n,n).
        Can also accept 'None' as argument, in which case the test will pass.

        Parameters
        ----------
        matrix : np.ndarray | None
            Matrix the squareness of which is to be evaluated.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            If the given matrix isn't 2D.
    """
    rows, columns = matrix.shape
    return rows == columns


@validate_2d_matrix(return_if_matrix_is_none=None)
def get_number_of_users(matrix: np.ndarray | None = None) -> int:
    """ Checks the number of users that the matrix implies in the context
        of a Friedkin-Johnsen model, concretely corresponding to the
        number of rows of every matrix involved.
        Can also accept 'None' as argument, in which case the returned value
        will be 'None'.

        Parameters
        ----------
        matrix : np.ndarray | None
            Matrix from which the number of users is to be extracted.

        Returns
        -------
        no_of_users : int

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            If the given matrix isn't 2D.
    """
    no_of_users, _ = matrix.shape
    return no_of_users


@validate_2d_matrix(return_if_matrix_is_none=True)
def is_matrix_diag(matrix: np.ndarray | None = None) -> bool:
    """ Checks if the input 2D matrix is diagonal.
        Can also accept 'None' as argument, in which case the test will pass.

        Parameters
        ----------
        matrix : np.ndarray | None
            Matrix the diagonality of which is to be evaluated.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray type.
        ValueError
            If the given matrix isn't 2D.
    """

    if not is_square_matrix(matrix):
        return False

    return np.allclose(matrix, np.diag(np.diagonal(matrix)))

# ---------------


def matrices_are_compatible(
        adjacency_matrix: np.ndarray | None = None,
        prejudice_matrix: np.ndarray | None = None,
        susceptibility_matrix: np.ndarray | None = None) -> bool:
    """ Checks if the input matrices can be part of the same Friedkin-Johnsen
        model.
        The model is comprised of an adjacency matrix (W); a prejudice
        matrix (u) (alongside the opinion matrix (x), which represents its
        evolution) and a susceptibility matrix (Λ) that will need to be either
        summed or multiplied with each other.

        The formula (ΛWx + (I-Λ)u) demands the following properties:
        -   all matrices have same number of rows (or users);
        -   adjacency matrix W is square;
        -   susceptibility matrix Λ is square (and diagonal).

        These three requirements guarantee the possibility of
        multiplying/adding the matrices as required from the model.

        This function can also accommodate for 'None' inputs, interpreted as
        missing matrices that will not hinder the check for compatibility of
        the remaining ones.
        If all the matrices are 'None', the test passes.

        Parameters
        ----------
        adjacency_matrix : np.ndarray | None
            adjacency matrix (W)

        prejudice_matrix: np.ndarray | None
            prejudice matrix (u)

        susceptibility_matrix: np.ndarray | None
            susceptibility matrix (Λ)

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If any of the input matrices isn't of np.ndarray type.
        ValueError
            If any of the requirements listed above isn't met:
                - matrices have different number of rows;
                - adjacency matrix isn't square;
                - susceptibility matrix isn't square or isn't diagonal.
    """

    matrices = {
        "adjacency": adjacency_matrix,
        "prejudice": prejudice_matrix,
        "susceptibility": susceptibility_matrix,
    }

    valid_matrices = {matrix_name: matrix for matrix_name, matrix
                      in matrices.items() if matrix is not None}

    for matrix_name, matrix in valid_matrices.items():
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"{matrix_name} is of {type(matrix)} type, "
                            "expected a np.ndarray matrix.")

    if len(valid_matrices) > 1:
        for (matrix_a, matrix_b) in combinations(valid_matrices.values(), 2):
            users_a = get_number_of_users(matrix_a)
            users_b = get_number_of_users(matrix_b)

            if users_a != users_b:

                # The error message outputs the number of users of all
                # matrices, so that it can give a clearer picture.
                error_users_overview = ";\n".join(
                    f"-\t{matrix_name}: {get_number_of_users(matrix)} users"
                    for matrix_name, matrix in valid_matrices.items()
                    )

                raise ValueError(f"Input matrices have an incompatible "
                                 "numbers of users.\n"
                                 f"{error_users_overview}.")

    if not is_square_matrix(adjacency_matrix):
        raise ValueError("Inserted adjacency matrix isn't "
                         "a square matrix.")
    if not is_square_matrix(susceptibility_matrix):
        raise ValueError("Inserted susceptibility matrix isn't "
                         "a square matrix.")
    if not is_matrix_diag(susceptibility_matrix):
        raise ValueError("Inserted susceptibility matrix isn't "
                         "a diagonal matrix.")

    # If 'prejudice' is the only valid matrix, its dimension would
    # never be checked otherwise
    matrix_is_2D(prejudice_matrix)

    return True


def has_matrix_zero_row(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """ Checks if the input matrix has any row comprised exclusively of "zero"
        values. Also applicable to vectors, interpreted as a degenerate matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix or vector whose rows will be evaluated to check if there
            are any comprised exclusively of "zeros".

        tolerance : float, optional
            Tolerance for considering a floating-point value as a zero,
            by default 1e-10.

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            In case the size of the "matrix" parameter isn't compatible with
            either a matrix or a vector.
    """
    if type(tolerance) not in (int, float):
        raise TypeError(f"Tolerance is of {type(tolerance)} type, "
                        "expected a floating-point value")
    if tolerance < 0:
        tolerance = abs(tolerance)

    if matrix.ndim == 1:
        axis = 0
    elif matrix.ndim == 2:
        axis = 1
    else:
        raise ValueError(f"{truncate_item_for_error(matrix)} has unexpected "
                         f"shape {matrix.shape}, incompatible with a matrix "
                         "or a vector.")

    return ((abs(matrix) < tolerance).all(axis=axis)).any()
