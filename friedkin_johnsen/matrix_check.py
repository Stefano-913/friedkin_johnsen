import numpy as np
from scipy.sparse import sparray, issparse, linalg, diags
from itertools import combinations
from functools import wraps
from .error_messages import truncate_item_for_error


def validate_2d_matrix(return_if_matrix_is_none=None):
    """ Decorator that validates a function's matrix input.
        The wrapped function is expected to have a 2D object
        as first positional argument.

        The decorator groups the checking of None as input and of the
        matrix type (np.ndarray or sp.sparray), as well as its
        dimension (dim=2).

        Parameters
        ----------
        return_if_matrix_is_none : any, optional
            Value to return immediately if the matrix is None, bypassing the
            decorated function entirely (default is None).

        Returns
        -------
        Callable
            A decorator that wraps a function.

        Raises
        ------
        TypeError
            If the matrix is not None and not an instance of either np.ndarray
            or sp.sparray.
        ValueError
            If the matrix isn't 2D.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(matrix: np.ndarray | sparray | None = None,
                    *args,
                    **kwargs):
            if matrix is None:
                return return_if_matrix_is_none

            if not isinstance(matrix, (np.ndarray, sparray)):
                raise TypeError(f"Input is of {type(matrix)} type, "
                                "expected a np.ndarray or sp.sparray matrix.")

            if matrix.ndim != 2:
                raise ValueError(f"Input is of {matrix.ndim} dimension, "
                                 "expected a 2D matrix.")
            return func(matrix, *args, **kwargs)
        return wrapper
    return decorator


@validate_2d_matrix(return_if_matrix_is_none=True)
def matrix_is_2D(matrix: np.ndarray | sparray | None = None) -> bool:
    """ Checks that the input matrix is 2D.
        Can also accept 'None' as argument, in which case the test will pass.

        Parameters
        ----------
        matrix : np.ndarray | sp.sparray | None
            Matrix the dimension of which is to be evaluated.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray or sp.sparray type.
        ValueError
            If the given matrix isn't 2D.
    """
    return True


@validate_2d_matrix(return_if_matrix_is_none=True)
def is_square_matrix(matrix: np.ndarray | sparray | None = None) -> bool:
    """ Checks if the input 2D matrix is of square shape (n,n).
        Can also accept 'None' as argument, in which case the test will pass.

        Parameters
        ----------
        matrix : np.ndarray | sp.sparray | None
            Matrix the squareness of which is to be evaluated.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray or sp.sparray type.
        ValueError
            If the given matrix isn't 2D.
    """
    rows, columns = matrix.shape
    return rows == columns


@validate_2d_matrix(return_if_matrix_is_none=None)
def get_number_of_users(matrix: np.ndarray | sparray | None = None) -> int:
    """ Checks the number of users that the matrix implies in the context
        of a Friedkin-Johnsen model, concretely corresponding to the
        number of rows of every matrix involved.
        Can also accept 'None' as argument, in which case the returned value
        will be 'None'.

        Parameters
        ----------
        matrix : np.ndarray | sp.sparray | None
            Matrix from which the number of users is to be extracted.

        Returns
        -------
        no_of_users : int

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray or sp.sparray type.
        ValueError
            If the given matrix isn't 2D.
    """
    no_of_users, _ = matrix.shape
    return no_of_users


@validate_2d_matrix(return_if_matrix_is_none=True)
def is_matrix_diag(matrix: np.ndarray | sparray | None = None) -> bool:
    """ Checks if the input 2D matrix is diagonal.
        Can also accept 'None' as argument, in which case the test will pass.

        Parameters
        ----------
        matrix : np.ndarray | sp.sparray | None
            Matrix the diagonality of which is to be evaluated.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray or sp.sparray type.
        ValueError
            If the given matrix isn't 2D.
    """

    if not is_square_matrix(matrix):
        return False

    if isinstance(matrix, np.ndarray):
        return np.allclose(matrix, np.diag(np.diagonal(matrix)))

    if isinstance(matrix, sparray):
        return (matrix - diags(matrix.diagonal().astype(float))).nnz == 0


@validate_2d_matrix(return_if_matrix_is_none=None)
def highest_eigenvalues(matrix: np.ndarray | sparray,
                        no_eigens: int = None,
                        dense_limit: int = 1000) -> np.ndarray:
    """ Computes the N highest eigenvalues in magnitude of the input 2D matrix.
        The solver algorithms for the dense or the sparse formats are chosen
        according to the most optimal/efficient scenario.

        The dense solver is used in the following cases:
        - if the matrix has a number of users lower than 'dense_limit' or
        - if the number of requested eigenvalues 'no:eigens' is = matrix size.
        For matrices with a higher number of users, when applicable, the sparse
        notation is favoured.

        Can also accept a 'None' matrix as argument, in which case the returned
        value will also be 'None'.

        Parameters
        ----------
        matrix : np.ndarray | sp.sparray | None
            Matrix the eigenvalues of which are to be computed.

        no_eigens : int, default = None
            Number of eigenvalues to compute, required to be positive
            and lower than the number of users.
            If left as None, the entirety of the eigenvalues will be
            computed.

        dense_limit : int, default = 1000
            Number of users below which the dense solver is used,
            expected to be non-negative.

        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues of the matrix, ordered by decreasing magnitude.

        Raises
        ------
        TypeError
            If the input matrix isn't of np.ndarray or sp.sparray type;
            if no_eigs is lower than zero or above the matrix dimension
            or if dense_limit is lower than zero.
        ValueError
            If the given matrix isn't 2D, if either no_eigs or dense_limit
            aren't int type.
        LinAlgError
            If the algorithm for the dense format doesn't converge.
        ArpackNoConvergence
            If the algorithm for the sparse format doesn't converge.
    """
    no_users = get_number_of_users(matrix)

    if no_users == 0:
        return None

    if no_eigens is None:
        no_eigens = get_number_of_users(matrix)

    if not isinstance(no_eigens, int):
        raise TypeError(f"Input is of {type(no_eigens)} type, "
                        "expected an int type.")
    if no_eigens <= 0:
        raise ValueError(f"Number of eigenvalues requested is {no_eigens}, "
                         "expected a positive number.")
    if no_eigens > no_users:
        raise ValueError(f"Number of eigenvalues requested is {no_eigens}, "
                         "expected it to be equal or lower than number of "
                         f"users ({no_users}).")

    if not isinstance(dense_limit, int):
        raise TypeError(f"Input is of {type(dense_limit)} type, "
                        "expected an int type.")
    if dense_limit < 0:
        raise ValueError(f"Dense limit is {dense_limit}, expected a "
                         "non-negative value.")

    if (no_users < dense_limit or
            no_eigens >= no_users - 1):
        matrix = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
        eigenvalues = np.linalg.eigvals(matrix)

    else:
        eigenvalues = linalg.eigs(matrix.astype(float),
                                  k=no_eigens,
                                  which="LM",
                                  return_eigenvectors=False
                                  )
    # To order by magnitude while maintaining the correct sign,
    # pass through indices.
    indices = np.argsort(-np.abs(eigenvalues))[:no_eigens]
    return eigenvalues[indices]

# ---------------


def matrices_are_compatible(
        adjacency: np.ndarray | None = None,
        prejudice: np.ndarray | None = None,
        susceptibility: np.ndarray | None = None) -> bool:
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
        adjacency : np.ndarray | None
            adjacency matrix (W)

        prejudice: np.ndarray | None
            prejudice matrix (u)

        susceptibility: np.ndarray | None
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
        "adjacency": adjacency,
        "prejudice": prejudice,
        "susceptibility": susceptibility,
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

    if not is_square_matrix(adjacency):
        raise ValueError("Inserted adjacency matrix isn't "
                         "a square matrix.")
    if not is_square_matrix(susceptibility):
        raise ValueError("Inserted susceptibility matrix isn't "
                         "a square matrix.")
    if not is_matrix_diag(susceptibility):
        raise ValueError("Inserted susceptibility matrix isn't "
                         "a diagonal matrix.")

    # If 'prejudice' is the only valid matrix, its dimension would
    # never be checked otherwise
    matrix_is_2D(prejudice)

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
