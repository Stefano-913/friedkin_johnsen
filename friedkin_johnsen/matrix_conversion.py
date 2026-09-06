import numpy as np
import numpy.typing as npt
import ast

from .error_messages import item_summary_for_error


def to_array(array: str | npt.ArrayLike,
             array_error_name: str = "") -> np.ndarray:
    """ Convert the input to a floating-point Numpy ndarray.

        Strings are also accepted, provided they are convertible
        to an array-like object and do not need to be evaluated
        (e.g. "2+5", "np.sin(1)" or "np.e" are not be considered valid inputs).

        Does not accept objects containing complex numbers, infinite values or
        NaNs.

        Parameters
        ----------
        array : str or array_like
            Array-like object (or string representing one) to be converted to
            Numpy ndarray.

        array_error_name : str, optional
            Name of the array to be used in error messages, by default an empty
             string.

        Returns
        -------
        array: np.ndarray
            A floating-point Numpy ndarray.

        Raises
        ------
        TypeError
            If the input cannot be converted to a floating-point ndarray
            or contains complex numbers.

        ValueError
            If the input object contains NaN, infinite values or if the string
            contains an invalid expression.

        SyntaxError
            If the input is entered by a string with a syntax error.
    """
    array_summary = item_summary_for_error(array, array_error_name)

    if isinstance(array, str):
        if array == "":
            return np.array([], dtype=float)

        try:
            array = ast.literal_eval(array)
        except (ValueError) as e:
            raise ValueError(f"String '{array_summary}' contains invalid "
                             f"expression that might need evaluation.\n"
                             f"{e}") from e

        except (SyntaxError) as e:
            raise SyntaxError(f"String '{array_summary}' contains a syntax "
                              f"error that invalidates the input.\n{e}") from e

    try:
        array = np.asarray(array)
    except (ValueError) as e:
        raise ValueError(f"{array_summary} cannot be "
                         f"converted to a matrix.\n{e}") from e

    if np.iscomplexobj(array):
        raise TypeError(f"{array_summary} contains complex values, "
                        f"which are not supported.")

    try:
        array = array.astype(float)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Values in {array_summary} cannot be "
                        f"converted to floating-point.\n{e}") from e

    if not np.isfinite(array).all():
        raise ValueError(f"{array_summary} contains NaN or infinite values.")

    return array


def to_2D_matrix(mtx: str | npt.ArrayLike,
                 array_error_name: str = "") -> np.ndarray:
    """ Converts the input to a floating point 2D matrix.
        Strings are also accepted, provided they are convertible to an
        array-like object.
        One dimensional matrices and scalars will also be accepted as
        degenerate cases and interpreted as column vectors.
        Objects that result in a dimensionality higher than 2 will not be
        accepted.

        Does not accept objects containing complex numbers, infinite
        values or NaNs.

        Parameters
        ----------
        mtx : str or array_like
            Array-like object (or string representing one) to be interpreted
            as a matrix. Must have a dimensionality of either one or two.

        Returns
        -------
        np.ndarray
            Returns a floating-point Numpy ndarray.

        Raises
        ------
        TypeError
            If the input cannot be converted to a floating-point ndarray
            or contains complex numbers.
        ValueError
            If the input object has dimensionality higher than 2 or
            if contains NaN or infinite values.

        See Also
        --------
        to_array : Convert input to a floating-point ndarray.
    """

    mtx = to_array(mtx, array_error_name=array_error_name)

    if mtx.ndim in (0, 1):
        mtx = mtx.reshape(-1, 1)

    elif mtx.ndim != 2:
        raise ValueError(f"In {item_summary_for_error(mtx, array_error_name)} "
                         f"expected a matrix of dimension equal or lower than two, "
                         f"got ndim={mtx.ndim}.")

    return mtx
