import json
import os
from pathlib import Path
import numpy as np
from functools import wraps

from .matrix_conversion import to_2D_matrix


def file_loading(func):
    @wraps(func)
    def wrapper(file_name: str, relative_path: str = Path.cwd()) -> np.ndarray:
        file_path = os.path.join(relative_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")

        if os.path.getsize(file_path) == 0:
            raise ValueError(f"{file_path} is empty.")

        try:
            data = func(file_path)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error in the loading of {file_path}:"
                             f" {e}") from e

        matrix = to_2D_matrix(data)
        if matrix.size == 0:
            raise ValueError(f"{file_path} contains no data.")

        return matrix
    return wrapper


@file_loading
def load_matrix_from_csv(file_path: str) -> np.ndarray:
    r""" Loads a matrix from a .csv file and converts it to a 2D Numpy ndarray.
        Expected format is a set of numbers separated by commas where the rows
        are separated by newlines (e.g., "11, 12\n 21, 22").
        The matrix provided in the file must be compatible with np.array()
        conversion, must not contain complex numbers as well as NaN or
        infinite values.

        String version of lists are also accepted as long as they do not need
        to be evaluated and contain purely numeric values (e.g. sqrt(5) is
        not accepted).

        Parameters
        ----------
        file_path : str
            Path of the .csv file from which to import the matrix.

        Returns
        -------
        np.ndarray

        Raises
        ------
        FileNotFoundError
            If the file isn't in the inserted path or if its extension
            isn't correct.
        ValueError
            If the conversion to np.ndarray failed.
    """
    return np.loadtxt(file_path, delimiter=",", dtype=float)


@file_loading
def load_matrix_from_json(file_path: str) -> np.ndarray:
    """ Loads a matrix from a .json file and converts it to a 2D Numpy ndarray.
        Expected format of the data is analogous to a 2D python list
        (e.g. "[[11, 12], [21, 22]]").

        The data can be provided without any other structure (in which case,
        the entirety of the content of the file will be evaluated for
        conversion) or inside the standard .json format, in which case the
        data is expected to be under the "matrix" key.

        The matrix provided in the file must be compatible with np.array()
        conversion, must not contain complex numbers as well as NaN or
        infinite values.

        If the data is provided as a dictionary (and in that circumstance
        only), the data can also be provided as a string version
        (e.g. {"matrix": "[[11, 12], [21, 22]]"}).

        Parameters
        ----------
        file_path : str
            Path of the .json file from which to import the matrix.

        Returns
        -------
        np.ndarray

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist, is in the wrong inserted path or if its
            extension isn't correct.
        ValueError
            If the conversion to np.ndarray failed, due to either defective
            content (the accepted formats are list or dictionary).
        KeyError
            In case the data is provided as a dictionary, if there is no
            "matrix" key.
    """
    with open(file_path) as f:
        data = json.load(f)

        if isinstance(data, (list, int, float)):
            return data

        if not isinstance(data, dict):
            raise ValueError(f"The data in {file_path} must be either in "
                             f"list or dictionary format, currently of type "
                             f"'{type(data).__name__}'.")

        if "matrix" not in data:
            raise KeyError(f"Expected a 'matrix' key in {file_path}, "
                           f"found keys: {list(data.keys())}")
        return data["matrix"]


def load_matrix_from_file(file_name: str,
                          relative_path: str = Path.cwd()) -> np.ndarray:

    """ Parameters
        ----------
        file_path : str
            Path of the file from which to import the matrix
            (supported formats are .csv and .json).

        Returns
        -------
        np.ndarray

        Raises
        ------
        FileNotFoundError
            If the file isn't in the inserted path or if its extension isn't
            correct.
        ValueError
            If the conversion to np.ndarray failed, due to either defective
            data or a wrong corresponding key in the .json format
            (key must be "matrix").
    """
    file_extension = os.path.splitext(file_name)[1]

    if file_extension in ("", ".", ". "):
        raise ValueError(f"File '{file_name}' is without extension.")

    if file_extension == ".csv":
        return load_matrix_from_csv(file_name, relative_path)
    elif file_extension == ".json":
        return load_matrix_from_json(file_name, relative_path)
    else:
        raise ValueError(f"File '{file_name}' has an unsupported file "
                         f"extension: '{file_extension}'. Only .csv "
                         f"and .json are supported.")
