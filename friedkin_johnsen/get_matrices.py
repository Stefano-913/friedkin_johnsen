from .build_rng_matrices import (build_adjacency_matrix,
                                 build_prejudice_matrix,
                                 build_susceptibility_matrix)

from .load_matrices_from_files import load_matrix_from_file

from .matrix_check import (matrices_are_compatible,
                           get_number_of_users,
                           has_matrix_zero_row)

from .matrix_utils import (make_adjacency_matrix_stochastic,
                           normalize_prejudice_matrix,
                           normalize_susceptibility_diagonal)
import numpy as np
from pathlib import Path
import json
from scipy.sparse import csr_array, dia_array, issparse


def load_matrices(matrix_names: list,
                  config_dict: dict,
                  directory: str | None = Path.cwd()
                  ) -> dict[str, np.ndarray | None]:
    """ Loads the matrices whose names are listed based on the paths
        indicated in the configuration dictionary.

        Each matrix's file path is constructed by joining the shared directory
        with its "file_name" entry in config_dict.
        The shared directory is the "mtx_files_directory" entry of config_dict
        (resolved against the input directory if it is relative) or the
        input directory itself if that entry is None.

        Parameters
        ----------
        matrix_names : list of strings
            Names of the matrices to load, each of which must appear as a key
            under config_dict["matrices"].

        config_dict : dict
            Configuration file (e.g. parsed from a JSON file), required to
            have the following structure:

                {
                    "mtx_files_directory": <shared directory or None>,
                    "matrices": {
                        "A_matrix_name": {"file_name": <A's file or None>},
                        "B_matrix_name": {"file_name": <B's file or None>},
                    }
                }

        directory : Path, optional
            Fallback directory, only used when "mtx_files_directory" in
            config_dict is None (default value is the current working
            directory).

        Returns
        -------
        dict of {str: numpy.ndarray or None}
            Connects each name in matrix_names to its loaded matrix
            (if a matrix's file_name entry is None, a None will be in the
            dictionary instead of the loaded matrix).

        Raises
        ------
        KeyError
            If "mtx_files_directory" or "matrices" keys are missing from
            config_dict; if a name in matrix_names has no entry or if said
            entry has no "file_name" key.
    """

    if "matrices" not in config_dict:
        raise KeyError("The configuration file provided is missing the "
                       "required key 'matrices'.")
    if "mtx_files_directory" not in config_dict:
        raise KeyError("The configuration file provided is missing the "
                       "required key 'mtx_files_directory'.")

    matrices_not_in_config = [name for name in matrix_names
                              if name not in config_dict["matrices"]]

    if len(matrices_not_in_config) >= 1:
        missing_matrices = [matrix for matrix in matrices_not_in_config]
        missing_matrices_error = ",\n".join(missing_matrices)
        raise KeyError(
            "The following matrices' names do not appear in the configuration "
            f"file under 'config_dict['matrices']':\n{missing_matrices_error}."
        )
    # wo = without
    matrices_wo_file_name = [n for n in matrix_names
                             if "file_name" not in config_dict["matrices"][n]]

    if len(matrices_wo_file_name) >= 1:
        matrices_wo_file = [matrix for matrix in matrices_wo_file_name]
        matrices_wo_file_error = ",\n".join(matrices_wo_file)
        raise KeyError(
            "The following matrices lack the 'file_name' key in the "
            f"configuration file: {matrices_wo_file_error}."
        )

    loaded_matrices = dict([(matrix_name, None)
                            for matrix_name in matrix_names])

    if config_dict["mtx_files_directory"] is not None:
        relative_dir = Path(directory) / config_dict["mtx_files_directory"]
        directory = (relative_dir).resolve()

    for matrix_name in matrix_names:
        matrix_config = config_dict["matrices"][matrix_name]

        if matrix_config["file_name"] is not None:
            loaded_matrices[matrix_name] = load_matrix_from_file(
                file_name=matrix_config["file_name"],
                relative_path=directory)

    return loaded_matrices


def load_and_validate_matrices_info(matrix_names: list,
                                    config_params: dict,
                                    mtx_directory: str | Path
                                    ) -> tuple[int, dict]:
    """ Loads the matrices whose names are listed and validates their
        properties (the accepted matrix names are "adjacency", "prejudice"
        and "susceptibility").

        The loading is done for each matrix according to the file name
        indicated in the configuration file, with the matrices expected
        to be in the input directory.

        Parameters
        ----------
        matrix_names : list of strings
            Names of the matrices to load. Accepted names are "adjacency",
            "prejudice" and "susceptibility".

        config_params : dict
            Configuration file (e.g. parsed from a JSON file), required to have
            the following structure:

                {
                    "mtx_files_directory": <shared directory or None>,
                    "matrices": {
                        "A_matrix_name": {"file_name": <A's file or None>},
                        "B_matrix_name": {"file_name": <B's file or None>},
                    }
                }

        mtx_directory : Path | str, optional
            Directory where it's expected to find the  matrix files.

        Returns
        -------
        (no_of_users, loaded_matrices) : (int, dict[str, np.ndarray | None])
            Returns a tuple with the number of users and a dictionary that
            associates each matrix expected in the model ("adjacency",
            "prejudice" and "susceptibility") to its loading from the
            corresponding file.
            (if a matrix's file_name entry in the configuration file is None
            or its name is missing in matrix_names, a None will be in the
            dictionary instead of the loaded matrix).

        Raises
        ------
        ValueError
            If unexpected matrices are listed in "matrix_names" that go beyond
            the model's necessity; if the loaded adjacency matrix has a row
            comprised exlusively of zero values.
    """

    if not set(matrix_names) <= {"adjacency", "prejudice", "susceptibility"}:
        raise ValueError("Input matrix names go beyond the expected names "
                         "for the scope of the program: "
                         "('adjacency', 'prejudice', 'susceptibility').")

    loaded_matrices = {name: None for name in
                       ["adjacency", "prejudice", "susceptibility"]}
    loaded_matrices.update(load_matrices(matrix_names,
                                         config_params,
                                         mtx_directory))

    matrices_are_compatible(**loaded_matrices)

    valid_matrices = [mtx for mtx in loaded_matrices.values()
                      if mtx is not None]
    no_of_users = (get_number_of_users(valid_matrices[0])
                   if valid_matrices else None)

    if loaded_matrices["adjacency"] is not None:
        if has_matrix_zero_row(loaded_matrices["adjacency"]):
            raise ValueError("adjacency matrix has a row comprised "
                             "exclusively of zero values, impossible "
                             "to render stochastic")

    return (no_of_users, loaded_matrices)


def adjust_matrix_type(matrix_name: str,
                       matrix: np.ndarray,
                       matrix_types: dict) -> dict:
    """ Converts a matrix to the type that matrix_types assigns to its name
        (if said matrix already has the desired type it is returned as is).
        The supported types are csr_array, np.ndarray and dia_array.

        Parameters
        ----------
        matrix_name : str
            Name of the matrix to convert, which must be a key of matrix_types.

        matrix : np.ndarray | csr_array | dia_array
            Matrix to convert.

        matrix_types : dict
            Connects each matrix name to its desired type (csr_array,
            np.ndarray or dia_array).

        Returns
        -------
        np.ndarray | csr_array | dia_array
            The converted matrix.

        Raises
        ------
        ValueError
            If any type in matrix_types isn't supported; if matrix_name isn't
            a key of matrix_types or if the matrix is None.
    """

    converters = {
        csr_array: csr_array,
        np.ndarray: lambda m: m.toarray() if issparse(m) else np.asarray(m),
        dia_array: dia_array
    }

    for matrix_type in matrix_types.values():
        if matrix_type not in converters.keys():
            raise ValueError(f"{matrix_type} type not covered")

    if matrix_name not in matrix_types.keys():
        raise ValueError(f"{matrix_name} unspecified type")

    if matrix is None:
        raise ValueError("All matrices should've been loaded")

    desired_matrix_type = matrix_types[matrix_name]
    if isinstance(matrix, desired_matrix_type):
        return matrix

    matrix = converters[desired_matrix_type](matrix)

    return matrix


def get_matrices(
        config_file: str | None = "matrix_config.json",
        directory: str | None = Path.cwd()
        ) -> dict[str, csr_array | np.ndarray | dia_array]:

    """ Gets the adjacency, prejudice and susceptibility matrices of the
        Friedkin-Johnsen model from a JSON configuration file.

        The matrices with a "file_name" entry in the configuration file
        are loaded, validated, normalized and lastly converted to
        their most optimal type:
        -   csr_array for adjacency;
        -   np.ndarray for prejudice and
        -   dia_array for susceptibility.

        The matrices whose "file_name" is None are generated by their
        respective builder using the individual "params" in the same
        configuration file (the matrices built this way already satisfy
        the required constraints).
        The number of users is either implied by the
        correctly loaded matrices or by the "number_of_nodes" entry in
        the file if no matrix was loaded.

        Parameters
        ----------
        config_file : str, optional
            Name of the JSON configuration file located in "directory"
            (default is "matrix_config.json").

        directory : Path or str, optional
            Directory where the configuration file is and also the
            reference point for a relative "mtx_files_directory" value
            (default value is the current working directory).

        Returns
        -------
        dict of {str: csr_array, numpy.ndarray or dia_array}
            Connects "adjacency", "prejudice" and "susceptibility" to their
            matrix.

        Raises
        ------
        FileNotFoundError
            If the configuration file or a matrix file doesn't exist.
        KeyError
            If a required key is missing from the configuration file
            ("mtx_files_directory", "matrices", "number_of_nodes" when no
            matrix is loaded or a generated matrix's "params").
    """

    matrix_names = ["adjacency", "prejudice", "susceptibility"]
    matrix_builders = {"adjacency": build_adjacency_matrix,
                       "prejudice": build_prejudice_matrix,
                       "susceptibility": build_susceptibility_matrix}

    matrix_types = {"adjacency": csr_array,
                    "prejudice": np.ndarray,
                    "susceptibility": dia_array}

    matrix_normalizers = {"adjacency": make_adjacency_matrix_stochastic,
                          "prejudice": normalize_prejudice_matrix,
                          "susceptibility": normalize_susceptibility_diagonal}

    directory = Path(directory).resolve()
    with open(directory / config_file) as f:
        config_params = json.load(f)

    matrices_config = config_params["matrices"]

    no_of_users, matrices = load_and_validate_matrices_info(matrix_names,
                                                            config_params,
                                                            directory)

    if no_of_users is None:
        no_of_users = config_params["number_of_nodes"]

    for mtx_name in matrix_names:
        if matrices[mtx_name] is None:
            matrices[mtx_name] = matrix_builders[mtx_name](
                nodes=no_of_users,
                **matrices_config[mtx_name]["params"]
                )
        else:
            matrix_normalizers[mtx_name](matrices[mtx_name])
            matrices[mtx_name] = adjust_matrix_type(mtx_name,
                                                    matrices[mtx_name],
                                                    matrix_types)

    return matrices
