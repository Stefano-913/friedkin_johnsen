from pathlib import Path
import json
import os


def adjacency_default_params() -> dict:
    """ Returns a copy of the default configuration for the
        adjacency matrix at every call.
    """
    return {
        "file_name": None,
        "params": {
            "mean_node_degree": 6,
            "prob_rewire_edge": 0.1,
            "weights_variability": 0.5,
            "min_edge_weight": 0.01,
            "tries": 100,
            "seed": None,
        },
    }


def prejudice_default_params() -> dict:
    """ Returns a copy of the default configuration for the
        prejudice matrix at every call.
    """
    return {
        "file_name": None,
        "params": {
            "no_of_topics": 1,
            "mean_prejudice": 0.5,
            "std_prejudice": 0.25,
            "seed": None,
        },
    }


def susceptibility_default_params() -> dict:
    """ Returns a copy of the default configuration for the
        susceptibility matrix at every call.
    """
    return {
        "file_name": None,
        "params": {
            "mean_susceptibility": 0.5,
            "std_susceptibility": 0.25,
            "seed": None,
        },
    }


def default_config_file() -> dict:
    """ Returns a copy of the default configuration file at every call."""
    return {
        "mtx_files_directory": None,
        "number_of_nodes": 1000,
        "max_cycles": 10_000,
        "matrices": {
            "adjacency": adjacency_default_params(),
            "prejudice": prejudice_default_params(),
            "susceptibility": susceptibility_default_params(),
        },
    }


def build_config_file(file_name: str = "matrix_config.json",
                      relative_path: str = Path.cwd(),
                      overwrite: bool = False) -> None:
    """ Write the default configuration of the model parameters to
        a .json file.
        If the file already exists, the user can choose whether or not to
        replace it using the overwrite option.

        Parameters
        ----------
        file_name : str, default="matrix_config.json"
            Name of the configuration file to create.

        relative_path : str, optional (defaults to current directory)
            Directory in which the file should reside, missing directories are
            created.

        overwrite : bool, default=False
            In case a file with the same name is already present, this flag
            determines whether or not it should be supplanted (if True) or
            not (if False).

        Returns
        -------
        None
    """
    config_path = Path(relative_path) / file_name
    if config_path.exists() and not overwrite:
        return None

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(default_config_file(), indent=2))

    return None


def nested_dict_merge(default: dict, override: dict) -> dict:
    """ Recursively merge a dictionary with a default one,
        returning a new object.

        The result will start as a copy of "default", but every key that also
        appears in "override"" will take its value from there instead.
        Keys that appear in the default dictionary and not in the overrider
        will not see their value updated; keys that appear only in "override"
        will instead raise a warning without changing the default value.

        Parameters
        ----------
        default : dict
            The reference dictionary.

        override : dict
            The dictionary of user-specified values, only its keys that are
            also present in the default are used.

        Returns
        -------
        dict
            A new dictionary with the structure of the default one, where
            values have been replaced by those from "override" where provided.

        Raises
        ------
        ValueError
            If the overrider dictionary contains keys that are not present in
            "default" or if it replaces a dictionary of "default" with a value
            of another type.

        Notes
        -----
        In case of nested dictionaries, if a key maps to a dict in "default"
        but to a non-dict in "override" (or vice versa), the value of
        "override" replaces the default without further merging;
        while if they both are dicts, the merging proceeds going deeper
        in the nested keys.
    """
    result = dict(default)

    for key, value in override.items():
        if key not in default:
            raise ValueError(f"Unrecognized key '{key}' in the configuration, "
                             "please check for typos.")

        if isinstance(default[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"'{key}' must be a dictionary, "
                                 f"current value is '{value}'.")
            result[key] = nested_dict_merge(result[key], value)
        else:
            result[key] = value

    return result


def fill_out_config_file(file_path: str) -> dict:
    """ Complete a user configuration file with any missing default options.

        Reads the .json file and fills the missing entries with the default
        configuration as described in build_config_file()
        (the original file is modified in place).

        Parameters
        ----------
        file_path : str
            Path to an existing .json configuration file.

        Returns
        -------
        dict
            The merged configuration, that substitutes the one provided by
            the user.

        Raises
        ------
        FileNotFoundError
            If the file_path does not exist.
        ValueError
            If the user config file contains keys that are not present in the
            default options or if it replaces a default dictionary with a value
            of another type.

        See Also
        --------
        build_config_file()

        Notes
        -----
        The file is rewritten in place.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path) as f:
        user_config = json.load(f)

    merged_files = nested_dict_merge(default_config_file(), user_config)

    with open(file_path, "w") as f:
        json.dump(merged_files, f, indent=2)

    return merged_files
