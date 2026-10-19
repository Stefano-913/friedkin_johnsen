from scipy.sparse import csr_array, dia_array
import pytest
import sys
import numpy as np
import json

from ..get_matrices import (load_matrices,
                            load_and_validate_matrices_info,
                            adjust_matrix_type,
                            get_matrices)

from ..handle_config_file import default_config_file, build_config_file


def write_matrix_files(matrix_dict, matrices_directory, config_params):
    """ Save a matrix in a file and point at said file
        in the configuration.
    """

    matrices_directory.mkdir(parents=True, exist_ok=True)

    for matrix_name, matrix in matrix_dict.items():
        file_name = f"{matrix_name}.json"
        (matrices_directory / file_name).write_text(matrix)
        config_params["matrices"][matrix_name]["file_name"] = file_name


def save_config(config_params, config_directory, file_name):
    # Since the JSON format is incompatible with Paths, one
    # needs to convert the Paths to string before the save

    matrices_directory = config_params["mtx_files_directory"]

    if matrices_directory is None:
        config_params_for_json = config_params
    else:
        config_params_for_json = {
            **config_params,
            "mtx_files_directory": str(matrices_directory)
            }

    with open(config_directory / file_name, "w") as f:
        json.dump(config_params_for_json, f)


def simulate_config_file(tmp_path,
                         matrix_dict,
                         no_of_nodes=None,
                         file_name="matrix_config.json"):

    """ Create a configuration file and matrix files in tmp_path
        so that they can later be recovered from the functions to test.
        Config_file will be created under tmp_path/file, while the
        matrix files will be under tmp_path/file next to it.

        Directory layout created is the following:

        tmp_path/
        ├── file/
        │   └── <config_param>.json
        └── matrices/
            └── <matrix_file>.json

        Returns (config_params, matrices_directory) for ease of testing.
    """

    config_directory = tmp_path / "file"
    build_config_file(file_name=file_name, relative_path=config_directory)

    with open(config_directory / file_name) as f:
        config_params = json.load(f)

        matrices_directory = tmp_path / "matrices"
        config_params["mtx_files_directory"] = matrices_directory

        write_matrix_files(matrix_dict, matrices_directory, config_params)

        if no_of_nodes is not None:
            config_params["number_of_nodes"] = no_of_nodes

    save_config(config_params, config_directory, file_name)

    return config_params, matrices_directory

# ---------------


class TestLoadMatrices:
    def test_correct_with_default_config(self, tmp_path):
        config_params = simulate_config_file(tmp_path, {})[0]
        loaded_matrices = load_matrices(
            ["adjacency", "prejudice", "susceptibility"],
            config_params
            )

        assert loaded_matrices == {"adjacency": None,
                                   "prejudice": None,
                                   "susceptibility": None}

    @pytest.mark.parametrize("matrix_in_file, result", [
        ("[0, 1, 0]", [[0], [1], [0]]),
        ("[[0], [3.14], [0]]", [[0], [3.14], [0]]),
        ("[[1, 2], [3, 4], [5, 6]]", [[1, 2], [3, 4], [5, 6]]),
        ("[[1, 2, 3], [4, 5, 6]]", [[1, 2, 3], [4, 5, 6]])
    ])
    @pytest.mark.parametrize("matrix_name", [
        "adjacency", "prejudice", "susceptibility"
    ])
    def test_correctly_load_matrix(self,
                                   matrix_in_file,
                                   result,
                                   matrix_name,
                                   tmp_path,
                                   subtests):
        config_params = simulate_config_file(tmp_path,
                                             {f"{matrix_name}": matrix_in_file}
                                             )[0]
        loaded_matrices = load_matrices(
            ["adjacency", "prejudice", "susceptibility"],
            config_params
            )

        with subtests.test(msg="matrix is correctly loaded"):
            assert np.allclose(loaded_matrices[matrix_name], result)

        with subtests.test(msg="matrices left as default are None"):
            default_matrices = [matrix for matrix
                                in ["adjacency", "prejudice", "susceptibility"]
                                if matrix != matrix_name]

            assert all([loaded_matrices[matrix] is None
                        for matrix in default_matrices])

    def test_correct_with_relative_mtx_files_directory(self, tmp_path):
        config_params, matrix_dir = simulate_config_file(
            tmp_path,
            {"adjacency": "[[0, 1], [1, 0]]"}
            )

        # A directory can be expressed in relative terms to the directory
        # given to load_matrices (here matrix_dir = tmp_path/matrices).
        config_params["mtx_files_directory"] = "../matrices"
        loaded_matrices = load_matrices(["adjacency"],
                                        config_params,
                                        matrix_dir)

        assert np.allclose(loaded_matrices["adjacency"], [[0, 1], [1, 0]])

    @pytest.mark.parametrize("config_dir, relative_dir, expected_dir", [
        pytest.param("file", ".", "file",
                     id="same directory"),

        pytest.param("file/sub", "../../new_folder", "new_folder",
                     id="up two levels"),

        pytest.param("file", "new_folder/nested", "file/new_folder/nested",
                     id="down into new folders"),

        pytest.param("file", "./new_folder/../old_folder", "file/old_folder",
                     id="path with . and ..")
    ])
    def test_correct_with_relative_mtx_files_directory_variants(
            self, tmp_path, config_dir, relative_dir, expected_dir):
        # config_dir:    directory of the config_file;
        # relative_dir: value of "mtx_files_directory" in the config
        #               (relative to config_subdir);
        # expected_dir:  where the matrix file really is.

        config_directory = tmp_path / config_dir
        config_directory.mkdir(parents=True, exist_ok=True)
        config_params = simulate_config_file(tmp_path, {})[0]

        write_matrix_files({"adjacency": "[[0, 1], [1, 0]]"},
                           tmp_path / expected_dir,
                           config_params)

        config_params["mtx_files_directory"] = relative_dir
        loaded_matrices = load_matrices(["adjacency"],
                                        config_params,
                                        config_directory)

        assert np.allclose(loaded_matrices["adjacency"], [[0, 1], [1, 0]])

    def test_raises_matrix_names_not_in_file(self):
        config = default_config_file()

        with pytest.raises(KeyError, match="do not appear"):
            load_matrices(["not_in_file"], config)

    def test_raises_missing_mtx_files_directory_key(self):
        config = default_config_file()
        del config["mtx_files_directory"]

        with pytest.raises(KeyError, match="mtx_files_directory"):
            load_matrices(["adjacency"], config)

    def test_raises_without_matrix_key(self):
        config = default_config_file()
        del config["matrices"]

        with pytest.raises(KeyError, match="matrices"):
            load_matrices(["adjacency", "prejudice"], config)

    def test_raises_matrix_without_file_name_key(self):
        config = default_config_file()
        del config["matrices"]["prejudice"]["file_name"]

        with pytest.raises(KeyError, match="prejudice"):
            load_matrices(["adjacency", "prejudice"], config)


class TestLoadAndValidateMatricesInfo:
    def test_correct_with_default_config(self, tmp_path):
        config_params, matrix_dir = simulate_config_file(tmp_path, {})

        no_of_users, loaded_matrices = load_and_validate_matrices_info(
            ["adjacency", "prejudice", "susceptibility"],
            config_params,
            matrix_dir)

        assert no_of_users is None
        assert loaded_matrices == {"adjacency": None,
                                   "prejudice": None,
                                   "susceptibility": None}

    def test_correct_with_user_matrices(self, tmp_path, subtests):
        config_params, matrix_dir = simulate_config_file(
            tmp_path,
            {"adjacency": "[[0, 1, 0], [1, 0, 1], [0, 1, 0]]",
             "susceptibility": "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"})

        no_of_users, loaded_matrices = load_and_validate_matrices_info(
            ["adjacency", "prejudice", "susceptibility"],
            config_params,
            matrix_dir
            )

        with subtests.test(msg="number of users is correct"):
            assert no_of_users == 3

        with subtests.test(msg="matrices are loaded"):
            assert np.allclose(loaded_matrices["adjacency"],
                               [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            assert np.allclose(loaded_matrices["susceptibility"],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        with subtests.test(msg="matrix left as default is None"):
            assert loaded_matrices["prejudice"] is None

    def test_correct_with_unrequested_matrices(self, tmp_path):
        config_params, mtx_dir = simulate_config_file(
            tmp_path,
            {"susceptibility": "[[1, 0], [0, 1]]"})

        no_of_users, loaded_matrices = load_and_validate_matrices_info(
            ["susceptibility"], config_params, mtx_dir)

        assert no_of_users == 2
        assert loaded_matrices["adjacency"] is None
        assert loaded_matrices["prejudice"] is None

    def test_raises_incompatible_matrices(self, tmp_path):
        config_params, mtx_dir = simulate_config_file(
            tmp_path,
            {"adjacency": "[[0, 1, 0], [1, 0, 1], [0, 1, 0]]",
             "susceptibility": "[[0.1], [0.2]]"})

        with pytest.raises(ValueError, match="incompatible"):
            load_and_validate_matrices_info(
                ["adjacency", "susceptibility"], config_params, mtx_dir)

    def test_raises_adjacency_with_zero_row(self, tmp_path):
        config_params, matrix_dir = simulate_config_file(
            tmp_path,
            {"adjacency": "[[0, 1, 0], [0, 0, 0], [0, 1, 0]]"}
            )

        with pytest.raises(ValueError, match="zero values"):
            load_and_validate_matrices_info(["adjacency"],
                                            config_params,
                                            matrix_dir)

    def test_raises_unexpected_matrix_name(self, tmp_path):
        config_params, matrix_dir = simulate_config_file(tmp_path, {})

        with pytest.raises(ValueError, match="expected names"):
            load_and_validate_matrices_info(["not_a_matrix"],
                                            config_params,
                                            matrix_dir)


class TestAdjustMatrixType:
    @pytest.mark.parametrize("input_matrix", [
        [[0, 1], [1, 1], [0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, -np.pi, 0], [1, np.sqrt(2), 1], [0, 1+2j, 0]]
    ])
    @pytest.mark.parametrize("convert_to_start_type",
                             [np.asarray, csr_array, dia_array]
                             )
    @pytest.mark.parametrize("desired_type",
                             [np.ndarray, csr_array, dia_array]
                             )
    def test_correctly_converts_matrix(self,
                                       input_matrix,
                                       convert_to_start_type,
                                       desired_type,
                                       subtests):

        matrix = convert_to_start_type(input_matrix)

        result = adjust_matrix_type("adjacency", matrix,
                                    {"adjacency": desired_type})

        with subtests.test(msg="matrix has the desired type"):
            assert isinstance(result, desired_type)

        with subtests.test(msg="matrix values are preserved"):
            if isinstance(result, np.ndarray):
                assert np.allclose(result, np.asarray(input_matrix))
            else:
                assert np.allclose(result.toarray(), np.asarray(input_matrix))

    @pytest.mark.parametrize("convert_to_start_type, desired_type", [
        (np.asarray, np.ndarray),
        (csr_array, csr_array),
        (dia_array, dia_array)
        ])
    def test_returns_same_object_if_already_desired_type(self,
                                                         convert_to_start_type,
                                                         desired_type):
        matrix = convert_to_start_type([[0, 1], [1, 1]])

        result = adjust_matrix_type("adjacency", matrix,
                                    {"adjacency": desired_type})

        assert result is matrix

    def test_raises_if_type_not_covered(self):
        matrix_types = {"adjacency": csr_array, "prejudice": list}
        with pytest.raises(ValueError, match="not covered"):
            adjust_matrix_type("adjacency",
                               csr_array([[0, 1], [1, 1]]),
                               matrix_types)

    def test_raises_matrix_name_without_type(self):
        with pytest.raises(ValueError, match="unspecified type"):
            adjust_matrix_type("prejudice",
                               np.asarray([[0, 1], [1, 1]]),
                               {"adjacency": csr_array})

    def test_raises_matrix_is_none(self):
        with pytest.raises(ValueError, match="loaded"):
            adjust_matrix_type("adjacency", None, {"adjacency": csr_array})


class TestGetMatrices:
    def test_correct_with_default_config(self, tmp_path, subtests):
        config_params = simulate_config_file(tmp_path, {})[0]
        no_nodes = config_params["number_of_nodes"]
        config_file_path = tmp_path / "file"
        matrices = get_matrices(directory=config_file_path)

        with subtests.test(msg="all matrices are built with the right type"):
            assert isinstance(matrices["adjacency"], csr_array)
            assert isinstance(matrices["prejudice"], np.ndarray)
            assert isinstance(matrices["susceptibility"], dia_array)

        with subtests.test(msg="all matrices have number_of_nodes users"):
            assert all(mtx.shape[0] == no_nodes for mtx in matrices.values())

    def test_correct_with_user_matrix(self, tmp_path, subtests):
        simulate_config_file(tmp_path,
                             {"adjacency": "[[0, 1, 0], [1, 0, 1], [0, 1, 0]]"},
                             no_of_nodes=10)

        matrices = get_matrices(directory=tmp_path / "file")

        with subtests.test(msg="loaded matrix is of right type"):
            assert isinstance(matrices["adjacency"], csr_array)

        with subtests.test(msg="loaded matrix is normalized"):
            assert np.allclose(matrices["adjacency"].toarray(),
                               [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])

        with subtests.test(msg="matrices use the loaded number of users"):
            assert matrices["prejudice"].shape[0] == 3
            assert matrices["susceptibility"].shape[0] == 3

        with subtests.test(msg="built matrices have the right type"):
            assert isinstance(matrices["prejudice"], np.ndarray)
            assert isinstance(matrices["susceptibility"], dia_array)

    def test_correct_with_all_user_matrices(self, tmp_path, subtests):
        simulate_config_file(
            tmp_path,
            {"adjacency": "[[0, 1, 0], [1, 0, 1], [1, 1, 1]]",
             "prejudice": "[[0.2], [0.5], [0.9]]",
             "susceptibility": "[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]"},
             no_of_nodes=10
             )
 
        matrices = get_matrices(directory=tmp_path / "file")
 
        with subtests.test(msg="loaded matrices have the right type"):
            assert isinstance(matrices["adjacency"], csr_array)
            assert isinstance(matrices["prejudice"], np.ndarray)
            assert isinstance(matrices["susceptibility"], dia_array)
 
        with subtests.test(msg="loaded adjacency matrix is normalized"):
            assert np.allclose(matrices["adjacency"].toarray(),
                               [[0, 1, 0], [0.5, 0, 0.5], [1./3, 1./3, 1./3]])
 
        with subtests.test(msg="all matrices use the loaded number of users"):
            assert all(mtx.shape[0] == 3 for mtx in matrices.values())
 
    def test_raises_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_matrices(directory=tmp_path / "file")


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
