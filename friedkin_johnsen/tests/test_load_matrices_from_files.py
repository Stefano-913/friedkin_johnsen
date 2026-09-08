import pytest
import sys
import numpy as np
import json

from ..load_matrices_from_files import (load_matrix_from_csv,
                                        load_matrix_from_json,
                                        load_matrix_from_file, file_loading)


# ----------

class TestFileLoading:
    def test_file_loading_decorator(self, tmp_path_factory):
        # GIVEN: a function wrapped by the decorator file_loading.
        # WHEN: the wrapped function is called with a valid file path.
        # THEN: the wrapped function is executed and returns the expected
        #       result.

        @file_loading
        def basic_func(unused_file_path):
            return np.array([[1, 2], [3, 4]])

        fn = tmp_path_factory.mktemp("data") / "matrix"
        fn.write_text("content irrelevant for this test")
        assert np.allclose(basic_func(fn.name, relative_path=str(fn.parent)),
                           [[1, 2], [3, 4]])

    def test_invalid_path(self, tmp_path_factory):
        # GIVEN: a function wrapped by the decorator file_loading.
        # WHEN: the wrapped function is called with an invalid file path.
        # THEN: a FileNotFoundError is raised before the function is executed.

        @file_loading
        def never_exec_func(unused_file_path):
            raise RuntimeError("This function should not be executed.")

        fn = tmp_path_factory.mktemp("data") / "matrix"
        fn.write_text("content irrelevant for this test")
        with pytest.raises(FileNotFoundError,
                           match=r".*wrong_file_name does not exist."):
            never_exec_func("wrong_file_name", relative_path=str(fn.parent))

        with pytest.raises(FileNotFoundError,
                           match=r"wrong_directory.* does not exist."):
            never_exec_func(fn.name, relative_path="wrong_directory")

    def test_empty_file(self, tmp_path_factory):
        # GIVEN: a function wrapped by the decorator file_loading.
        # WHEN: the wrapped function is called with a file path
        #       to an empty file.
        # THEN: a ValueError is raised before the function is executed.

        @file_loading
        def never_exec_func(unused_file_path):
            raise RuntimeError("This function should not be executed.")

        fn = tmp_path_factory.mktemp("data") / "matrix"
        fn.write_text("")
        with pytest.raises(ValueError,
                           match=f"{fn} is empty."):
            never_exec_func(fn.name, relative_path=str(fn.parent))

    def test_empty_array(self, tmp_path_factory):
        # GIVEN: a function wrapped by the decorator file_loading.
        # WHEN: the wrapped function is called with a file path
        #       to a file containing an empty array.
        # THEN: a ValueError is raised.

        @file_loading
        def empty_array(unused_file_path):
            return np.array([])

        fn = tmp_path_factory.mktemp("data") / "matrix"
        fn.write_text("content that translates to an empty array")
        with pytest.raises(ValueError,
                           match=f"{fn} contains no data."):
            empty_array(fn.name, relative_path=str(fn.parent))

    def test_value_error_gets_caught(self, tmp_path):
        fn = tmp_path / "matrix.csv"
        np.savetxt(fn, np.array([[1, 2], [3, 4]]), delimiter=",")

        @file_loading
        def raise_value_error(unused_file_path):
            raise ValueError("error raised by wrapped function")

        with pytest.raises(ValueError,
                           match=r".*Error in the loading of .*: error "
                           "raised by wrapped function"):
            raise_value_error(fn.name, relative_path=str(tmp_path))

    def test_key_error_gets_caught(self, tmp_path):
        fn = tmp_path / "matrix.csv"
        np.savetxt(fn, np.array([[1, 2], [3, 4]]), delimiter=",")

        @file_loading
        def raise_key_error(unused_file_path):
            raise KeyError("error raised by wrapped function")

        # quotes around the KeyError message are added by
        # the KeyError exception itself
        with pytest.raises(ValueError,
                           match=r".*Error in the loading of .*: 'error "
                           "raised by wrapped function'"):
            raise_key_error(fn.name, relative_path=str(tmp_path))

    def test_error_propagates_unwrapped(self, tmp_path):
        # GIVEN: a function wrapped by the decorator file_loading.
        # WHEN: the wrapped function raises an error.
        # THEN: the error is propagated as is, without being wrapped
        #       in a ValueError.

        fn = tmp_path / "matrix.csv"
        np.savetxt(fn, np.array([[1, 2], [3, 4]]), delimiter=",")

        @file_loading
        def raise_attribute_error(unused_file_path):
            raise AttributeError("error raised by wrapped function")

        with pytest.raises(AttributeError):
            raise_attribute_error(fn.name, relative_path=str(tmp_path))

    def test_nested_function_error_propagates_unwrapped(self, tmp_path):
        # GIVEN: a function f() wrapped by the decorator file_loading
        #        and a second function g() called within f().
        # WHEN: the g() function function raises an error within f().
        # THEN: the error is propagated as is, without being wrapped
        #       by the decorator.

        fn = tmp_path / "matrix.csv"
        np.savetxt(fn, np.array([[1, 2], [3, 4]]), delimiter=",")

        def g():
            raise AttributeError("error raised by nested function")

        @file_loading
        def f(unused_file_path):
            g()
            return np.array([[1, 2], [3, 4]])

        with pytest.raises(AttributeError, ):
            f(fn.name, relative_path=str(tmp_path))

    def test_nested_function_valueerror_propagates_unwrapped(self, tmp_path):
        # GIVEN: a function f() wrapped by the decorator file_loading
        #        and a second function g() called within f().
        # WHEN: the g() function function raises a ValueError within f().
        # THEN: the error is propagated as is, without being wrapped
        #       by the decorator.

        fn = tmp_path / "matrix.csv"
        np.savetxt(fn, np.array([[1, 2], [3, 4]]), delimiter=",")

        def g():
            raise ValueError("error raised by nested function")

        @file_loading
        def f(unused_file_path):
            g()
            return np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError,
                           match="error raised by nested function"):
            f(fn.name, relative_path=str(tmp_path))

# ---------


@pytest.fixture
def matrix_to_csv(tmp_path_factory, matrix):
    """ Builds a temporary .csv file containing the matrix to be loaded for
        testing purposes.
        If a string is provided, it gets written as is to the file for easier
        testing of string formats, otherwise the matrix is saved
        in .csv format.
    """

    fn = tmp_path_factory.mktemp("data") / "matrix.csv"

    if isinstance(matrix, str):
        fn.write_text(matrix)
    else:
        np.savetxt(fn, np.array(matrix), delimiter=",")
    return fn


class TestMatrixFromCSV:

    @pytest.mark.parametrize("matrix", [
        [1],
        [np.sqrt(2)],
        [[1]],
        [[np.e]]
        ])
    def test_single_element_csv_list_format(self, matrix_to_csv, matrix):

        assert np.allclose(load_matrix_from_csv(matrix_to_csv), matrix)

    @pytest.mark.parametrize("matrix", [
        [1, 2, 3],
        [1, np.e, np.sqrt(5)/2, -4, 5-np.sqrt(9)],
        [[1], [2], [3]],
        [[1], [np.e], [np.sqrt(5)/2], [-4], [5-np.sqrt(9)]]
        ])
    def test_vector_csv_list_format(self, matrix_to_csv, matrix):

        expected = np.array(matrix).reshape(-1, 1)
        assert np.allclose(load_matrix_from_csv(matrix_to_csv), expected)

    @pytest.mark.parametrize("matrix", [
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, np.e], [np.sqrt(5)/2, -4], [5, np.sqrt(9)]]
        ])
    def test_matrix_csv_list_format(self, matrix_to_csv, matrix):

        assert np.allclose(load_matrix_from_csv(matrix_to_csv), matrix)

    @pytest.mark.parametrize("matrix, expected_result", [
        ("3", 3),
        ("1.4142135623730951", np.sqrt(2)),
        ("-1", -1),
        ("2.718281828459045", np.e)
        ])
    def test_single_element_csv_string_format(self,
                                              matrix_to_csv,
                                              expected_result):

        assert np.allclose(load_matrix_from_csv(matrix_to_csv),
                           expected_result)

    @pytest.mark.parametrize("matrix, expected_result", [
        ("1, 2, 3", [1, 2, 3]),
        ("1, -2, 3, -4, 5, 6", [1, -2, 3, -4, 5, 6]),
        ("1, 2.718, 3.14", [1, 2.718, 3.14])
        ])
    def test_vector_csv_string_format(self, matrix_to_csv, expected_result):

        expected = np.array(expected_result).reshape(-1, 1)
        assert np.allclose(load_matrix_from_csv(matrix_to_csv), expected)

    @pytest.mark.parametrize("matrix, expected_result", [
        ("1, 2, 3", [[1], [2], [3]]),
        ("1, 2, 3\n4, 5, 6", [[1, 2, 3], [4, 5, 6]]),
        ("1, 2.718, 3.14\n4, 5, 6", [[1, 2.718, 3.14], [4, 5, 6]])
    ])
    def test_matrix_csv_string_format(self, matrix_to_csv, expected_result):

        assert np.allclose(load_matrix_from_csv(matrix_to_csv),
                           expected_result)

    def test_missing_csv_file(self, tmp_path):

        with pytest.raises(FileNotFoundError,
                           match=r".*not_exists.csv does not exist."):
            load_matrix_from_csv("not_exists.csv", relative_path=str(tmp_path))

    def test_wrong_csv_directory(self, tmp_path_factory):

        fn = tmp_path_factory.mktemp("data") / "exists.csv"
        with pytest.raises(FileNotFoundError,
                           match="wrong_dir/exists.csv does not exist."):
            load_matrix_from_csv(fn.name, relative_path="wrong_dir")

    def test_empty_csv_file(self, tmp_path):

        fn = tmp_path / "empty.csv"
        fn.write_text("")

        with pytest.raises(ValueError, match=r".*is empty"):
            load_matrix_from_csv(fn.name, relative_path=str(tmp_path))

    def test_empty_csv_array(self, tmp_path):

        fn = tmp_path / "empty.csv"
        fn.write_text("\n")

        with pytest.warns(UserWarning, match=r".*input contained no data"):
            with pytest.raises(ValueError, match=r".*contains no data"):
                load_matrix_from_csv(fn.name, relative_path=str(tmp_path))

    def test_invalid_csv_values(self, tmp_path):

        fn = tmp_path / "invalid.csv"
        fn.write_text("1,2,3\n4,invalid,6\n")
        with pytest.raises(ValueError,
                           match=r"Error in the loading of .*"):
            load_matrix_from_csv(fn.name, relative_path=str(tmp_path))

    @pytest.mark.parametrize("matrix", [
        [[1, 2], [3, 4], [5, np.inf]],
        [[1, 2], [3, 4], [5, np.nan]]
        ])
    def test_infinite_csv_values(self, matrix_to_csv):

        fn = matrix_to_csv
        with pytest.raises(ValueError,
                           match=r".*contains NaN or infinite values"):
            load_matrix_from_csv(fn)

    def test_ragged_csv_matrix(self, tmp_path):

        fn = tmp_path / "ragged.csv"
        fn.write_text("1,2,3\n4,5\n")
        with pytest.raises(ValueError,
                           match=r"Error in the loading of .*"):
            load_matrix_from_csv(fn.name, relative_path=str(tmp_path))

# -------


@pytest.fixture
def item_to_json(tmp_path_factory):
    """ Builds a temporary .json file containing the content to be loaded for
        testing purposes.
        Most generic function that loads input as is, especially useful for
        testing of invalid formats
    """

    def _item_to_json(item):

        fn = tmp_path_factory.mktemp("data") / "item.json"
        with open(fn, "w") as f:
            json.dump(item, f)
        return fn
    return _item_to_json


@pytest.fixture
def matrix_to_list_json(item_to_json, matrix):
    """Temporarily builds a valid .json file in the "list" format
        (e.g. [[1, 0], [0, 1]])
    """

    matrix = np.array(matrix)
    return item_to_json(matrix.tolist())


@pytest.fixture
def matrix_to_dict_json(item_to_json, matrix):
    """Temporarily builds a valid .json file in the "dictionary" format
        (e.g. {..., "matrix": [[1, 0], [0, 1]]})
    """

    matrix = np.array(matrix)
    return item_to_json({
                "name": "susceptibility_matrix",
                "matrix": matrix.tolist()
            })


class TestMatrixFromJSON:
    """ The testing of expected behaviour generally follows the following
        GIVEN-WHEN-THEN pattern:

        GIVEN:  a .json file containing an array_like object in a valid format:
                - list (e.g. [[1, 0], [0, 1]])
                - dictionary with "matrix" key, possibly with string
                version of the matrix.
                (e.g.   {"matrix": [[1, 0], [0, 1]]},
                        {"matrix": "[[1, 0], [0, 1]]"})
        WHEN:   the load_matrix_from_json() function is called with
                the file path.
        THEN:   the function returns a numpy array with the expected shape
                and values.

        List and dictionary formats are sometimes tested together,
        as to avoid excessive duplication of code since the difference
        is minimal.
        """

    @pytest.mark.parametrize("matrix", [
        1,
        np.sqrt(2),
        [[1]],
        [[np.e]]
        ])
    def test_single_element_from_json(self,
                                      matrix_to_dict_json,
                                      matrix_to_list_json,
                                      matrix):

        expected = np.array(matrix).reshape(-1, 1)
        assert np.allclose(load_matrix_from_json(matrix_to_dict_json),
                           expected)
        assert np.allclose(load_matrix_from_json(matrix_to_list_json),
                           expected)

    @pytest.mark.parametrize("matrix", [
        [1, 2, 3],
        [1, np.e, np.sqrt(5)/2, -4, 5-np.sqrt(9)],
        [[1], [2], [3]],
        [[1], [np.e], [np.sqrt(5)/2], [-4], [5-np.sqrt(9)]]
        ])
    def test_vector_from_json(self,
                              matrix_to_dict_json,
                              matrix_to_list_json,
                              matrix):

        expected = np.array(matrix).reshape(-1, 1)
        assert np.allclose(load_matrix_from_json(matrix_to_dict_json),
                           expected)
        assert np.allclose(load_matrix_from_json(matrix_to_list_json),
                           expected)

    @pytest.mark.parametrize("matrix", [
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, np.e], [np.sqrt(5)/2, -4], [5, np.sqrt(9)]]
        ])
    def test_matrix_from_json(self,
                              matrix_to_dict_json,
                              matrix_to_list_json,
                              matrix):

        assert np.allclose(load_matrix_from_json(matrix_to_dict_json), matrix)
        assert np.allclose(load_matrix_from_json(matrix_to_list_json), matrix)

    @pytest.mark.parametrize("matrix, expected_result", [
        ("1", [1]),
        ("2.718281828459045", np.e),
        ("[1]", [[1]]),
        ("[2.718281828459045]", [[np.e]])
        ])
    def test_string_single_element_from_json(self,
                                             matrix_to_dict_json,
                                             expected_result):

        assert np.allclose(load_matrix_from_json(matrix_to_dict_json),
                           expected_result)

    @pytest.mark.parametrize("matrix, expected_result", [
            ("[1, 2, 3]", [1, 2, 3]),
            ("[1, 2.718281828459045, 1.118033988749895, -4, 2.0]",
             [1, 2.718281828459045, 1.118033988749895, -4, 2.0]),
            ("[[1], [2], [3]]", [[1], [2], [3]]),
            ("[[1], [2.718281828459045], [1.118033988749895], [-4], [2.0]]",
             [[1], [2.718281828459045], [1.118033988749895], [-4], [2.0]])
            ])
    def test_string_vector_from_json(self,
                                     matrix_to_dict_json,
                                     expected_result):

        expected = np.array(expected_result).reshape(-1, 1)
        assert np.allclose(load_matrix_from_json(matrix_to_dict_json),
                           expected)

    @pytest.mark.parametrize("matrix, expected_result", [
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("[[1, 2], [3, 4], [5, 6]]", [[1, 2], [3, 4], [5, 6]]),
        ("[[1, 2.718], [3.14, -4], [5, 3]]", [[1, 2.718], [3.14, -4], [5, 3]])
        ])
    def test_string_matrix_from_json(self, matrix_to_dict_json,
                                     expected_result):

        assert np.allclose(load_matrix_from_json(matrix_to_dict_json),
                           np.array(expected_result))

    def test_missing_json_file(self, tmp_path):

        with pytest.raises(FileNotFoundError,
                           match=r".*not_exists.json does not exist."):
            load_matrix_from_json("not_exists.json",
                                  relative_path=str(tmp_path))

    def test_wrong_json_directory(self, tmp_path_factory):

        fn = tmp_path_factory.mktemp("data") / "exists.json"
        with pytest.raises(FileNotFoundError,
                           match="wrong_dir/exists.json does not exist."):
            load_matrix_from_json(fn.name, relative_path="wrong_dir")

    def test_empty_json_file(self, tmp_path):

        fn = tmp_path / "empty.json"
        fn.write_text("")
        with pytest.raises(ValueError, match=r".*is empty"):
            load_matrix_from_json(fn.name, relative_path=str(tmp_path))

    def test_empty_array(self, tmp_path):

        fn = tmp_path / "empty.json"
        fn.write_text("[]")

        with pytest.raises(ValueError, match=r".*contains no data"):
            load_matrix_from_json(fn.name, relative_path=str(tmp_path))

    def test_invalid_json_values(self, item_to_json):

        fn = item_to_json({
                "matrix_name": "susceptibility",
                "no_users": 3,
                "matrix": [[1, 2], [3, 4], [5, "invalid"]]
            })
        with pytest.raises(TypeError,
                           match=r".*cannot be converted to floating-point.*"):
            load_matrix_from_json(fn)

    @pytest.mark.parametrize("matrix", [
        [[1, 2], [3, 4], [5, np.inf]],
        [[1, 2], [3, 4], [5, np.nan]],
        [[1, 2], [3, 4], [5, None]]
        ])
    def test_infinite_values_json(self, item_to_json, matrix):

        fn = item_to_json(matrix)
        with pytest.raises(ValueError,
                           match=r".*contains NaN or infinite values"):
            load_matrix_from_json(fn)

    def test_ragged_json_matrix(self, item_to_json):

        fn = item_to_json({
                "matrix_name": "susceptibility",
                "no_users": 3,
                "matrix": [[1, 2], [3], [4, 5]]
            })
        with pytest.raises(ValueError,
                           match=r".*cannot be converted to a matrix"):
            load_matrix_from_json(fn)

    def test_wrong_key(self, item_to_json):

        fn = item_to_json({
                "matrix_name": "susceptibility",
                "no_users": 3,
                "wrong_key": [[1, 2], [3, 4], [5, 6]]
            })
        with pytest.raises(ValueError,
                           match=r".*Expected a 'matrix' key in .*"):
            load_matrix_from_json(fn)

    @pytest.mark.parametrize("matrix", [
            "1",
            "[1, 2.71, 1.11, -4, 2.0]",
            "[[1], [2], [3]]",
            ])
    def test_error_string_from_json(self,  matrix_to_list_json):

        with pytest.raises(ValueError,
                           match=r"The data in .* must be either in list or "
                           r"dictionary format, currently of type.*'str'"):
            load_matrix_from_json(matrix_to_list_json)


# ------

class TestLoadInputMatrices:

    def test_load_matrix_from_file_csv(self, tmp_path_factory):

        fn = tmp_path_factory.mktemp("data") / "matrix.csv"
        np.savetxt(fn, np.array([[1, 2], [3, 4]]), delimiter=",")

        loaded_matrix = load_matrix_from_file("matrix.csv",
                                              relative_path=str(fn.parent))
        assert np.allclose(loaded_matrix, [[1, 2], [3, 4]])

    def test_load_matrix_from_file_json(self, tmp_path_factory):

        fn = tmp_path_factory.mktemp("data") / "matrix.json"
        with open(fn, "w") as f:
            json.dump({"matrix": [[1, 2], [3, 4]]}, f)

        loaded_matrix = load_matrix_from_file("matrix.json",
                                              relative_path=fn.parent)
        assert np.allclose(loaded_matrix, [[1, 2], [3, 4]])

    @pytest.mark.parametrize("file_extension", [
        ".txt",
        ".xlsx",
        ".xml"
        ])
    def test_unsupported_extensions(self, tmp_path_factory, file_extension):

        fn = tmp_path_factory.mktemp("data") / f"matrix{file_extension}"
        with open(fn, "w") as f:
            f.write("content of the file with a wrong extension, "
                    "will not be read")

        with pytest.raises(ValueError,
                           match=f"File '{fn.name}' has an unsupported file "
                           f"extension: '{file_extension}'"):
            load_matrix_from_file(fn.name, relative_path=fn.parent)

    @pytest.mark.parametrize("file_extension", [
        "",
        ".",
        ". "
        ])
    def test_load_matrix_from_file_null_extension(
            self, tmp_path_factory, file_extension):

        fn = tmp_path_factory.mktemp("data") / f"matrix{file_extension}"
        with open(fn, "w") as f:
            f.write("content of the file with a null extension is unused")

        with pytest.raises(ValueError, match=r".*is without extension."):
            load_matrix_from_file(fn.name, relative_path=fn.parent)


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
