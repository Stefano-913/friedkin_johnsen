import pytest
import sys
import json

from ..handle_config_file import (adjacency_default_params,
                                  prejudice_default_params,
                                  susceptibility_default_params,
                                  default_config_file,
                                  build_config_file,
                                  nested_dict_merge,
                                  fill_out_config_file)


class TestBuildConfigFile:
    def test_correct_default(self, tmp_path, subtests):
        build_config_file(file_name="matrix_config.json",
                          relative_path=tmp_path)

        file_path = tmp_path / "matrix_config.json"
        with subtests.test(msg="creates file"):
            assert (file_path).exists()

        with subtests.test(msg="default file is correct"):
            assert json.loads(file_path.read_text()) == default_config_file()

    @pytest.mark.parametrize("file_name, subdirectory", [
        ("custom.json", "subdirectory"),
        ("matrix_config.json", "first_dir/second_dir/nested_dir")
    ])
    def test_change_directory_or_file_name(self,
                                           file_name,
                                           subdirectory,
                                           tmp_path,
                                           subtests):
        relative_path = tmp_path / subdirectory

        build_config_file(file_name=file_name, relative_path=relative_path)
        file_path = relative_path / file_name

        with subtests.test(msg="creates file"):
            assert (file_path).exists()

        with subtests.test(msg="default file is correct"):
            assert json.loads(file_path.read_text()) == default_config_file()

    def test_no_overwrite_by_default(self, tmp_path):
        path = tmp_path / "matrix_config.json"
        path.write_text('{"entry": 1}')

        build_config_file(relative_path=tmp_path)
        assert json.loads(path.read_text()) == {"entry": 1}

    def test_overwrite_function(self, tmp_path):
        path = tmp_path / "matrix_config.json"
        path.write_text('{"entry": true}')

        build_config_file(relative_path=tmp_path, overwrite=True)
        assert json.loads(path.read_text()) == default_config_file()


class TestNestedDictMerge:
    @pytest.mark.parametrize("to_overwrite, overwriter, expected_result", [
        ({}, {}, {}),
        ({"a": 1}, {}, {"a": 1}),
        ({"a": 1}, {"a": 1}, {"a": 1}),
        ({"a": 1}, {"a": 2}, {"a": 2}),

        ({"a": 1, "b": 2}, {"a": 2}, {"a": 2, "b": 2}),
        ({"a": 1, "b": 2}, {"a": 5, "b": 6}, {"a": 5, "b": 6})
    ])
    def test_plain_dict_merge(self, to_overwrite, overwriter, expected_result):
        result = nested_dict_merge(to_overwrite, overwriter)
        assert result == expected_result

    @pytest.mark.parametrize("to_overwrite, overwriter, expected_result", [
        ({"a": {"b": 1}}, {}, {"a": {"b": 1}}),

        ({"a": {"b": 1}}, {"a": {"b": 2}}, {"a": {"b": 2}}),

        ({"a": {"b": 1}, "c": 4}, {"a": {"b": 2}}, {"a": {"b": 2}, "c": 4}),

        ({"a": {"b": {"c": 1}}, "d": 4, "e": 4},
         {"a": {"b": {"c": 5}}, "d": -1},
         {"a": {"b": {"c": 5}}, "d": -1, "e": 4})
    ])
    def test_nested_dict_merge(self,
                               to_overwrite,
                               overwriter,
                               expected_result):
        result = nested_dict_merge(to_overwrite, overwriter)
        assert result == expected_result

    @pytest.mark.parametrize("overwriter", [
        {"a": 7},
        {"a": {"b": 1}}
    ])
    def test_replacement_of_none(self, overwriter):
        assert nested_dict_merge({"a": None}, overwriter) == overwriter

    @pytest.mark.parametrize("to_overwrite, overwriter", [
        ({}, {"a": 1}),
        ({"a": 1}, {"a": 2, "b": 2}),

        ({"a": {"b": {"c": 5}}, "d": -1},
         {"a": {"b": {"c": 1}}, "d": 4, "e": 4})
    ])
    def test_raises_unrecognized_key(self,
                                     to_overwrite,
                                     overwriter):
        with pytest.raises(ValueError, match=r"Unrecognized key"):
            nested_dict_merge(to_overwrite, overwriter)

    @pytest.mark.parametrize("to_overwrite, overwriter", [
        ({"a": dict()}, {"a": 3}),
        ({"a": {"b": 1}}, {"a": []}),
        ({"a": {"b": {"c": 5}}}, {"a": {"b": "not_dict"}})
    ])
    def test_raises_if_attempt_to_change_dict_type(self,
                                                   to_overwrite,
                                                   overwriter):
        with pytest.raises(ValueError, match=r"must be a dictionary"):
            nested_dict_merge(to_overwrite, overwriter)


class TestFillOutConfigFile:

    def test_empty_user_config_same_as_default(self, tmp_path):
        file_path = tmp_path / "user_file.json"
        file_path.write_text(json.dumps({}))

        assert fill_out_config_file(file_path) == default_config_file()

    def test_default_unchanged(self, tmp_path):
        file_path = tmp_path / "matrix_config.json"
        build_config_file(relative_path=tmp_path)

        assert fill_out_config_file(file_path) == default_config_file()

    def test_plain_config_keys_are_filled(self, tmp_path):
        file_path = tmp_path / "user_file.json"
        file_path.write_text(json.dumps({"number_of_nodes": 10}))

        result = fill_out_config_file(file_path)
        assert result["number_of_nodes"] == 10
        assert result["max_cycles"] == default_config_file()["max_cycles"]
        assert "matrices" in result

    def test_nested_config_keys_are_filled(self, tmp_path, subtests):
        file_path = tmp_path / "user_file.json"
        file_path.write_text(
            json.dumps({"matrices": {"adjacency": {"params": {"tries": 5}}}})
            )

        result = fill_out_config_file(file_path)
        matrices = result["matrices"]

        adjacency_params = matrices["adjacency"]["params"]

        with subtests.test(msg="specified values remain as defined"):
            assert adjacency_params["tries"] == 5

        default_adjacency_params = adjacency_default_params()["params"]
        default_mean_degree = default_adjacency_params["mean_node_degree"]

        with subtests.test(msg="undefined values are as default"):
            assert adjacency_params["mean_node_degree"] == default_mean_degree

        with subtests.test(msg="undefined matrices have default values"):
            assert matrices["prejudice"] == prejudice_default_params()
            assert matrices["susceptibility"] == susceptibility_default_params()

    def test_writes_merged_config_back_to_file(self, tmp_path):
        file_path = tmp_path / "user_file.json"
        file_path.write_text(json.dumps({"number_of_nodes": 10}))

        result = fill_out_config_file(file_path)
        assert json.loads(file_path.read_text()) == result
        assert "max_cycles" in json.loads(file_path.read_text())

    def test_filling_is_idempotent(self, tmp_path):
        file_path = tmp_path / "user.json"
        file_path.write_text(json.dumps({"number_of_nodes": 10}))

        fill_out_config_file(file_path)
        first_func_call = file_path.read_text()

        fill_out_config_file(file_path)
        second_func_call = file_path.read_text()

        assert second_func_call == first_func_call

    def test_raises_if_unknown_key(self, tmp_path):
        file_path = tmp_path / "typo.json"
        file_path.write_text(json.dumps({"max_cyles": 5}))

        with pytest.raises(ValueError, match="max_cyles"):
            fill_out_config_file(file_path)

    def test_raises_if_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fill_out_config_file(tmp_path / "nope.json")


class TestDefaultConfigFile:
    # Default values are arbitrary and can be changed, the testing is
    # hence limited to their sensibility.

    def test_general_values_constraints(self, subtests):
        config = default_config_file()

        with subtests.test(msg="positive integer number of nodes"):
            assert isinstance(config["number_of_nodes"], int)
            assert config["number_of_nodes"] > 0

        with subtests.test(msg="positive integer number of max cycles"):
            assert isinstance(config["max_cycles"], int)
            assert config["max_cycles"] > 0


class TestAdjacencyDefaultParams:
    # Default values are arbitrary and can be changed, the testing is
    # hence limited to their sensibility.

    def test_default_values_are_sensible(self, subtests):
        config = default_config_file()
        adjacency = adjacency_default_params()["params"]

        with subtests.test(msg="mean node degree is positive integer"):
            assert isinstance(adjacency["mean_node_degree"], int)
            assert 0 < adjacency["mean_node_degree"]

        with subtests.test(msg="mean degree is lower than number of nodes"):
            assert adjacency["mean_node_degree"] < config["number_of_nodes"]

        with subtests.test(msg="probability of edge rewire is float in [0,1]"):
            assert isinstance(adjacency["prob_rewire_edge"], float)
            assert 0 <= adjacency["prob_rewire_edge"] <= 1

        with subtests.test(msg="minimum edge weight is positive float"):
            assert isinstance(adjacency["min_edge_weight"], float)
            assert adjacency["min_edge_weight"] > 0

        with subtests.test(msg="number of tries is integer higher than 0"):
            assert isinstance(adjacency["tries"], int)
            assert adjacency["tries"] >= 1

        with subtests.test(msg="matrix is randomized each run"):
            assert adjacency["seed"] is None


class TestPrejudiceDefaultParams:
    # Default values are arbitrary and can be changed, the testing is
    # hence limited to their sensibility.
    def test_default_values_are_sensible(self, subtests):
        prejudice = prejudice_default_params()["params"]

        with subtests.test(msg="number of topics is integer higher than 0"):
            assert isinstance(prejudice["no_of_topics"], int)
            assert prejudice["no_of_topics"] >= 1

        with subtests.test(msg="mean prejudice is float in [0,1]"):
            assert isinstance(prejudice["mean_prejudice"], float)
            assert 0 <= prejudice["mean_prejudice"] <= 1

        with subtests.test(msg="prejudice std is non negative float"):
            assert isinstance(prejudice["std_prejudice"], float)
            assert prejudice["std_prejudice"] >= 0

        with subtests.test(msg="matrix is randomized each run"):
            assert prejudice["seed"] is None


class TestSusceptibilityDefaultParams:
    # Default values are arbitrary and can be changed, the testing is
    # hence limited to their sensibility.

    def test_default_values_are_sensible(self, subtests):
        susceptibility = susceptibility_default_params()["params"]

        with subtests.test(msg="mean susceptibility is float in [0,1]"):
            assert isinstance(susceptibility["mean_susceptibility"], float)
            assert 0 <= susceptibility["mean_susceptibility"] <= 1

        with subtests.test(msg="susceptibility std is non negative float"):
            assert isinstance(susceptibility["std_susceptibility"], float)
            assert susceptibility["std_susceptibility"] >= 0

        with subtests.test(msg="matrix is randomized each run"):
            assert susceptibility["seed"] is None


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
