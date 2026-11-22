import json
import numpy as np
import pytest
from ..output_file import write_output_file


@pytest.mark.parametrize("sims, expected_result", [
    ([{"0": np.array([1, 2, 3, 4])}],
     {"0": {"0": [1, 2, 3, 4]}}),
    ([{"0": np.array([[1], [2], [3], [4]])}],
     {"0": {"0": [1, 2, 3, 4]}}),

    # ---
    ([{0: np.array([[1], [2], [3], [4]]),
       1: np.array([[5], [6], [7], [8]]),
       2: np.array([[9], [10], [11], [12]])},

      {0: np.array([[13], [14], [15], [16]]),
       1: np.array([[17], [18], [19], [20]]),
       2: np.array([[21], [22], [23], [24]])}],

     {"0": {"0": [1, 2, 3, 4],
            "1": [5, 6, 7, 8],
            "2": [9, 10, 11, 12]},
      "1": {"0": [13, 14, 15, 16],
            "1": [17, 18, 19, 20],
            "2": [21, 22, 23, 24]}})
])
def test_correct_output(tmp_path, sims, expected_result):
    result = write_output_file(tmp_path / "output.json", sims)
    assert json.loads(result.read_text()) == expected_result


@pytest.mark.parametrize("sims, expected_result", [
    ([{"0": np.array([[1, 2], [3, 4]]),
       "1": np.array([[5, 6], [7, 8]])}],
     {"0": {"0": [[1, 2], [3, 4]],
            "1": [[5, 6], [7, 8]]}}),

    # ----
    ([{0: np.array([[1, 2], [3, 4]]),
       1: np.array([[5, 6], [7, 8]]),
       2: np.array([[9, 10], [11, 12]])},

      {0: np.array([[13, 14], [15, 16]]),
       1: np.array([[17, 18], [19, 20]]),
       2: np.array([[21, 22], [23, 24]])}],

     {"0": {"0": [[1, 2], [3, 4]],
            "1": [[5, 6], [7, 8]],
            "2": [[9, 10], [11, 12]]},
      "1": {"0": [[13, 14], [15, 16]],
            "1": [[17, 18], [19, 20]],
            "2": [[21, 22], [23, 24]]}})
])
def test_correct_if_multi_topic(tmp_path, sims, expected_result):
    result = write_output_file(tmp_path / "output.json", sims)
    assert json.loads(result.read_text()) == expected_result


def test_empty_simulations(tmp_path):
    result = write_output_file(tmp_path / "output.json", [])
    assert json.loads(result.read_text()) == {}


def test_creates_parent_directory(tmp_path):
    output_path = tmp_path / "new_dir" / "output.json"
    write_output_file(output_path, [{"0": np.array([1, 2])}])

    assert output_path.parent.is_dir()


def test_existing_file_is_not_overwritten(tmp_path):
    output_path = tmp_path / "output.json"
    first = write_output_file(output_path,
                              [{"0": np.array([1, 2])}])
    second = write_output_file(output_path,
                               [{"0": np.array([3, 4])}])

    assert first == output_path
    assert second == tmp_path / "output_1.json"
    assert json.loads(first.read_text()) == {"0":{"0": [1, 2]}}
    assert json.loads(second.read_text()) == {"0":{"0": [3, 4]}}


def test_takes_first_available_name(tmp_path):
    (tmp_path / "output.json").write_text("file_content")
    (tmp_path / "output_1.json").write_text("file_content")

    result = write_output_file(tmp_path / "output.json",
                               [{"0": np.array([1, 2])}])
    assert result == tmp_path / "output_2.json"


def test_file_index_up_to_999(tmp_path):
    (tmp_path / "output.json").write_text("file_content")
    for i in range(1, 999):
        (tmp_path / f"output_{i}.json").write_text("file_content")

    result = write_output_file(tmp_path / "output.json",
                               [{"0": np.array([1, 2])}])
    assert result == tmp_path / "output_999.json"


def test_raises_after_999_index(tmp_path):
    (tmp_path / "output.json").write_text("file_content")
    for i in range(1, 1000):
        (tmp_path / f"output_{i}.json").write_text("file_content")

    with pytest.raises(FileExistsError):
        write_output_file(tmp_path / "output.json", [{"0": np.array([1, 2])}])
