import pytest
import sys
import numpy as np
import re

from ..error_messages import truncate_item_for_error, item_summary_for_error

# --------


class TestTruncateItemForError:

    def test_no_input(self):
        assert truncate_item_for_error() == ""
        assert truncate_item_for_error("") == ""

    @pytest.mark.parametrize("input_item, expected_return", [
         (0, "0"),
         ("a", "a"),
         (1+2j, "(1+2j)"),
         (1.49, "1.49"),
         (True, "True"),
         (np.array([0 for i in range(5)]), "[0 0 0 0 0]")
        ])
    def test_return_short_input_as_is(self, input_item, expected_return):
        assert truncate_item_for_error(input_item) == expected_return

    def test_short_input_custom_max_length(self):
        assert truncate_item_for_error(
             '0'*10, max_length=20) == ('0'*10)
        assert truncate_item_for_error(
             np.array([0 for i in range(5)]), max_length=20) == "[0 0 0 0 0]"

    @pytest.mark.parametrize("input_item, expected_return", [
        ('0'*70, '0'*70),
        ('0'*100, f"{'0'*33}...{'0'*33}"),
        (pow(10, 100), f"1{'0'*32}...{'0'*33}"),

        # space between charaters in an array halves the entries
        # necessary to reach max_length -> 34 length + parenthesis is 70
        (np.array([1]*34), f"[{'1 '*33}1]"),
        (np.array([0]*100), f"[{'0 '*16}...{' 0'*16}]")
    ])
    def test_truncate_long_input(self, input_item, expected_return):
        result = truncate_item_for_error(input_item)
        assert result == expected_return
        assert len(result) <= 70

    @pytest.mark.parametrize("input_item, max_length, expected_return", [
        ('0'*100, 10, f"{'0'*3}...{'0'*3}"),
        ('0'*100, 9, f"{'0'*3}...{'0'*3}"),
        ('0'*100, 8, f"{'0'*2}...{'0'*2}"),
        ('0'*100, 7, f"{'0'*2}...{'0'*2}"),

        # max_length clamped
        ('0'*100, 3, "0...0"),
        ('0'*100, 0, "0...0"),
        ('0'*100, -5, "0...0"),
        ('0'*100, True, "0...0"),
        ('0'*100, False, "0...0"),

        (np.pi, 10, "3.1...793"),  # np.pi has 17 digits
        (1+1j, 8, "(1+1j)"),
        (456+789j, 9, "(45...9j)"),
        (np.array([0]*100), 10, "[0 ... 0]")
        ])
    def test_truncate_input_with_max_length(self, input_item,
                                            max_length, expected_return):
        assert truncate_item_for_error(
             input_item, max_length=max_length) == expected_return

    @pytest.mark.parametrize("max_length, error_message_section", [
         ("5", "5 is <class 'str'>"),
         (0.5, "0.5 is <class 'float'>"),
         (1+2j, "(1+2j) is <class 'complex'>")
         ])
    def test_wrong_max_length_type(self, max_length, error_message_section):
        error_message = re.escape(f"max_length = {error_message_section} "
                                  "instead of an integer.")
        with pytest.raises(TypeError, match=error_message):
            truncate_item_for_error(1, max_length=max_length)

    @pytest.mark.parametrize("max_length", [
         (np.nan),
         (np.inf),
         (np.inf + 1),
         (np.nan + np.inf),
         (np.inf * np.nan),
         (np.nan + 9j)
         ])
    def test_max_length_not_finite(self, max_length):
        error_message = ("max_length must be a finite value, "
                         "not infinite or nan.")
        with pytest.raises(ValueError, match=error_message):
            truncate_item_for_error(1, max_length=max_length)

# --------


class TestItemSummaryForError:

    @pytest.mark.parametrize(
            "input_item, item_name, expected_return", [
                (1, "", "(1, <class 'int'>)"),
                (1, "item", "'item' (1, <class 'int'>)"),
                (1, "variable", "'variable' (1, <class 'int'>)"),

                (np.array([0 for i in range(5)]), "",
                    "([0 0 0 0 0], <class 'numpy.ndarray'>)"),
                (np.array([0 for i in range(5)]), "array_name",
                    "'array_name' ([0 0 0 0 0], <class 'numpy.ndarray'>)"),

                (np.array([1 for i in range(100)]), "very_long_array",
                 (f"'very_long_array' ([{'1 '*16}...{' 1'*16}], "
                  "<class 'numpy.ndarray'>)"))
                ])
    def test_item_summary_for_error(self,
                                    input_item, item_name, expected_return):
        assert item_summary_for_error(input_item, item_name) == expected_return


if __name__ == "__main__":
    sys.exit(pytest.main(
         ["-v", "--cov=your_package", "--cov-report=term-missing"]))
