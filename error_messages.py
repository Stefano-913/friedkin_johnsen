import typing
import numpy as np


def truncate_item_for_error(item: typing.Any = "",
                            max_length: int = 70) -> str:
    """ Truncate the string representation of an item for use in error
        messages.

        If the string representation of the item contains more characters than
        the "max_length" argument dictates, its central part is omitted to
        avoid unwieldy output on the terminal.
        "max length" can range between 5 and the item's actual character size.

        Parameters
        ----------
        item : any
            Object that will have its string representation limited
            in characters.

        max_length : int, optional
            Maximum lenght of the string representation.
            Its value is clamped between 5 and the item's actual
            character size, default value is chosen to be 70.

        Returns
        -------
        str
            The final string representation of "item" contained within
            "max_length" characters.

        Raises
        ------
        TypeError
            If the "max_length" argument is not an integer.

        Note
        ----
        In case of lengthy structures, it's possible that important cues
        might be lost in the truncation process
        (e.g. complex numbers might not show the "+/-" symbol separating
        the real and imaginary parts).
        """

    str_item = str(item)

    if not isinstance(max_length, (int)):

        if isinstance(max_length, (float | complex)) and \
                not np.isfinite(max_length):
            raise ValueError("max_length must be a finite value, "
                             "not infinite or nan.")

        raise TypeError(f"max_length = {max_length} is {type(max_length)} "
                        "instead of an integer.")

    max_length = min(max(5, max_length), len(str_item))

    if len(str_item) > max_length:
        # subtract three characters to make up for the three dots in the middle
        half = (max_length - 3) // 2
        str_item = str_item[:half] + "..." + str_item[-half:]

    return str_item


def item_summary_for_error(item: typing.Any, item_name: str = "") -> str:
    """ Create a summary of an item for use in error messages.

        The summary is a string representation of the item, truncated to a
        maximum length of 70 characters, and optionally includes the name of
        the item.

        Parameters
        ----------
        item : any
            Object that will have its string representation summarized.

        item_name : str, optional
            Name of the item to be included in the summary, default is
            an empty string.

        Returns
        -------
        str
            A summary string representation of "item" with optional name
            included.

        See Also
        --------
        truncate_item_for_error : Function that truncates the string
        representation.
    """

    truncated_item = truncate_item_for_error(item)

    if item_name != "":
        return f"'{str(item_name)}' ({truncated_item}, {type(item)})"
    else:
        return f"({truncated_item}, {type(item)})"
