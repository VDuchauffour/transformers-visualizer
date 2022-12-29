class DimensionIndicesError(IndexError):
    """
    Error raises when dimension indices is out of range.
    """

    def __init__(self, index_name: str, value: int, max_value: int):
        """
        Instantiate a DimensionIndicesError.

        Args:
            index_name (str): name of the index.
            value (int): value of the indice.
            max_value (int): maximum value of the indice.
        """
        self.index_name = index_name
        self.value = value
        self.max_value = max_value

    def __str__(self):
        """
        Print the error.
        """
        return (
            f"`{self.index_name}` expect value between {-self.max_value-1} and"
            f" {self.max_value}, got {self.value} instead."
        )


class TextInputError(NotImplementedError):
    """
    Error raises when multiple text input are given.
    """

    def __str__(self):
        """
        Print the error.
        """
        return "Multiple text input is not supported."


class OutputNotComputedEror(RuntimeError):
    """
    Error raised if the visualizer hasn't a `ModelOutput` in theirs attributes.
    """

    def __str__(self):
        """
        Print the error.
        """
        return (
            "Visualizer must be called with `__call__` or `compute` method and a text"
            " input before applying `plot` method."
        )
