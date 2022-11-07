from abc import ABC, abstractmethod
from typing import Union

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class Visualizer(ABC):
    """
    Abstractclass for representing a Visualizer object.
    """

    @abstractmethod
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        """
        Abtractmethod for instanciation of Visualizer object.

        Args:
            model (PreTrainedModel): A model given by HuggingFace ecosystem.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A tokenizer given by HuggingFace ecosystem.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, text: str) -> None:
        """
        Given a text input generates necessary elements for visualization.

        Args:
            text (str): Text input

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize(self) -> None:
        """
        Plot the Visualizer.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the class name of the Visualizer.

        Returns:
            str: Class name.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return the class name, the model and the tokenizer of the Visualizer.

        Returns:
            str: Class name, model and tokenizer.
        """
        s = f"{self.__class__.__name__}("
        s += f"model={self.model.__class__.__name__}, "
        s += f"tokenizer={self.tokenizer.__class__.__name__})"
        return s
