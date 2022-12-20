from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Union

import torch
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
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Abtractmethod for instantiate a Visualizer object.

        Args:
            model (PreTrainedModel): A `model` given by HuggingFace ecosystem.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A `tokenizer` given by HuggingFace ecosystem.
            device (Optional[Union[torch.device, str]]): A `device` where computation takes place. If no `torch.device` are provided, the `model`'s device is used.
        """
        self.model = deepcopy(model).eval()
        self.tokenizer = tokenizer

        if device is None:
            self._device = self.model.device
        else:
            self._device = device

    @property
    def device(self):
        """
        Return the `torch.device` used for computation.
        """
        return self._device

    @device.setter
    def device(self, value: Union[torch.device, str]) -> None:
        """
        Set a `torch.device` for computation.

        Args:
            value (Union[torch.device, str]): A `torch.device` or a `str`.

        Raises:
            ValueError: raised if `value` isn't a `torch.device` or acceptable `str`.
        """
        if isinstance(value, str):
            self._device = torch.device(value)
        elif isinstance(value, torch.device):
            self._device = value
        else:
            raise ValueError(
                "`value` expect a `torch.device` or a `str`, got"
                f" {type(value)} instead."
            )

        self.model = self.model.to(self._device)

    @abstractmethod
    def set_device(self, value: Union[torch.device, str]) -> None:
        """
        Set a `torch.device` for computation.

        Args:
            value (Union[torch.device, str]): A `torch.device` or a `str`.

        Raises:
            ValueError: raised if `value` isn't a `torch.device` or acceptable `str`.
        """
        self.device = value

    @abstractmethod
    def __call__(self, text: str) -> None:
        """
        Given a text input generates necessary elements for visualization. Multiple text input is not supported.

        Args:
            text (str): Text input

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, text: str):
        """
        Given a text input generates necessary elements for visualization. Multiple text input is not supported. Work in place.

        Args:
            text (str): Text input
        """
        self.__call__(text)
        return self

    @abstractmethod
    def plot(self, **plot_kwargs) -> None:
        """
        Plot the Visualizer. The purpose of kwargs are used to setup plotting parameters.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the class name, the model and the tokenizer of the Visualizer.

        Returns:
            str: Class name, model and tokenizer.
        """
        s = f"{self.__class__.__name__}("
        s += f"model={self.model.__class__.__name__}, "
        s += f"tokenizer={self.tokenizer.__class__.__name__})"
        return s
