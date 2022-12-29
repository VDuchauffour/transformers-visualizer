from typing import List, Optional, Set, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from transformers_visualizer.errors import (
    DimensionIndicesError,
    OutputNotComputedEror,
    TextInputError,
)
from transformers_visualizer.plotting import plot_token_to_token_specific_dimension
from transformers_visualizer.visualizer import Visualizer


class TokenToTokenAttentions(Visualizer):
    """
    Visualizer for plotting token-to-token attentions matrices.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Create a token-to-token attention matrices visualizer. Plot attention matrices of a specific layer index given a model and a tokenizer.

        Args:
            model (PreTrainedModel): A `model` given by HuggingFace ecosystem.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A `tokenizer` given by HuggingFace ecosystem.
            device (Optional[Union[torch.device, str]]): A `device` where computation takes place. If no `torch.device` are provided, the `model`'s device is used.
        """
        super().__init__(model, tokenizer, device)
        self.model.config.output_attentions = True

    def set_device(self, value: Union[torch.device, str]) -> None:
        """
        Set a `torch.device` for computation.

        Args:
            value (Union[torch.device, str]): A `torch.device` or a `str`.

        Raises:
            ValueError: raised if `value` isn't a `torch.device` or acceptable `str`.
        """
        super().set_device(value)

    def __call__(self, text: str) -> None:
        """
        Given a text input generates necessary elements for visualization. Multiple text input is not supported.

        Args:
            text (str): Text input.
        """
        if isinstance(text, Tuple) or isinstance(text, List) or isinstance(text, Set):  # type: ignore
            raise TextInputError

        self.tokens = self.tokenizer(text, return_tensors="pt").to(self._device)

        self.output = self.model(**self.tokens)
        self.attentions = torch.concat(self.output.attentions)

    def compute(self, text: str):
        """
        Given a text input generates necessary elements for visualization. Multiple text input is not supported. Work in place.

        Args:
            text (str): Text input.
        """
        return super().compute(text)

    def plot(
        self, layer_index: int = -1, skip_special_tokens: bool = False, **plot_kwargs
    ) -> None:
        """
        Plot the Visualizer. The purpose of kwargs are used to setup plotting parameters.

        Args:
            layer_index (int): Layer index of the model to be plotted. Defaults to last layer, i.e. -1.
            skip_special_tokens (bool): If True, plot doesn't contains special tokens. Defaults to False.
            figsize (Tuple[int, int], optional): Figsize of the plot. Defaults to (20, 20).
            ticks_fontsize (int, optional): Ticks fontsize. Defaults to 7.
            title_fontsize (int, optional): Title fontsize. Defaults to 9.
            cmap (str, optional): Colormap. Defaults to "viridis".
            colorbar (bool, optional): Display colorbars. Defaults to True.

        Raises:
            OutputNotComputedEror: raised if no `Output` present.
        """
        if not hasattr(self, "attentions"):
            raise OutputNotComputedEror

        max_layer_index = self.model.config.num_hidden_layers - 1
        try:
            attentions = self.attentions[layer_index]
        except IndexError:
            raise DimensionIndicesError("layer_index", layer_index, max_layer_index)

        if skip_special_tokens:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                self.tokens.input_ids.squeeze(), already_has_special_tokens=True
            )
            special_tokens_mask = (
                torch.where(torch.tensor(special_tokens_mask) == 1, 0, 1)
                .to(torch.bool)
                .squeeze()
            )
            attentions = attentions[..., special_tokens_mask]
            attentions = attentions[:, special_tokens_mask]

        all_tokens: List[str] = self.tokenizer.convert_ids_to_tokens(
            self.tokens["input_ids"].squeeze(), skip_special_tokens=skip_special_tokens  # type: ignore
        )

        plot_token_to_token_specific_dimension(
            attentions,
            all_tokens,
            "Head",
            **plot_kwargs,
        )

    def __str__(self) -> str:
        """
        Return the class name, the model and the tokenizer of the Visualizer.

        Returns:
            str: Class name, model and tokenizer.
        """
        return super().__str__()


class TokenToTokenNormalizedAttentions(Visualizer):
    """
    Visualizer for plotting token-to-token attentions matrices normalized on `attention_heads` dimension.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Create a token-to-token normalized attention matrices visualizer. Plot attention matrices normalized across head axis given a model and a tokenizer.

        Args:
            model (PreTrainedModel): A `model` given by HuggingFace ecosystem.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A `tokenizer` given by HuggingFace ecosystem.
            device (Optional[Union[torch.device, str]]): A `device` where computation takes place. If no `torch.device` are provided, the `model`'s device is used.
        """
        super().__init__(model, tokenizer, device)
        self.model.config.output_attentions = True

    def set_device(self, value: Union[torch.device, str]):
        """
        Set a `torch.device` for computation.

        Args:
            value (Union[torch.device, str]): A `torch.device` or a `str`.

        Raises:
            ValueError: raised if `value` isn't a `torch.device` or acceptable `str`.
        """
        return super().set_device(value)

    def __call__(self, text: str, order: Optional[Union[str, int]] = None) -> None:
        """
        Given a text input generates necessary elements for visualization. Multiple text input is not supported.

        Args:
            text (str): Text input.
            order (Optional[Union[str, int]]): Order used when `torch.norm` is applied.
        """
        if isinstance(text, Tuple) or isinstance(text, List) or isinstance(text, Set):  # type: ignore
            raise TextInputError

        self.tokens = self.tokenizer(text, return_tensors="pt").to(self._device)

        self.output = self.model(**self.tokens)
        self.attentions = torch.concat(self.output.attentions)
        self.normalized_attentions = torch.linalg.norm(
            self.attentions, dim=1, ord=order
        )

    def compute(self, text: str, order: Optional[Union[str, int]] = None):
        """
        Given a text input generates necessary elements for visualization. Multiple text input is not supported. Work in place.

        Args:
            text (str): Text input.
        """
        self.__call__(text, order)
        return self

    def plot(self, skip_special_tokens: bool = False, **plot_kwargs) -> None:
        """
        Plot the Visualizer. The purpose of kwargs are used to setup plotting parameters.

        Args:
            skip_special_tokens (bool): If True, plot doesn't contains special tokens. Defaults to False.
            figsize (Tuple[int, int], optional): Figsize of the plot. Defaults to (20, 20).
            ticks_fontsize (int, optional): Ticks fontsize. Defaults to 7.
            title_fontsize (int, optional): Title fontsize. Defaults to 9.
            cmap (str, optional): Colormap. Defaults to "viridis".
            colorbar (bool, optional): Display colorbars. Defaults to True.

        Raises:
            OutputNotComputedEror: raised if no `Output` present.
        """
        if not hasattr(self, "attentions") or not hasattr(
            self, "normalized_attentions"
        ):
            raise OutputNotComputedEror

        if skip_special_tokens:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                self.tokens.input_ids.squeeze(), already_has_special_tokens=True
            )
            special_tokens_mask = (
                torch.where(torch.tensor(special_tokens_mask) == 1, 0, 1)
                .to(torch.bool)
                .squeeze()
            )
            normalized_attentions = self.normalized_attentions[..., special_tokens_mask]
            normalized_attentions = normalized_attentions[:, special_tokens_mask]
        else:
            normalized_attentions = self.normalized_attentions

        all_tokens: List[str] = self.tokenizer.convert_ids_to_tokens(
            self.tokens["input_ids"].squeeze(), skip_special_tokens=skip_special_tokens  # type: ignore
        )
        plot_token_to_token_specific_dimension(
            normalized_attentions,
            all_tokens,
            "Layer",
            **plot_kwargs,
        )

    def __str__(self) -> str:
        """
        Return the class name, the model and the tokenizer of the Visualizer.

        Returns:
            str: Class name, model and tokenizer.
        """
        return super().__str__()
