from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from transformers_visualizer.plotting import plot_token_to_token
from transformers_visualizer.visualizer import Visualizer


class TokenToTokenAttentions(Visualizer):
    """
    Visualizer for visualizer token-to-token attentions matrices.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        """

        Args:
            model (PreTrainedModel): A model given by HuggingFace ecosystem.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A tokenizer given by HuggingFace ecosystem.
        """
        super().__init__(model, tokenizer)
        self.model.config.output_attentions = True

    def __call__(self, text: str) -> None:
        """
        Given a text input generates necessary elements for visualization.

        Args:
            text (str): Text input.
        """
        self.tokens = self.tokenizer(text, return_tensors="pt")
        self.all_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokens["input_ids"].squeeze()  # type: ignore
        )

        self.output = self.model(**self.tokens)
        self.attentions = torch.stack(self.output.attentions).squeeze()

    def visualize(self, layer_index: int = -1, **kwargs) -> None:
        """
        Plot the Visualizer. The purpose of kwargs are used to setup matplotlib parameter.

        Args:
            layer_index (int): Layer index of the model to be plotted. Defaults to last layer, i.e. -1.

        Raises:
            ValueError: _description_
        """
        max_layer_index = self.model.config.num_hidden_layers - 1
        if layer_index is None or layer_index < -1 or layer_index > max_layer_index:
            raise ValueError(
                f"`layer_index` expect value between -1 and {max_layer_index}, got"
                f" {layer_index} instead."
            )

        if not hasattr(self, "attentions") or not hasattr(self, "all_tokens"):
            raise RuntimeError(
                "Visualizer must be called with text input before applying `visualize`"
                " method."
            )

        plot_token_to_token(
            self.attentions[layer_index].detach().cpu().numpy(),
            self.all_tokens,
            "Head",
            **kwargs,
        )
        plt.show

    def __str__(self) -> str:
        """
        Return the class name of the Visualizer.

        Returns:
            str: Class name.
        """
        return super().__str__()

    def __repr__(self) -> str:
        """
        Return the class name, the model and the tokenizer of the Visualizer.

        Returns:
            str: Class name, model and tokenizer.
        """
        return super().__repr__()


class TokenToTokenNormalizedAttentions(Visualizer):
    """
    Visualizer for visualizer token-to-token attentions matrices normalized on `attention_heads` dimension.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        """

        Args:
            model (PreTrainedModel): A model given by HuggingFace ecosystem.
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A tokenizer given by HuggingFace ecosystem.
        """
        super().__init__(model, tokenizer)
        self.model.config.output_attentions = True

    def __call__(self, text: str, order: Optional[Union[str, int]] = None):
        """
        Given a text input generates necessary elements for visualization.

        Args:
            text (str): Text input.
            order (Optional[Union[str, int]]): Order used when `torch.norm` is applied.
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        self.all_tokens = self.tokenizer.convert_ids_to_tokens(
            tokens["input_ids"].squeeze()  # type: ignore
        )

        self.output = self.model(**tokens)
        self.attentions = torch.stack(self.output.attentions).squeeze()
        self.normalized_attentions = torch.linalg.norm(
            self.attentions, dim=1, ord=order
        )

    # TODO: handle layer name with sequence + None case (squeeze do the trick)
    def visualize(
        self, layer_index: Optional[Union[int, Sequence[int]]] = None, **kwargs
    ) -> None:
        """
        Plot the Visualizer. The purpose of kwargs are used to setup matplotlib parameter.

        Args:
            layer_index (Optional[int], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            RuntimeError: _description_
        """
        max_layer_index = self.model.config.num_hidden_layers - 1
        if isinstance(layer_index, int):
            if layer_index < -1 or layer_index > max_layer_index:
                raise ValueError(
                    f"`layer_index` expect value between -1 and {max_layer_index}, got"
                    f" {layer_index} instead."
                )
            else:
                layer_index = [layer_index]

        if (
            not hasattr(self, "attentions")
            or not hasattr(self, "normalized_attentions")
            or not hasattr(self, "all_tokens")
        ):
            raise RuntimeError(
                "Visualizer must be called with text input before applying `visualize`"
                " method."
            )

        plot_token_to_token(
            self.normalized_attentions[layer_index].squeeze().detach().cpu().numpy(),
            self.all_tokens,
            "Layer",
            **kwargs,
        )
        plt.show

    def __str__(self) -> str:
        """
        Return the class name of the Visualizer.

        Returns:
            str: Class name.
        """
        return super().__str__()

    def __repr__(self) -> str:
        """
        Return the class name, the model and the tokenizer of the Visualizer.

        Returns:
            str: Class name, model and tokenizer.
        """
        return super().__repr__()
