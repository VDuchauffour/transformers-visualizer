from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from torchtyping import TensorType

sequence_input, num_head = None, None


class SubplotColumnsConfig(Enum):
    """
    Number of subplot columns given plot type.
    """

    TOKEN_TO_TOKEN = 3
    TOKEN_TO_HEAD = 2


class SubplotConfig:
    """
    Subplot config given plot type.
    """

    def __init__(self, length_dimension: int, plot_type: SubplotColumnsConfig):
        """
        Instantiate a subplot config. Set `n_cols` and `n_rows` given the type of plot.

        Args:
            length_dimension (int): length of the dimension to plot.
            plot_type (SubplotColumnsConfig): plot type.
        """
        self.plot_type = plot_type.name
        self.n_cols = plot_type.value
        self.n_rows = max(1, length_dimension // self.n_cols)


def plot_token_to_token(
    matrice: TensorType["sequence_input", "sequence_input"],
    tokens: List[str],
    dimension_type: str,
    num_dimension: int,
    ax: Optional[plt.Axes] = None,
    ticks_fontsize: int = 7,
    title_fontsize: int = 9,
    cmap: str = "viridis",
    colorbar: bool = True,
) -> plt.Axes:
    """
    Generates a token-to-token matplotlib plot. A `torch.Tensor` is expected as input.

    Args:
        matrice (torch.Tensor): `torch.Tensor` to plot. Expect tensor of dimension `"sequence_input", "sequence_input"`.
        tokens (List[str]): List of tokens to plot on x and y ticks label.
        dimension_type (str): Dimension name to plot.
        num_dimension (int): Dimension number plotted in title.
        figsize (Tuple[int, int], optional): Figsize of the plot. Defaults to (20, 20).
        ticks_fontsize (int, optional): Ticks fontsize. Defaults to 7.
        title_fontsize (int, optional): Title fontsize. Defaults to 9.
        cmap (str, optional): Colormap. Defaults to "viridis".
        colorbar (bool, optional): Display colorbars. Defaults to True.

    Returns:
        plt.Axes: The generated axes.
    """
    if ax is None:
        ax = plt.gca()
    assert ax is not None

    im = ax.imshow(matrice.detach().cpu().numpy(), cmap=cmap)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontdict={"fontsize": ticks_fontsize}, rotation=90)
    ax.set_yticklabels(tokens, fontdict={"fontsize": ticks_fontsize})
    ax.set_title(
        f"{dimension_type} {num_dimension}",
        fontdict={"fontsize": title_fontsize},
    )
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.15, pad=0.05)
    return ax


def plot_token_to_token_specific_dimension(
    matrices: TensorType[..., "sequence_input", "sequence_input"],
    tokens: List[str],
    dimension_name: str,
    figsize: Tuple[int, int] = (20, 20),
    ticks_fontsize: int = 7,
    title_fontsize: int = 9,
    cmap: str = "viridis",
    colorbar: bool = True,
):
    """
    Generates a token-to-token matplotlib plot for all indices of a specific dimension. A `torch.Tensor` is expected as input.

    Args:
        matrices (torch.Tensor): `torch.Tensor` to plot. Expect tensor of dimension `"...", "sequence_input", "sequence_input"`.
        tokens (List[str]): List of tokens to plot on x and y ticks label.
        dimension_name (str): Dimension name to plot.
        figsize (Tuple[int, int], optional): Figsize of the plot. Defaults to (20, 20).
        ticks_fontsize (int, optional): Ticks fontsize. Defaults to 7.
        title_fontsize (int, optional): Title fontsize. Defaults to 9.
        cmap (str, optional): Colormap. Defaults to "viridis".
        colorbar (bool, optional): Display colorbars. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    length_dimension = matrices.size(0)
    subplot_config = SubplotConfig(
        length_dimension, SubplotColumnsConfig.TOKEN_TO_TOKEN
    )

    fig = plt.figure(figsize=figsize)
    for idx, matrice in enumerate(matrices):
        ax = fig.add_subplot(
            subplot_config.n_rows,
            subplot_config.n_cols,
            idx + 1,
        )
        ax = plot_token_to_token(
            matrice,
            tokens,
            dimension_name,
            idx + 1,
            ax,
            ticks_fontsize,
            title_fontsize,
            cmap,
            colorbar,
        )

    plt.tight_layout()
    return plt.gcf()
