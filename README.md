<h1 align="center">Transformers visualizer</h1>
<p align="center">Explain your 🤗 transformers without effort!</p>
<h1 align="center"></h1>

<p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img alt="Apache" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
    <a href="https://github.com/VDuchauffour/transformers-visualizer/blob/main/.github/workflows/unit_tests.yml">
        <img alg="Unit tests" src="https://github.com/VDuchauffour/transformers-visualizer/actions/workflows/unit_tests.yml/badge.svg">
    </a>
    <a href="">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/transformers-visualizer?color=red">
    </a>
    <a href="https://github.com/VDuchauffour/transformers-visualizer">
        <img alt="PyPI - Package Version" src="https://img.shields.io/pypi/v/transformers-visualizer?label=version">
    </a>
</p>

Transformers visualizer is a python package designed to work with the [🤗 transformers](https://huggingface.co/docs/transformers/index) package. Given a `model` and a `tokenizer`, this package supports multiple ways to explain your model by plotting its internal behavior.

This package is mostly based on the [Captum][Captum] tutorials [[1]][captum_part1] [[2]][Captum_part2].

## Installation

```shell
pip install transformers-visualizer
```

## Quickstart

Let's define a model, a tokenizer and a text input for the following examples.

```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."
```

### Visualizers

<details><summary>Attention matrices of a specific layer</summary>

<p>

```python
from transformers_visualizer import TokenToTokenAttentions

visualizer = TokenToTokenAttentions(model, tokenizer)
visualizer(text)
```

Instead of using `__call__` function, you can use the `compute` method. Both work in place, `compute` method allows chaining method.

`plot` method accept a layer index as parameter to specify which part of your model you want to plot. By default, the last layer is plotted.

```python
import matplotlib.pyplot as plt

visualizer.plot(layer_index = 6)
plt.savefig("token_to_token.jpg")
```

<p align="center">
    <img alt="token to token" src="https://raw.githubusercontent.com/VDuchauffour/transformers-visualizer/main/images/token_to_token.jpg" />
</p>

</p>

</details>

<details><summary>Attention matrices normalized across head axis</summary>

<p>

You can specify the `order` used in `torch.linalg.norm` in `__call__` and `compute` methods. By default, an L2 norm is applied.

```python
from transformers_visualizer import TokenToTokenNormalizedAttentions

visualizer = TokenToTokenNormalizedAttentions(model, tokenizer)
visualizer.compute(text).plot()
```

<p align="center">
    <img alt="normalized token to token"src="https://raw.githubusercontent.com/VDuchauffour/transformers-visualizer/main/images/token_to_token_normalized.jpg" />
</p>

</p>

</details>

## Plotting

`plot` method accept to skip special tokens with the parameter `skip_special_tokens`, by default it's set to `False`.

You can use the following imports to use plotting functions directly.

```python
from transformers_visualizer.plotting import plot_token_to_token, plot_token_to_token_specific_dimension
```

These functions or the `plot` method of a visualizer can use the following parameters.

- `figsize (Tuple[int, int])`: Figsize of the plot. Defaults to (20, 20).
- `ticks_fontsize (int)`: Ticks fontsize. Defaults to 7.
- `title_fontsize (int)`: Title fontsize. Defaults to 9.
- `cmap (str)`: Colormap. Defaults to "viridis".
- `colorbar (bool)`: Display colorbars. Defaults to True.

## Upcoming features

- [x] Add an option to mask special tokens.
- [ ] Add an option to specify head/layer indices to plot.
- [ ] Add other plotting backends such as Plotly, Bokeh, Altair.
- [ ] Implement other visualizers such as [vector norm](https://arxiv.org/pdf/2004.10102.pdf).

## References

- [[1]][captum_part1] Captum's BERT tutorial (part 1)
- [[2]][captum_part2] Captum's BERT tutorial (part 2)

## Acknowledgements

- [Transformers Interpret](https://github.com/cdpierse/transformers-interpret) for the idea of this project.

[Captum]: https://captum.ai/
[captum_part1]: https://captum.ai/tutorials/Bert_SQUAD_Interpret
[Captum_part2]: https://captum.ai/tutorials/Bert_SQUAD_Interpret2