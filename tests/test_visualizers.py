import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from transformers_visualizer import (
    TokenToTokenAttentions,
    TokenToTokenNormalizedAttentions,
)
from transformers_visualizer.visualizer import Visualizer

VISUALIZERS = [TokenToTokenAttentions, TokenToTokenNormalizedAttentions]
MODEL_NAME = "bert-base-uncased"
MODEL = AutoModel.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.mark.parametrize("visualizer", VISUALIZERS)
def test_device(visualizer):
    v = visualizer(MODEL, TOKENIZER)

    v.set_device("cpu")
    assert v.device == torch.device("cpu")

    if torch.cuda.is_available():
        v.set_device("cuda")
        assert v.device == torch.device("cuda")


@pytest.mark.parametrize("visualizer", VISUALIZERS)
def test_text_input(visualizer):
    v = visualizer(MODEL, TOKENIZER)
    with pytest.raises(NotImplementedError):
        for input in ["test", ["test"], {"test"}]:
            v.compute(input)
