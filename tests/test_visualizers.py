import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from transformers_visualizer import (
    TokenToTokenAttentions,
    TokenToTokenNormalizedAttentions,
)


@pytest.mark.parametrize(
    "visualizer", [TokenToTokenAttentions, TokenToTokenNormalizedAttentions]
)
def test_device(visualizer):
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    v = visualizer(model, tokenizer)

    assert v.device == torch.device("cpu")

    if torch.cuda.is_available():
        v.set_device("cuda")
        assert v.device == torch.device("cuda")
