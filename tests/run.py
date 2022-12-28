import subprocess

import matplotlib.pyplot as plt
from fire import Fire
from transformers import AutoModel, AutoTokenizer

from transformers_visualizer import (
    TokenToTokenAttentions,
    TokenToTokenNormalizedAttentions,
)

ROOTDIR = (
    subprocess.run("git rev-parse --show-toplevel", shell=True, capture_output=True)
    .stdout.decode()
    .strip()
)


def run_visualizers(model_name: str = "bert-base-uncased", plot: bool = False):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = (
        "The dominant sequence transduction models are based on complex recurrent or"
        " convolutional neural networks that include an encoder and a decoder."
    )

    run = [
        (TokenToTokenAttentions, "token_to_token"),
        (TokenToTokenNormalizedAttentions, "token_to_token_normalized"),
    ]

    for visualizer_class, filename in run:
        visualizer = visualizer_class(model, tokenizer)
        visualizer.compute(text)
        visualizer.plot()
        if plot:
            plt.savefig(f"{ROOTDIR}/images/{filename}.jpg")


if __name__ == "__main__":
    Fire(run_visualizers)
