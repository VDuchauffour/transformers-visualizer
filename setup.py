from setuptools import setup

from transformers_visualizer import __version__

setup(
    name="transformers-visualizer",
    version=__version__,
    description=(
        "Explain your 🤗 transformers without effort! Plot the internal behavior of"
        " your model."
    ),
    author="Vincent Duchauffour",
    author_email="vincent.duchauffour@proton.me",
    url="https://github.com/VDuchauffour/transformers-visualizer",
    license="Apache-2.0",
    packages=["transformers_visualizer", "transformers_visualizer.visualizers"],
    install_requires=[
        "captum>=0.5.0",
        "transformers>=4.0.0",
        "matplotlib>=3.5",
        "torchtyping>=0.1.4",
    ],
    extras_require={
        "dev": [
            "flake8",
            "bandit",
            "black",
            "isort",
            "mypy",
            "pre-commit",
            "pydocstyle",
            "interrogate",
            "pytest",
            "ipython",
            "ipykernel",
            "ipywidgets",
            "nb-black",
            "fire",
        ]
    },
    keywords=[
        "machine learning",
        "natural language processing",
        "nlp",
        "explainability",
        "transformers",
        "model interpretability",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Documentation :: Sphinx",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
