from setuptools import setup

setup(
    name="transformers-visualizer",
    version="0.1.0",
    description="Seamlessly display the internal behavior of your 🤗 transformers.",
    author="Vincent Duchauffour",
    author_email="vincent.duchauffour@proton.me",
    url="https://github.com/VDuchauffour/transformers-visualizer",
    license="Apache-2.0",
    packages=["transformers_visualizer"],
    install_requires=["captum>=0.5.0", "transformers>=4.24.0", "matplotlib>=3.6.1"],
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
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)
