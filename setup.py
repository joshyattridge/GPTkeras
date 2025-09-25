from setuptools import setup
import codecs
import os

VERSION = "0.0.2"
DESCRIPTION = "Toolkit for building Keras models with the help of GPT"

# read the contents of the README file
with codecs.open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="GPTkeras",
    version=VERSION,
    author="Joshua Attridge",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=["GPTkeras"],
    install_requires=["tensorflow", "numpy", "openai"],
    keywords=[
        "keras",
        "gpt",
        "tensorflow",
        "openai",
        "machine",
        "learning",
        "chatgpt",
        "ai",
    ],
    url="https://github.com/joshyattridge/GPTkeras",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
