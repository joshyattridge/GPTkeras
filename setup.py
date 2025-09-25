from setuptools import setup
import codecs
import os

VERSION = "0.0.1"
DESCRIPTION = "Toolkit for training Keras models with GPT-assisted workflows"

# read the contents of the README file
with codecs.open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="KerasGPT",
    version=VERSION,
    author="Joshua Attridge",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=["KerasGPT"],
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
    url="https://github.com/joshyattridge/KerasGPT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
