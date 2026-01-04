"""
Preprocessing package for mlproject.

Provides two approaches:
1. Basic cleaning (clean.py) - meets course requirements
2. Class-based preprocessing (advanced.py) - modular Preprocessor class
"""
from .clean import clean_dataframe, load_and_clean
from .advanced import Preprocessor

__all__ = [
    "clean_dataframe", 
    "load_and_clean",
    "Preprocessor"
]
