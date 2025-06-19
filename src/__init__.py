"""
Topic Modeling Research Package

A comparative analysis framework for LDA and BERTopic models on Russian text data.
"""

__version__ = "1.0.0"
__author__ = "ITMO University NLP Research"

from .data_preprocessing import TextPreprocessor
from .lda_model import LDAModel
from .bertopic_model import BERTopicModel
from .evaluation import TopicEvaluator
from .visualization import TopicVisualizer

__all__ = [
    "TextPreprocessor",
    "LDAModel", 
    "BERTopicModel",
    "TopicEvaluator",
    "TopicVisualizer"
] 