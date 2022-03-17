"""
Helper functions for aggregate embedddings based on different stradegy.
"""
from typing import List, Union

import numpy as np


def concat(embeddings: np.ndarray) -> np.ndarray:
    """aggregate embeddings by concatenting them"""
    return np.concatenate(embeddings, axis=0)


def mean(embeddings: np.ndarray) -> np.ndarray:
    """mean average of embeddings"""
    return np.mean(embeddings, axis=0)


def aggregate(
    embeddings: Union[List[np.ndarray], np.ndarray], stradegy="concat"
) -> np.ndarray:
    """
    aggregaet embeddings according the assigned stradegy
    """
    embeddings = np.array(embeddings)
    stradegy_to_aggregator = {
        "concat": lambda: concat(embeddings),
        "mean": lambda: mean(embeddings),
    }
    if stradegy not in stradegy_to_aggregator:
        raise ValueError(f"Invalid aggregate stradegy: {stradegy}")
    return stradegy_to_aggregator[stradegy]()
