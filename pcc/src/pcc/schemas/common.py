"""
Common schemas for all pipelines
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SmoreTrainResult:
    """
    Smore training result schema
    """

    model_name: str
    embedding_size: int
    index_to_embedding: Dict[str, List[float]]


@dataclass
class OutputModels:
    """Schema of multiple model outputs of one training graph"""

    outputs: List[SmoreTrainResult]


@dataclass
class Command:
    out: str
    err: str
    stdout: bytes
    stderr: bytes
    return_code: int


@dataclass
class ItemSeenStatus:
    """Lists of seen and unseen items"""

    unseen_items: List[str]
    seen_items: List[str]


@dataclass
class UserProfile:
    """
    User reviewed history, and the queries user prepared to make
    """

    exp_history: List[str]
    actual_history: List[str]
    queries: List[str]
