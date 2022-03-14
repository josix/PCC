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
    index_to_embedding: Dict[int, List[float]]


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
