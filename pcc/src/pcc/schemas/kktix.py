"""
KKTIX data schemas
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Event:
    """
    KKTIX event schema
    """

    event_id: int
    event_name: str
    event_description: str
    first_publish_at: int
    title_keywords: Optional[List[str]] = None
    description_keywords: Optional[List[str]] = None


@dataclass
class Transaction:
    """
    KKTIX transaction schema
    """

    kktix_msno: int
    event_id: int
    ts: int
