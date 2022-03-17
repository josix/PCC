"""
Goodreads data schemas
"""
import datetime
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Book:
    """
    Goodread book schema
    """

    title: str
    title_without_series: str
    book_id: str
    description: str
    similar_books: List[str]
    publication_date: Optional[datetime.datetime] = None
    title_keywords: Optional[List[str]] = None
    description_keywords: Optional[List[str]] = None


@dataclass
class Review:
    """
    Goodread review schema
    """

    user_id: str
    book_id: str
    rating: int
    review_text: str
    date_updated: datetime.datetime
