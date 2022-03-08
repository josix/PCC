"""
This is a boilerplate pipeline 'goodreads'
generated using Kedro 0.17.7
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from itertools import count

import networkx as nx
from gensim.utils import tokenize
from gensim.summarization import keywords
from gensim.parsing.preprocessing import remove_stopwords

from pcc.schemas.goodreads import Review, Book


log = logging.getLogger(__name__)

string_to_index_store: Dict[str, int] = {}
ids = count(0)


def build_raw_interaction_graph(
    behavior_records: List[Dict[str, Any]],
    books: List[Dict[str, Any]],
    interaction_use_fields: List[str],
) -> nx.Graph:
    """Sweep out no need information and convert the u-i interaction in networkx format.

    Args:
        behavior_records: Source data.
    Returns:
        Interaction Graph.
    """
    interaction_graph = nx.Graph()
    datetime_format = "%a %b %d %H:%M:%S %z %Y"
    exist_books = {book["book_id"] for book in books}
    review_fields = frozenset(interaction_use_fields)
    for record in behavior_records:
        if len(set(record.keys()) & review_fields) != len(review_fields):
            continue
        if record["book_id"] not in exist_books:
            continue
        review = Review(
            user_id=record["user_id"],
            book_id=record["book_id"],
            rating=int(record["rating"]),
            review_text=record["review_text"],
            date_updated=datetime.strptime(record["date_updated"], datetime_format),
        )
        user = f"u-{review.user_id}"
        item = f"i-{review.book_id}"
        date = str(review.date_updated.date())
        if user not in interaction_graph:
            if user not in string_to_index_store:
                idx = next(ids)
                string_to_index_store[user] = idx
                interaction_graph.add_node(user, index=idx, type="U")
            else:
                interaction_graph.add_node(
                    user, index=string_to_index_store[user], type="U"
                )
        if item not in interaction_graph:
            if item not in string_to_index_store:
                idx = next(ids)
                interaction_graph.add_node(item, index=idx, type="I")
                string_to_index_store[item] = idx
            else:
                interaction_graph.add_node(
                    item, index=string_to_index_store[item], type="I"
                )
        interaction_graph.add_edge(user, item, date=date)
    users = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "U"
    ]
    items = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "I"
    ]
    edges = interaction_graph.edges()
    log.info(
        f"Build Interaction Graph: #edges: {len(edges)} #users: {len(users)} #items:{len(items)}"
    )
    log.info(f"#Current loaded nodes: {len(string_to_index_store)}")
    return interaction_graph


def process_book_metadata(
    books: List[Dict[str, Any]], content_use_fields: List[str], top_k_keywords: int
) -> List[Dict[str, Any]]:
    """
    Extract required fields and also running textrank to take top k keywords in descriptions
    """
    book_required_fields = frozenset(content_use_fields)
    processed_books = []
    for book in books:
        if len(set(book.keys()) & book_required_fields) != len(book_required_fields):
            continue
        if book["language_code"] not in {"eng", "en-US"}:
            continue
        book = Book(
            title=book["title"],
            title_without_series=book["title_without_series"],
            book_id=book["book_id"],
            description=book["description"],
            similar_books=book["similar_books"],
        )
        book.title_keywords = list(
            tokenize(remove_stopwords(book.title_without_series), lowercase=True)
        )
        book.description_keywords = str(
            keywords(remove_stopwords(book.description))
        ).split("\n")[:top_k_keywords]
        processed_books.append(book.__dict__)
    return processed_books


def build_raw_content_graph(
    books: List[Dict[str, Any]],
) -> nx.Graph:
    """
    Extract required fields and build item-word graph.
    For the description in book we use textrank to extract the top k importants keywords
    """
    content_graph = nx.Graph()
    for book in books:
        book = Book(
            title=book["title"],
            title_without_series=book["title_without_series"],
            book_id=book["book_id"],
            description=book["description"],
            similar_books=book["similar_books"],
            title_keywords=book["title_keywords"],
            description_keywords=book["description_keywords"],
        )
        item = f"i-{book.book_id}"
        if item not in content_graph:
            if item not in string_to_index_store:
                idx = next(ids)
                content_graph.add_node(item, index=idx, type="I")
                string_to_index_store[item] = idx
            else:
                content_graph.add_node(
                    item, index=string_to_index_store[item], type="I"
                )
        if book.title_keywords and book.description_keywords:
            words = book.title_keywords + book.description_keywords
            for word in words:
                word = f"w-{word}"
                if word not in content_graph:
                    if word not in string_to_index_store:
                        idx = next(ids)
                        string_to_index_store[word] = idx
                        content_graph.add_node(word, index=idx, type="W")
                    else:
                        content_graph.add_node(
                            word, index=string_to_index_store[word], type="W"
                        )
                content_graph.add_edge(item, word)

    words = [n for n, attrs in content_graph.nodes(data=True) if attrs["type"] == "W"]
    items = [n for n, attrs in content_graph.nodes(data=True) if attrs["type"] == "I"]
    edges = content_graph.edges()
    log.info(
        f"Build Content Graph: #edges: {len(edges)} #items: {len(items)} #words:{len(words)}"
    )
    log.info(f"#Current loaded nodes: {len(string_to_index_store)}")
    return content_graph


def remove_sparse_nodes(
    interaction_graph: nx.Graph,
    content_graph: nx.Graph,
    degree_limit_config: Dict[str, int],
) -> Tuple[nx.Graph, nx.Graph]:
    """Removing nodes that degree lower than given limit

    Args:
        interactions: source interaction graph
    Returns:
        Interaction object after removing nodes with sparse degree.
    """
    for graph in (interaction_graph, content_graph):
        if graph is interaction_graph:
            degree_limit = degree_limit_config["interaction_graph"]
        else:
            degree_limit = degree_limit_config["content_graph"]
        unqulified_nodes = [n for n in graph.nodes() if graph.degree[n] < degree_limit]
        graph.remove_nodes_from(unqulified_nodes)
        users = [n for n, attrs in graph.nodes(data=True) if attrs["type"] == "U"]
        items = [n for n, attrs in graph.nodes(data=True) if attrs["type"] == "I"]
        words = [n for n, attrs in graph.nodes(data=True) if attrs["type"] == "W"]
        edges = graph.edges()
        log.info(
            f"After Removing Sparse Nodes: #edges: {len(edges)} #users: {len(users)} #items:{len(items)} #words:{len(words)}"
        )
    return interaction_graph, content_graph
