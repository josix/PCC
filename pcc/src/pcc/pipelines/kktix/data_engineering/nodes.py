"""
This is a boilerplate pipeline 'kktix'
generated using Kedro 0.17.7
"""
import logging
from datetime import datetime
from typing import List, Dict, Any
from itertools import count

import networkx as nx
import jieba.analyse

from pcc.schemas.kktix import Event, Transaction

log = logging.getLogger(__name__)

string_to_index_store: Dict[str, int] = {}
ids = count(0)


def process_event_metadata(
    events: List[Dict[str, Any]], content_use_fields: List[str], top_k_keywords: int
) -> List[Dict[str, Any]]:
    """
    Extract required fields and also running textrank to take top k keywords in descriptions
    """
    event_required_fields = frozenset(content_use_fields)
    log.info(f"{content_use_fields=}")
    processed_events = []
    for event in events:
        if len(set(event.keys()) & event_required_fields) != len(event_required_fields):
            continue
        event = Event(
            event_id=event["event_id"],
            event_name=event["event_name"],
            event_description=event["event_description"],
            first_publish_at=int(event["first_publish_at"]) // 1000,
        )
        event.title_keywords = jieba.analyse.extract_tags(event.event_name)
        event.description_keywords = jieba.analyse.textrank(
            event.event_description,
            topK=top_k_keywords,
            withWeight=False,
            allowPOS=("ns", "n"),
        )
        processed_events.append(event.__dict__)
    return processed_events


def build_raw_interaction_graph(
    behavior_records: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    interaction_use_fields: List[str],
) -> nx.Graph:
    """Sweep out no need information and convert the u-i interaction in networkx format.

    Args:
        behavior_records: Source data.
    Returns:
        Interaction Graph.
    """
    interaction_graph = nx.Graph()
    exist_event = {event["event_id"] for event in events}
    txn_fields = frozenset(interaction_use_fields)
    log.info(f"{txn_fields=}")
    for record in behavior_records:
        if len(set(record.keys()) & txn_fields) != len(txn_fields):
            continue
        if record["event_id"] not in exist_event:
            continue
        txn = Transaction(
            kktix_msno=record["kktix_msno"],
            event_id=record["event_id"],
            ts=record["ts"],
        )
        user = f"u-{txn.kktix_msno}"
        item = f"i-{txn.event_id}"
        date = str(datetime.fromtimestamp(txn.ts // 1000))
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


def build_raw_content_graph(
    events: List[Dict[str, Any]],
) -> nx.Graph:
    """
    Extract required fields and build item-word graph.
    For the description in event we use textrank to extract the top k importants keywords
    """
    content_graph = nx.Graph()
    for event in events:
        event = Event(
            event_name=event["event_name"],
            event_id=event["event_id"],
            event_description=event["event_description"],
            first_publish_at=event["first_publish_at"],
            title_keywords=event["title_keywords"],
            description_keywords=event["description_keywords"],
        )
        item = f"i-{event.event_id}"
        if item not in content_graph:
            if item not in string_to_index_store:
                idx = next(ids)
                content_graph.add_node(item, index=idx, type="I")
                string_to_index_store[item] = idx
            else:
                content_graph.add_node(
                    item, index=string_to_index_store[item], type="I"
                )
        if event.title_keywords and event.description_keywords:
            words = event.title_keywords + event.description_keywords
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
