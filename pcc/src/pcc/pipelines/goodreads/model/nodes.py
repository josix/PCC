"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""
from typing import Dict, List

import networkx as nx

from pcc.schemas.goodreads import ItemSeenStatus


def remove_unseen_items_from_interaction_graph(
    interaction_graph: nx.Graph,
    item_seen_status: Dict[str, List[str]],
) -> nx.Graph:
    """
    Remove unseen_items_from U-I graph
    """
    unseen_items = ItemSeenStatus(
        unseen_items=item_seen_status["unseen_items"],
        seen_items=item_seen_status["seen_items"],
    ).unseen_items
    interaction_graph.remove_nodes_from(unseen_items)
    return interaction_graph


def build_semantic_content_graph(
    interaction_graph: nx.Graph,
    content_graph: nx.Graph,
) -> nx.Graph:
    """
    Take item node as joint, connect each user and the words of items and build out a U-W graph.
    """
    pass
