import logging
import typing
from itertools import count

import networkx as nx


def build_graph(
    behavior_records: typing.List[typing.Dict[str, typing.Any]]
) -> nx.Graph:
    """Preprocess the data for user-item behavior.

    Args:
        behavior_records: Source data.
    Returns:
        Interaction Graph.
    """
    log = logging.getLogger(__name__)
    interaction_graph = nx.Graph()
    ids = count(0)
    for record in behavior_records:
        user = f'u-{record["user"]}'
        item = f'i-{record["article_id"]}'
        timestamp = record["ts"] // 1000
        if user not in interaction_graph:
            interaction_graph.add_node(user, index=next(ids), type="U")
        if item not in interaction_graph:
            interaction_graph.add_node(item, index=next(ids), type="I")
        interaction_graph.add_edge(user, item, timestamp=timestamp)
    users = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "U"
    ]
    items = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "I"
    ]
    edges = interaction_graph.edges()
    log.info(
        f"Build Graph: #edges: {len(edges)} #users: {len(users)} #items:{len(items)}"
    )
    return interaction_graph


def remove_sparse_nodes(interaction_graph: nx.Graph, degree_limit: int) -> nx.Graph:
    """Preprocess the data for behavior.

    Args:
        interactions: source interaction graph
    Returns:
        Interaction object after removing nodes with sparse degree.
    """
    log = logging.getLogger(__name__)
    past_nodes = set(interaction_graph.nodes())
    for node in past_nodes:
        if node in interaction_graph and interaction_graph.degree[node] < degree_limit:
            interaction_graph.remove_node(node)
    users = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "U"
    ]
    items = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "I"
    ]
    edges = interaction_graph.edges()
    log.info(
        f"After Removing Sparse Nodes: #edges: {len(edges)} #users: {len(users)} #items:{len(items)}"
    )
    return interaction_graph
