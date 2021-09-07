import logging
import typing
from datetime import datetime

import networkx as nx


def split_graph(
    processed_interaction_graaph: nx.Graph, split_point: str
) -> typing.List[nx.Graph]:
    """Split source graph into training graph and testing graph

    Args:
        Interaction object after removing nodes with sparse degree.
    Returns:
        List of splitting graph includeing training graph, tseting 
        graph in networkx format
    """
    log = logging.getLogger(__name__)
    split_time = datetime.fromisoformat(split_point)

    training_edges = []
    testing_edges = []
    for source, target, attrs in processed_interaction_graaph.edges(data=True):
        timestamp = datetime.fromtimestamp(attrs["timestamp"])
        if timestamp < split_time:
            training_edges.append((source, target))
        else:
            testing_edges.append((source, target))
    training_graph = processed_interaction_graaph.edge_subgraph(training_edges).copy()
    testing_graph = processed_interaction_graaph.edge_subgraph(testing_edges).copy()
    for graph_type, graph in (
        ("Training Graph", training_graph),
        ("Testing Graph", testing_graph),
    ):
        users = [n for n, attrs in graph.nodes(data=True) if attrs["type"] == "U"]
        items = [n for n, attrs in graph.nodes(data=True) if attrs["type"] == "I"]
        edges = graph.edges()
        log.info(
            f"Build {graph_type}: #edges: {len(edges)} #users: {len(users)} #items:{len(items)}"
        )
    return [training_graph, testing_graph]


def convert_graph_format(training_graph: nx.Graph) -> str:
    output = []
    users = {
        node
        for node, attrs in training_graph.nodes(data=True)
        if attrs and attrs["type"] == "U"
    }
    for user in users:
        for user, item, attrs in training_graph.edges(user, data=True):
            weight = 1 if "weight" not in attrs else attrs["weight"]
            output.append(f"{user}\t{item}\t{weight}")
    return "\n".join(output)


def training() -> None:
    pass


def testing():
    pass
