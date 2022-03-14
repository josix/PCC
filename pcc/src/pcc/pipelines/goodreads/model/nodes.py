"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""
import logging
from dataclasses import asdict
from typing import Dict, List, Any
from pathlib import Path

import networkx as nx

from pcc.schemas.goodreads import ItemSeenStatus
from pcc.schemas.common import OutputModels, SmoreTrainResult
from pcc.utils.smore_helper import run_smore_command

log = logging.getLogger(__name__)


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
    users = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "U"
    ]
    items = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "I"
    ]
    words = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "W"
    ]
    edges = interaction_graph.edges()
    log.info(
        f"After Removing Unseen items Nodes: #edges: {len(edges)} #users: {len(users)} #items:{len(items)} #words:{len(words)}"
    )
    return interaction_graph


def build_semantic_content_graph(
    interaction_graph: nx.Graph,
    content_graph: nx.Graph,
) -> nx.Graph:
    """
    Take item node as joint, connect each user and the words of items and build out a U-W graph.
    """
    semantic_content_graph = nx.Graph()
    for node, attrs in interaction_graph.nodes(data=True):
        if attrs["type"] != "U":
            continue
        user = node
        semantic_content_graph.add_node(user, **attrs)
        user_history = {item for item in interaction_graph[user]}
        for item in user_history:
            for word in content_graph[item]:
                if word not in semantic_content_graph:
                    semantic_content_graph.add_node(word, **content_graph.nodes[word])
                semantic_content_graph.add_edge(user, word)
    users = [
        n
        for n, attrs in semantic_content_graph.nodes(data=True)
        if attrs["type"] == "U"
    ]
    items = [
        n
        for n, attrs in semantic_content_graph.nodes(data=True)
        if attrs["type"] == "I"
    ]
    words = [
        n
        for n, attrs in semantic_content_graph.nodes(data=True)
        if attrs["type"] == "W"
    ]
    edges = semantic_content_graph.edges()
    log.info(
        f"Build Semantic Content Graph: #edges: {len(edges)} #users: {len(users)} #items:{len(items)} #words:{len(words)}"
    )
    return semantic_content_graph


def export_smore_format(
    interaction_graph: nx.Graph,
    content_graph: nx.Graph,
    semantic_content_graph: nx.Graph,
) -> List[str]:
    """
    Export graph in smore format
    """
    output: List[str] = []
    for graph in (interaction_graph, content_graph, semantic_content_graph):
        converted_line = []
        for source, target, attrs in graph.edges(data=True):
            weight = 1 if "weight" not in attrs else attrs["weight"]
            source_node_idx = graph.nodes[source]["index"]
            target_node_idx = graph.nodes[target]["index"]
            converted_line.append(f"{source_node_idx}\t{target_node_idx}\t{weight}")
        converted_line.append("")
        output.append("\n".join(converted_line))
    return output


def smore_train(
    interaction_graph,
    content_graph,
    semantic_content_graph,
    training_graph_configs,
) -> List[Dict[str, Any]]:
    """
    Training on interaction graph, content graph, semantic_content_graph
    according to training configurations saperately.
    """
    graph_type_to_model_result: Dict[str, OutputModels] = {
        graph_type: OutputModels(outputs=[])
        for graph_type in training_graph_configs.keys()
    }
    for graph_type, graph_configuration in training_graph_configs.items():
        file_path = Path(graph_configuration["path"])
        for model, parameters in graph_configuration["models"].items():
            model_output: SmoreTrainResult = run_smore_command(
                model_name=model, parameters=parameters, file_path=file_path
            )
            graph_type_to_model_result[graph_type].outputs.append(model_output)
    return [
        asdict(graph_type_to_model_result["interaction"]),
        asdict(graph_type_to_model_result["content"]),
        asdict(graph_type_to_model_result["semantic_content"]),
    ]
