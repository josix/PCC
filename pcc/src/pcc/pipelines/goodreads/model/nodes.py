"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""
import random
import logging
from dataclasses import asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

import networkx as nx
import numpy as np

from pcc.schemas.common import OutputModels, Model, ItemSeenStatus
from pcc.utils.smore_helper import run_smore_command
from pcc.utils.aggregator import aggregate

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
    interaction_graph,  # no used since it will trained by external smore process
    content_graph,
    semantic_content_graph,
    training_graph_configs: Dict[str, Any],
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
            model_output: Model = run_smore_command(
                model_name=model, parameters=parameters, file_path=file_path
            )
            graph_type_to_model_result[graph_type].outputs.append(model_output)
    return [
        asdict(graph_type_to_model_result["interaction"]),
        asdict(graph_type_to_model_result["content"]),
        asdict(graph_type_to_model_result["semantic_content"]),
    ]


def aggregate_item_emb(
    content_embedding: Dict[str, Any],
    semantic_content_embedding: Dict[str, Any],
    content_graph: nx.Graph,
    aggregate_configuration,
    include_content_i: bool,
    include_content_w: bool,
) -> Dict[str, Any]:
    """Aggregate embeeding for old and new item by aggreate
    the content embedding and semantic_content_embedding"""
    index_to_embedding: Dict[str, List[float]] = {}
    content_model = Model(
        **[
            model
            for model in content_embedding["outputs"]
            if model["model_name"] == aggregate_configuration["content_graph_model"]
        ][0]
    )
    semantic_content_model = Model(
        **[
            model
            for model in semantic_content_embedding["outputs"]
            if model["model_name"]
            == aggregate_configuration["semantic_content_graph_model"]
        ][0]
    )
    items = [n for n, attrs in content_graph.nodes(data=True) if attrs["type"] == "I"]

    for item in items:
        item_idx = str(content_graph.nodes[item]["index"])
        neighbor_words = list(content_graph.neighbors(item))
        random.shuffle(neighbor_words)
        neighbor_words_semantic_content_embeddings: List[List[float]] = []
        neighbor_words_content_embeddings: List[List[float]] = []
        n_words = min(aggregate_configuration["n_words"], len(neighbor_words))
        # collect n_words of word embeddings of one given item
        for i, word in enumerate(neighbor_words):
            if i + 1 == n_words:
                break
            idx = str(content_graph.nodes[word]["index"])
            if idx in semantic_content_model.index_to_embedding:
                neighbor_words_semantic_content_embeddings.append(
                    semantic_content_model.index_to_embedding[idx]
                )
            if idx in content_model.index_to_embedding:
                neighbor_words_content_embeddings.append(
                    content_model.index_to_embedding[idx]
                )
        # take mean average of word embeddings to build item embedding
        agg_semantic_item_emb: np.ndarray
        agg_content_item_emb: Optional[np.ndarray] = None
        content_item_emb: Optional[np.ndarray] = None
        if not neighbor_words_semantic_content_embeddings:
            agg_semantic_item_emb = np.zeros(
                (1, semantic_content_model.embedding_size)
            )[0]
        else:
            agg_semantic_item_emb = np.mean(
                np.array(neighbor_words_semantic_content_embeddings), axis=0
            )
        if include_content_w:
            if not neighbor_words_content_embeddings:
                agg_content_item_emb = np.zeros(
                    (1, semantic_content_model.embedding_size)
                )
            else:
                agg_content_item_emb = np.mean(
                    np.array(neighbor_words_content_embeddings), axis=0
                )
        if include_content_i:
            content_item_emb = np.array(content_model.index_to_embedding[item_idx])
        # aggregate different type item embedding to represent one item
        if agg_content_item_emb is not None and content_item_emb is not None:
            index_to_embedding[item_idx] = aggregate(
                [content_item_emb, agg_content_item_emb, agg_semantic_item_emb],
                stradegy=aggregate_configuration["stradegy"],
            ).tolist()
        elif agg_content_item_emb is not None and content_item_emb is None:
            index_to_embedding[item_idx] = aggregate(
                [agg_content_item_emb, agg_semantic_item_emb],
                stradegy=aggregate_configuration["stradegy"],
            ).tolist()
        elif agg_content_item_emb is None and content_item_emb is not None:
            index_to_embedding[item_idx] = aggregate(
                [content_item_emb, agg_semantic_item_emb],
                stradegy=aggregate_configuration["stradegy"],
            ).tolist()
        else:
            index_to_embedding[item_idx] = agg_semantic_item_emb.tolist()
    if include_content_i and include_content_w:
        embedding_size = (
            semantic_content_model.embedding_size + content_model.embedding_size * 2
        )
    elif include_content_i or include_content_w:
        embedding_size = (
            semantic_content_model.embedding_size + content_model.embedding_size
        )
    else:
        embedding_size = semantic_content_model.embedding_size

    model_output = Model(
        model_name="pcc",
        embedding_size=embedding_size,
        index_to_embedding=index_to_embedding,
    )
    log.info(f"Training {len(index_to_embedding)} items' pcc embedding is completed")
    return asdict(model_output)
