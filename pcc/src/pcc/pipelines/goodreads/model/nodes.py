"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""
import logging
from dataclasses import asdict
from typing import Dict, List, Any
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from lightfm import LightFM

from pcc.schemas.common import OutputModels, Model, ItemSeenStatus
from pcc.utils.smore_helper import run_smore_command
from pcc.utils.lightfm_helper import LightFMInteraction, get_lightfm_input

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
    interaction_graph: nx.Graph,
    aggregate_configuration,
    use_content_item_emb: bool = False,
) -> Dict[str, Any]:
    """Aggregate embeeding for old and new item by aggreate
    the content embedding and semantic_content_embedding"""
    lightfm_input: LightFMInteraction = get_lightfm_input(interaction_graph)

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
        neighbor_words = [
            node[0]
            for node in sorted(
                content_graph.degree(content_graph.neighbors(item)),
                key=lambda x: x[1],
                reverse=True,
            )
        ][:]
        n_words = min(aggregate_configuration["n_words"], len(neighbor_words))
        neighbor_words = neighbor_words[:n_words]
        neighbor_words_semantic_content_embeddings: List[List[float]] = []
        # collect n_words of word embeddings of one given item
        for word in neighbor_words:
            idx = str(content_graph.nodes[word]["index"])
            if idx in semantic_content_model.index_to_embedding:
                neighbor_words_semantic_content_embeddings.append(
                    semantic_content_model.index_to_embedding[idx]
                )
        agg_semantic_item_emb: np.ndarray
        if not neighbor_words_semantic_content_embeddings:
            agg_semantic_item_emb = np.zeros(
                (1, semantic_content_model.embedding_size)
            )[0]
        else:
            agg_semantic_item_emb = np.sum(
                np.array(neighbor_words_semantic_content_embeddings), axis=0
            )
        index_to_embedding[item_idx] = agg_semantic_item_emb.tolist()

    item_lightfm_idx_to_emb: Dict[int, List[float]] = {}
    for item_idx in index_to_embedding:
        item_idx = int(item_idx)
        if item_idx not in lightfm_input.item_graph_idx_to_model_idx:
            continue
        item_lightfm_idx_to_emb[
            lightfm_input.item_graph_idx_to_model_idx[item_idx]
        ] = index_to_embedding[str(item_idx)]

    item_features = []
    for idx in range(lightfm_input.interaction_matrix.shape[1]):
        item_features.append(item_lightfm_idx_to_emb[idx])
    log.info(f"shape of aggregated item_features {np.array(item_features).shape}")
    item_features = sparse.csr_matrix(np.array(item_features))

    model = LightFM(
        learning_rate=aggregate_configuration["lightfm"]["lr"],
        loss=aggregate_configuration["lightfm"]["loss"],
        no_components=aggregate_configuration["lightfm"]["emb_size"],
    )
    model.fit(
        lightfm_input.interaction_matrix,
        epochs=aggregate_configuration["lightfm"]["epoch"],
        num_threads=20,
        verbose=True,
        item_features=item_features,
    )
    item_bias, item_embeddings = model.get_item_representations(features=item_features)
    lightfm_index_to_embedding: Dict[str, List[float]] = {}
    assert not item_embeddings is None
    for idx, emb in enumerate(item_embeddings):
        lightfm_index_to_embedding[
            str(lightfm_input.item_model_idx_to_graph_idx[idx])
        ] = emb.tolist()

    model_output = Model(
        model_name="pcc",
        embedding_size=aggregate_configuration["lightfm"]["emb_size"],
        index_to_embedding=lightfm_index_to_embedding,
    )
    log.info(
        f"Training {len(lightfm_index_to_embedding)} items' pcc embedding is completed"
    )
    return asdict(model_output)


def lightfm_pcc_smore(
    aggregated_item_embedding: Dict[str, Any],
    interaction_embedding: Dict[str, Any],
    interaction_graph: nx.Graph,
    lightfm_configs: Dict[str, Any],
    smore_model_name: str,
    use_user_feature: bool = False,
):
    """Concatenate trained content-based item embedding and interaction-based smore embedding"""
    lightfm_input: LightFMInteraction = get_lightfm_input(interaction_graph)

    pcc_model = Model(
        model_name=aggregated_item_embedding["model_name"],
        embedding_size=aggregated_item_embedding["embedding_size"],
        index_to_embedding=aggregated_item_embedding["index_to_embedding"],
    )
    smore_model = Model(
        **[
            model
            for model in interaction_embedding["outputs"]
            if model["model_name"] == smore_model_name
        ][0]
    )
    item_model_idx_to_emb: Dict[int, List[float]] = {}
    for item_idx in pcc_model.index_to_embedding:
        if item_idx in smore_model.index_to_embedding:
            output_embedding: List[float] = (
                pcc_model.index_to_embedding[item_idx]
                + smore_model.index_to_embedding[item_idx]
            )
        else:
            output_embedding: List[float] = (
                pcc_model.index_to_embedding[item_idx]
                + [0.0] * smore_model.embedding_size
            )
        item_idx = int(item_idx)
        if item_idx not in lightfm_input.item_graph_idx_to_model_idx:
            continue
        item_model_idx_to_emb[
            lightfm_input.item_graph_idx_to_model_idx[item_idx]
        ] = output_embedding

    item_features = []
    for idx in range(lightfm_input.interaction_matrix.shape[1]):
        item_features.append(item_model_idx_to_emb[idx])
    log.info(f"shape of item_features {np.array(item_features).shape}")
    item_features = sparse.csr_matrix(np.array(item_features))

    user_features = None
    if use_user_feature:
        user_model_idx_to_emb: Dict[int, List[float]] = {}
        for graph_idx in smore_model.index_to_embedding:
            if int(graph_idx) in lightfm_input.user_graph_idx_to_model_idx:
                user_model_idx_to_emb[
                    lightfm_input.user_graph_idx_to_model_idx[int(graph_idx)]
                ] = smore_model.index_to_embedding[graph_idx]

        user_features = []
        for idx in range(lightfm_input.interaction_matrix.shape[0]):
            if idx not in user_model_idx_to_emb:
                user_features.append([0.0] * smore_model.embedding_size)
            else:
                user_features.append(user_model_idx_to_emb[idx])
        log.info(f"shape of user_features {np.array(user_features).shape}")
        user_features = sparse.csr_matrix(np.array(user_features))

    model = LightFM(
        learning_rate=lightfm_configs["lr"],
        loss=lightfm_configs["loss"],
        no_components=lightfm_configs["emb_size"],
    )
    model.fit(
        lightfm_input.interaction_matrix,
        epochs=lightfm_configs["epoch"],
        num_threads=20,
        verbose=True,
        item_features=item_features,
        user_features=user_features,
    )

    item_bias, item_embeddings = model.get_item_representations(features=item_features)
    index_to_embedding: Dict[str, List[float]] = {}
    assert not item_embeddings is None
    for idx, emb in enumerate(item_embeddings):
        index_to_embedding[
            str(lightfm_input.item_model_idx_to_graph_idx[idx])
        ] = emb.tolist()

    log.info(
        f"Concatenating {len(index_to_embedding)} items' pcc and smore embeddings is completed"
    )

    model_output = Model(
        model_name=f"pcc_smore_{smore_model_name}",
        embedding_size=lightfm_configs["emb_size"],
        index_to_embedding=index_to_embedding,
    )
    return asdict(model_output)
