"""
Helper functions for dealing with external lightfm training stuffs.
"""
import logging
import dataclasses
from typing import Dict

import networkx as nx
from scipy.sparse import csr_array

log = logging.getLogger(__name__)


@dataclasses.dataclass
class LightFMInteraction:
    """
    Interaction class that is compatible with LightFM training format, and it suports
    the index conversion between given networkx graph and LightFM interaction matrix.
    """

    user_graph_idx_to_model_idx: Dict[int, int]
    user_model_idx_to_graaph_idx: Dict[int, int]
    item_graph_idx_to_model_idx: Dict[int, int]
    item_model_idx_to_graph_idx: Dict[int, int]
    item_id_to_model_idx: Dict[str, int]
    interaction_matrix: csr_array


def reverse_mapping(mapping: Dict) -> Dict:
    """
    Reversing mapping by making keys as value and vice versa.
    """
    return {value: key for key, value in mapping.items()}


def get_lightfm_input(interaction_graph: nx.Graph) -> LightFMInteraction:
    """
    Converting Networkx Interaction to CSR Matrix which is supported format by LightFM
    """
    users = [
        (n, attrs)
        for n, attrs in interaction_graph.nodes(data=True)
        if attrs["type"] == "U"
    ]
    user_graph_idx_to_model_idx: Dict[int, int] = {
        attrs["index"]: user_idx for user_idx, (n, attrs) in enumerate(users)
    }
    items = [
        (n, attrs)
        for n, attrs in interaction_graph.nodes(data=True)
        if attrs["type"] == "I"
    ]
    item_graph_idx_to_model_idx: Dict[int, int] = {
        attrs["index"]: item_idx for item_idx, (n, attrs) in enumerate(items)
    }
    item_id_to_model_idx: Dict[str, int] = {
        item[0]: idx for idx, item in enumerate(items)
    }
    row = []
    col = []
    for user_idx, (user, _) in enumerate(users):
        for item in interaction_graph[user]:
            row.append(user_idx)
            col.append(item_id_to_model_idx[item])
    data = [1] * len(row)
    interaction_matrix = csr_array((data, (row, col)), shape=(len(users), len(items)))
    log.info(
        f"Shape of training interaction {interaction_matrix.shape}",
    )
    return LightFMInteraction(
        user_graph_idx_to_model_idx=user_graph_idx_to_model_idx,
        user_model_idx_to_graaph_idx=reverse_mapping(user_graph_idx_to_model_idx),
        item_graph_idx_to_model_idx=item_graph_idx_to_model_idx,
        item_model_idx_to_graph_idx=reverse_mapping(item_graph_idx_to_model_idx),
        item_id_to_model_idx=item_id_to_model_idx,
        interaction_matrix=interaction_matrix,
    )
