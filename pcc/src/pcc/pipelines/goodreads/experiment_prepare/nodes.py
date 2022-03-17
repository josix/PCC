"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""
from dataclasses import dataclass
import logging
import random
from typing import Dict, List, Any

import networkx as nx

from pcc.schemas.common import ItemSeenStatus

log = logging.getLogger(__name__)
random.seed(2022)


def select_unseen_items(interaction_graph: nx.Graph, ratio: float) -> Dict[str, Any]:
    """
    Take out `ratio * len(items)` numbers of books as cold item, and return a object
    contains `unseen_items` and `seen_items`
    """
    items = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "I"
    ]
    random.shuffle(items)
    unseen_items = items[: int(len(items) * ratio)]
    log.info(
        f"Took {len(unseen_items)} items as unseen items from total {len(items)} books"
    )
    seen_items = list(set(items) - set(unseen_items))
    log.info(f"#Seen items {len(seen_items)}, #Unseen items: {len(unseen_items)}")
    return {"unseen_items": unseen_items, "seen_items": seen_items}


def generate_user_profile(
    interaction_graph: nx.Graph,
    item_seen_status: Dict[str, List[str]],
    user_query_num: int,
) -> List[Dict[str, Any]]:
    """
    Take out unseen items from user history and generate list of seen items and query list for each user
    """
    user_profile = []
    unseen_items = set(
        ItemSeenStatus(
            unseen_items=item_seen_status["unseen_items"],
            seen_items=item_seen_status["seen_items"],
        ).unseen_items
    )

    users = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "U"
    ]
    # check if unseen items in user histroy, take them out and pick queries from the remain items
    # if no unseen items in user history, skip this user
    for user in users:
        user_history = {item for item in interaction_graph[user]}
        if len(user_history & unseen_items) == 0:
            continue
        exp_history = list(user_history - unseen_items)
        if len(exp_history) < user_query_num:
            continue
        queries = random.choices(exp_history, k=user_query_num)
        user_profile.append(
            {
                "exp_history": exp_history,
                "actual_history": list(user_history),
                "queries": queries,
            }
        )
    log.info(f"#Experiment User (with {user_query_num} queries): {len(user_profile)}")
    return user_profile
