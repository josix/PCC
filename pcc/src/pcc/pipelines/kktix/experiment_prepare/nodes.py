"""
This is a boilerplate pipeline 'experiment_prepare'
generated using Kedro 0.17.7
"""

from datetime import datetime
import logging
import random
from typing import Dict, Any

import networkx as nx

from pcc.schemas.kktix import Event

log = logging.getLogger(__name__)
random.seed(2022)


def select_unseen_items(
    interaction_graph: nx.Graph, item_profiles: Dict[str, Any], split_date: str
) -> Dict[str, Any]:
    """
    Take out items that published after `split_date` as unseen items and
    the before ones as seen items for users
    """
    cut_date = datetime.fromisoformat(split_date)
    future_items = set()
    for item in item_profiles:
        event = Event(**item)
        date = datetime.fromtimestamp(event.first_publish_at)
        if date >= cut_date:
            future_items.add(f"i-{event.event_id}")

    items = [
        n for n, attrs in interaction_graph.nodes(data=True) if attrs["type"] == "I"
    ]
    unseen_items = [item for item in items if item in future_items]

    log.info(
        f"Took {len(unseen_items)} items as unseen items from total {len(items)} events"
    )
    seen_items = list(set(items) - set(unseen_items))
    log.info(f"#Seen items {len(seen_items)}, #Unseen items: {len(unseen_items)}")
    return {"unseen_items": unseen_items, "seen_items": seen_items}
