"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""
import logging
from collections import defaultdict
from typing import Dict, Any, List, Set

from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

from pcc.schemas.common import UserProfile
from pcc.schemas.kktix import Event
from pcc.pipelines.goodreads.experiment.nodes import get_candidates


ANNOY_TREE_NUM = 10
log = logging.getLogger(__name__)


def tfidf_recommend(
    items: Dict[str, Any],
    user_profile: List[Dict[str, Any]],
    items_with_metadata: List[Dict[str, Any]],
    rec_num: int,
    user_num: int,
    should_rec_seen_items: bool = False,
):
    """Recommend items based on embeddings which trained on interaction graph"""
    corpus: List[str] = []
    idx_to_eventid = {}
    eventid_to_idx = {}
    for idx, item in enumerate(items_with_metadata):
        event = Event(**item)
        corpus.append(f"{event.event_name} {event.event_description}")
        idx_to_eventid[idx] = f"i-{event.event_id}"
        eventid_to_idx[f"i-{event.event_id}"] = idx
    vectorizer = TfidfVectorizer(max_features=1024)
    document_term_matrix = vectorizer.fit_transform(corpus)
    dim = document_term_matrix.shape[1]
    candidates = get_candidates(items, should_rec_seen_items)
    candidates = set(eventid_to_idx.keys()) & set(candidates)
    annoy_index = AnnoyIndex(dim, metric="angular")
    for candidate_id in candidates:
        annoy_index.add_item(
            eventid_to_idx[candidate_id],
            document_term_matrix[eventid_to_idx[candidate_id]].toarray()[0],
        )
    annoy_index.build(ANNOY_TREE_NUM)
    average_precision: List[float] = []
    recommended_items: Set[str] = set()
    recall: List[float] = []
    for user in user_profile[:user_num]:
        profile = UserProfile(
            exp_history=user["exp_history"],
            actual_history=user["actual_history"],
            queries=user["queries"],
        )
        rec_item_to_score = defaultdict(float)
        for query_item_id in profile.queries:
            query_item_emb = document_term_matrix[
                eventid_to_idx[query_item_id]
            ].toarray()[0]
            nn_items = annoy_index.get_nns_by_vector(
                query_item_emb, rec_num, search_k=-1, include_distances=True
            )
            for item_idx, score in zip(nn_items[0], nn_items[1]):
                rec_item_to_score[idx_to_eventid[item_idx]] += score
        ranked_rec_result = sorted(
            [(score, item_id) for item_id, score in rec_item_to_score.items()],
            reverse=True,
        )[:rec_num]
        rec_items = [item_id for _, item_id in ranked_rec_result]
        recommended_items = recommended_items | set(rec_items)
        match_num = len(set(rec_items) & set(profile.actual_history))
        average_precision.append(match_num / rec_num)
        recall.append(match_num / len(profile.actual_history))
    return {
        f"MAP@{rec_num}": sum(average_precision) / len(average_precision),
        f"Recall@{rec_num}": sum(recall) / len(recall),
        f"Coverage@{rec_num}": len(recommended_items) / (user_num * rec_num),
    }
