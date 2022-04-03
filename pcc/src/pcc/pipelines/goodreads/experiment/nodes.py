"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""
import random
from collections import defaultdict
from typing import Dict, Any, List, Set

import networkx as nx
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

from pcc.schemas.common import UserProfile, Model, OutputModels
from pcc.schemas.goodreads import Book


ANNOY_TREE_NUM = 10


def random_recommend(
    items: Dict[str, Any],
    user_profile: List[Dict[str, Any]],
    rec_num: int,
    user_num: int,
) -> Dict[str, float]:
    """Recommend items based on random selection"""
    unseen_items: List[str] = items["unseen_items"]
    average_precision: List[float] = []
    recommended_items: Set[str] = set()
    recall: List[float] = []
    for user in user_profile[:user_num]:
        profile = UserProfile(
            exp_history=user["exp_history"],
            actual_history=user["actual_history"],
            queries=user["queries"],
        )
        random.shuffle(unseen_items)
        rec_items = unseen_items[:rec_num]
        recommended_items = recommended_items | set(rec_items)
        match_num = len(set(rec_items) & set(profile.actual_history))
        average_precision.append(match_num / rec_num)
        recall.append(match_num / len(profile.actual_history))
    return {
        f"MAP@{rec_num}": sum(average_precision) / len(average_precision),
        f"Recall@{rec_num}": sum(recall) / len(recall),
        f"Coverage@{rec_num}": len(recommended_items) / (user_num * rec_num),
    }


def pcc_recommend(
    items: Dict[str, Any],
    user_profile: List[Dict[str, Any]],
    pcc_model: Dict[str, Any],
    content_graph: nx.Graph,
    rec_num: int,
    user_num: int,
):
    """Recommend items based on trained PCC item embeddings"""
    model = Model(
        model_name=pcc_model["model_name"],
        embedding_size=pcc_model["embedding_size"],
        index_to_embedding=pcc_model["index_to_embedding"],
    )
    unseen_items: List[str] = items["unseen_items"]
    trained_idx_to_nodeid: Dict[str, str] = {
        str(attrs["index"]): node_id
        for node_id, attrs in content_graph.nodes(data=True)
        if str(attrs["index"]) in model.index_to_embedding.keys()
    }
    trained_nodeid_to_idx = {value: key for key, value in trained_idx_to_nodeid.items()}
    candidates = set(trained_nodeid_to_idx.keys()) & set(unseen_items)
    annoy_index = AnnoyIndex(model.embedding_size, metric="angular")
    for candidate_id in candidates:
        annoy_index.add_item(
            int(trained_nodeid_to_idx[candidate_id]),
            model.index_to_embedding[trained_nodeid_to_idx[candidate_id]],
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
            query_item_emb = model.index_to_embedding[
                trained_nodeid_to_idx[query_item_id]
            ]
            nn_items = annoy_index.get_nns_by_vector(
                query_item_emb, rec_num, search_k=-1, include_distances=True
            )
            for item_idx, score in zip(nn_items[0], nn_items[1]):
                rec_item_to_score[trained_idx_to_nodeid[str(item_idx)]] += score
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


def smore_content_model_recommend(
    items: Dict[str, Any],
    user_profile: List[Dict[str, Any]],
    content_model: Dict[str, Any],
    content_graph: nx.Graph,
    rec_num: int,
    user_num: int,
    model_name: str,
):
    """Recommend items based on embeddings which trained on interaction graph"""
    model = [
        model for model in content_model["outputs"] if model["model_name"] == model_name
    ]
    if not model:
        raise ValueError("Non trained model")
    model = Model(
        model_name=model[0]["model_name"],
        embedding_size=model[0]["embedding_size"],
        index_to_embedding=model[0]["index_to_embedding"],
    )
    unseen_items: List[str] = items["unseen_items"]
    trained_idx_to_nodeid: Dict[str, str] = {
        str(attrs["index"]): node_id
        for node_id, attrs in content_graph.nodes(data=True)
        if str(attrs["index"]) in model.index_to_embedding.keys()
    }
    trained_nodeid_to_idx = {value: key for key, value in trained_idx_to_nodeid.items()}
    candidates = set(trained_nodeid_to_idx.keys()) & set(unseen_items)
    annoy_index = AnnoyIndex(model.embedding_size, metric="angular")
    for candidate_id in candidates:
        annoy_index.add_item(
            int(trained_nodeid_to_idx[candidate_id]),
            model.index_to_embedding[trained_nodeid_to_idx[candidate_id]],
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
            if query_item_id not in trained_nodeid_to_idx:
                continue
            query_item_emb = model.index_to_embedding[
                trained_nodeid_to_idx[query_item_id]
            ]
            nn_items = annoy_index.get_nns_by_vector(
                query_item_emb, rec_num, search_k=-1, include_distances=True
            )
            for item_idx, score in zip(nn_items[0], nn_items[1]):
                rec_item_to_score[trained_idx_to_nodeid[str(item_idx)]] += score
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


def tfidf_recommend(
    items: Dict[str, Any],
    user_profile: List[Dict[str, Any]],
    items_with_metadata: List[Dict[str, Any]],
    rec_num: int,
    user_num: int,
):
    """Recommend items based on embeddings which trained on interaction graph"""
    corpus: List[str] = []
    idx_to_bookid = {}
    bookid_to_idx = {}
    for idx, item in enumerate(items_with_metadata):
        book = Book(
            title=item["title"],
            title_without_series=item["title_without_series"],
            book_id=item["book_id"],
            description=item["description"],
            similar_books=item["similar_books"],
        )
        corpus.append(f"{book.title_without_series} {book.description}")
        idx_to_bookid[idx] = f"i-{book.book_id}"
        bookid_to_idx[f"i-{book.book_id}"] = idx
    vectorizer = TfidfVectorizer()
    document_term_matrix = vectorizer.fit_transform(corpus)
    dim = document_term_matrix.shape[1]
    unseen_items: List[str] = items["unseen_items"]
    candidates = set(bookid_to_idx.keys()) & set(unseen_items)
    annoy_index = AnnoyIndex(dim, metric="angular")
    for candidate_id in candidates:
        annoy_index.add_item(
            bookid_to_idx[candidate_id],
            document_term_matrix[bookid_to_idx[candidate_id]].toarray()[0],
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
                bookid_to_idx[query_item_id]
            ].toarray()[0]
            nn_items = annoy_index.get_nns_by_vector(
                query_item_emb, rec_num, search_k=-1, include_distances=True
            )
            for item_idx, score in zip(nn_items[0], nn_items[1]):
                rec_item_to_score[idx_to_bookid[item_idx]] += score
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
