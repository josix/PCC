"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""
from typing import List

from kedro.pipeline import Pipeline, node
from kedro.pipeline.node import Node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    random_recommend,
    pcc_recommend,
    smore_content_model_recommend,
    tfidf_recommend,
)


def smore_rec_wrapper(model_name: str, func):
    def wrapper(*args, **kwargs):
        return func(model_name=model_name, *args, **kwargs)

    return wrapper


def rec_type_wrapper(should_rec_seen_items: bool, func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs, should_rec_seen_items=should_rec_seen_items)

    return wrapper


def i2i_rec_exp_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=rec_type_wrapper(True, random_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_random_recommend_result",
                name="random_recommend",
            ),
            # PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_recommend_result",
                name="pcc_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding_I",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_I_recommend_result",
                name="pcc_I_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding_W",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_W_recommend_result",
                name="pcc_W_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding_IW",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_IW_recommend_result",
                name="pcc_IW_recommend",
            ),
            # LightFM MF + PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_smore_mf_recommend_result",
                name="lightfm_pcc_smore_mf_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_I_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_I_smore_mf_recommend_result",
                name="lightfm_pcc_I_smore_mf_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_W_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_W_smore_mf_recommend_result",
                name="lightfm_pcc_W_smore_mf_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_IW_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_IW_smore_mf_recommend_result",
                name="lightfm_pcc_IW_smore_mf_recommend",
            ),
            # LightFM Line + PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_smore_line_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_smore_line_recommend_result",
                name="lightfm_pcc_smore_line_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_I_smore_line_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_I_smore_line_recommend_result",
                name="lightfm_pcc_I_smore_line_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_W_smore_line_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_W_smore_line_recommend_result",
                name="lightfm_pcc_W_smore_line_recommend",
            ),
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_IW_smore_line_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_IW_smore_line_recommend_result",
                name="lightfm_pcc_IW_smore_line_recommend",
            ),
            # Content based recommending
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("line", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_content_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_content_model_line_recommend_result",
                name="smore_content_model_recommend_line",
            ),
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("mf", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_content_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_content_model_mf_recommend_result",
                name="smore_content_model_recommend_mf",
            ),
            # CF based recommending
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("mf", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_interaction_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_interactoin_model_mf_recommend_result",
                name="smore_interaction_model_recommend_mf",
            ),
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("line", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_interaction_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_interactoin_model_line_recommend_result",
                name="smore_interaction_model_recommend_line",
            ),
            # Tf-Idf
            node(
                func=rec_type_wrapper(True, tfidf_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "processed_goodreads_comics_graphic_books",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_tfidf_recommend_result",
                name="tfidf_recommend",
            ),
        ]
    )


def ccs_exp_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=rec_type_wrapper(False, random_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_random_recommend_result",
                name="random_recommend",
            ),
            # PCC
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_recommend_result",
                name="pcc_recommend",
            ),
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding_I",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_I_recommend_result",
                name="pcc_I_recommend",
            ),
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding_W",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_W_recommend_result",
                name="pcc_W_recommend",
            ),
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_pcc_item_embedding_IW",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_pcc_IW_recommend_result",
                name="pcc_IW_recommend",
            ),
            # LightFM + PCC
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_smore_mf_recommend_result",
                name="lightfm_pcc_smore_mf_recommend",
            ),
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_I_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_I_smore_mf_recommend_result",
                name="lightfm_pcc_I_smore_mf_recommend",
            ),
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_W_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_W_smore_mf_recommend_result",
                name="lightfm_pcc_W_smore_mf_recommend",
            ),
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_lightfm_pcc_IW_smore_mf_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_lightfm_pcc_IW_smore_mf_recommend_result",
                name="lightfm_pcc_IW_smore_mf_recommend",
            ),
            # Content based recommending
            node(
                func=rec_type_wrapper(
                    False, smore_rec_wrapper("line", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_content_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_content_model_line_recommend_result",
                name="smore_content_model_recommend_line",
            ),
            node(
                func=rec_type_wrapper(
                    False, smore_rec_wrapper("mf", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_content_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_content_model_mf_recommend_result",
                name="smore_content_model_recommend_mf",
            ),
            # Tf-Idf
            node(
                func=rec_type_wrapper(False, tfidf_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "processed_goodreads_comics_graphic_books",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_tfidf_recommend_result",
                name="tfidf_recommend",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    if kwargs["ccs_exp"]:
        return ccs_exp_pipeline()
    return i2i_rec_exp_pipeline()
