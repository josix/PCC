"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""

from typing import List

from kedro.pipeline import Pipeline, node
from kedro.pipeline.node import Node
from kedro.pipeline.modular_pipeline import pipeline

from pcc.pipelines.goodreads.experiment.nodes import (
    random_recommend,
    pcc_recommend,
    smore_content_model_recommend,
)

from pcc.pipelines.goodreads.experiment.pipeline import (
    smore_rec_wrapper,
    rec_type_wrapper,
)

from .nodes import (
    tfidf_recommend,
)


def i2i_rec_exp_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=rec_type_wrapper(True, random_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_random_recommend_result",
                name="random_recommend",
            ),
            # PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_pcc_item_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_pcc_recommend_result",
                name="pcc_recommend",
            ),
            # LightFM MF + PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_lightfm_pcc_smore_mf_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_lightfm_pcc_smore_mf_recommend_result",
                name="lightfm_pcc_smore_mf_recommend",
            ),
            # LightFM Line + PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_lightfm_pcc_smore_line_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_lightfm_pcc_smore_line_recommend_result",
                name="lightfm_pcc_smore_line_recommend",
            ),
            # LightFM HPE + PCC
            node(
                func=rec_type_wrapper(True, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_lightfm_pcc_smore_hpe_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_lightfm_pcc_smore_hpe_recommend_result",
                name="lightfm_pcc_smore_hpe_recommend",
            ),
            # Content based recommending
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("line", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_content_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_content_model_line_recommend_result",
                name="smore_content_model_recommend_line",
            ),
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("mf", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_content_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_content_model_mf_recommend_result",
                name="smore_content_model_recommend_mf",
            ),
            # CF based recommending
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("mf", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_interaction_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_interactoin_model_mf_recommend_result",
                name="smore_interaction_model_recommend_mf",
            ),
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("line", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_interaction_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_interactoin_model_line_recommend_result",
                name="smore_interaction_model_recommend_line",
            ),
            node(
                func=rec_type_wrapper(
                    True, smore_rec_wrapper("hpe", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_interaction_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_interactoin_model_hpe_recommend_result",
                name="smore_interaction_model_recommend_hpe",
            ),
            # Tf-Idf
            node(
                func=rec_type_wrapper(True, tfidf_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "processed_kktix_events",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_tfidf_recommend_result",
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
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_random_ccs_recommend_result",
                name="random_ccs_recommend",
            ),
            # PCC
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_pcc_item_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_pcc_ccs_recommend_result",
                name="pcc_ccs_recommend",
            ),
            # LightFM MF + PCC
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_lightfm_pcc_smore_mf_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_lightfm_pcc_smore_mf_ccs_recommend_result",
                name="lightfm_pcc_smore_mf_ccs_recommend",
            ),
            # LightFM Line + PCC
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_lightfm_pcc_smore_line_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_lightfm_pcc_smore_line_ccs_recommend_result",
                name="lightfm_pcc_smore_line_ccs_recommend",
            ),
            # LightFM HPE + PCC
            node(
                func=rec_type_wrapper(False, pcc_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_lightfm_pcc_smore_hpe_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_lightfm_pcc_smore_hpe_ccs_recommend_result",
                name="lightfm_pcc_smore_hpe_ccs_recommend",
            ),
            # Content based recommending
            node(
                func=rec_type_wrapper(
                    False, smore_rec_wrapper("line", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_content_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_content_model_line_ccs_recommend_result",
                name="smore_content_model_ccs_recommend_line",
            ),
            node(
                func=rec_type_wrapper(
                    False, smore_rec_wrapper("mf", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_content_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_content_model_mf_ccs_recommend_result",
                name="smore_content_model_ccs_recommend_mf",
            ),
            node(
                func=rec_type_wrapper(
                    False, smore_rec_wrapper("hpe", smore_content_model_recommend)
                ),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "kktix_smore_content_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_smore_content_model_hpe_ccs_recommend_result",
                name="smore_content_model_ccs_recommend_hpe",
            ),
            # Tf-Idf
            node(
                func=rec_type_wrapper(False, tfidf_recommend),
                inputs=[
                    "experiment_unseen_kktix_events",
                    "kktix_experiment_user_profile",
                    "processed_kktix_events",
                    "params:kktix_rec_num",
                    "params:kktix_exp_user_num",
                ],
                outputs="kktix_exp_tfidf_ccs_recommend_result",
                name="tfidf_ccs_recommend",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    if kwargs["ccs_exp"]:
        return ccs_exp_pipeline()
    return i2i_rec_exp_pipeline()
