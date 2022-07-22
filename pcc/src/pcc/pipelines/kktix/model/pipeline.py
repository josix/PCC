"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from pcc.pipelines.goodreads.model.nodes import (
    remove_unseen_items_from_interaction_graph,
    build_semantic_content_graph,
    export_smore_format,
    smore_train,
    aggregate_item_emb,
    lightfm_pcc_smore,
)

from pcc.pipelines.goodreads.model.pipeline import (
    pcc_model_wrapper,
    lightfm_pcc_smore_wrapper,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=remove_unseen_items_from_interaction_graph,
                inputs=[
                    "processed_kktix_interaction_graph",
                    "experiment_unseen_kktix_events",
                ],
                outputs="kktix_trainging_interaction_graph",
                name="remove_unseen_items_from_interaction_graph",
            ),
            node(
                func=build_semantic_content_graph,
                inputs=[
                    "kktix_trainging_interaction_graph",
                    "processed_kktix_content_graph",
                ],
                outputs="kktix_training_semantic_content_graph",
                name="build_semantic_content_graph",
            ),
            node(
                func=export_smore_format,
                inputs=[
                    "kktix_trainging_interaction_graph",
                    "processed_kktix_content_graph",
                    "kktix_training_semantic_content_graph",
                ],
                outputs=[
                    "kktix_smore_interaction_training_graph",
                    "kktix_smore_content_training_graph",
                    "kktix_smore_semantic_content_training_graph",
                ],
                name="export_smore_format",
            ),
            node(
                func=smore_train,
                inputs=[
                    "kktix_smore_interaction_training_graph",
                    "kktix_smore_content_training_graph",
                    "kktix_smore_semantic_content_training_graph",
                    "params:kktix_training_graph_configs",
                ],
                outputs=[
                    "kktix_smore_interaction_training_embedding",
                    "kktix_smore_content_training_embedding",
                    "kktix_smore_semantic_content_training_embedding",
                ],
                name="smore_train",
            ),
            # pcc
            node(
                func=pcc_model_wrapper(aggregate_item_emb),
                inputs=[
                    "kktix_smore_content_training_embedding",
                    "kktix_smore_semantic_content_training_embedding",
                    "processed_kktix_content_graph",
                    "params:kktix_aggregate_item_configs",
                ],
                outputs="kktix_pcc_item_embedding",
                name="aggregate_item_emb",
            ),
            # LightFM (MF)
            node(
                func=lightfm_pcc_smore_wrapper(
                    lightfm_pcc_smore, smore_model_name="mf"
                ),
                inputs=[
                    "kktix_pcc_item_embedding",
                    "kktix_smore_interaction_training_embedding",
                    "processed_kktix_interaction_graph",
                    "params:kktix_lightfm_configs",
                ],
                outputs="kktix_lightfm_pcc_smore_mf_training_embedding",
                name="lightfm_pcc_smore_mf",
            ),
            # LightFM (Line)
            node(
                func=lightfm_pcc_smore_wrapper(
                    lightfm_pcc_smore, smore_model_name="line"
                ),
                inputs=[
                    "kktix_pcc_item_embedding",
                    "kktix_smore_interaction_training_embedding",
                    "processed_kktix_interaction_graph",
                    "params:kktix_lightfm_configs",
                ],
                outputs="kktix_lightfm_pcc_smore_line_training_embedding",
                name="lightfm_pcc_smore_line",
            ),
            # LightFM (HPE)
            node(
                func=lightfm_pcc_smore_wrapper(
                    lightfm_pcc_smore, smore_model_name="hpe"
                ),
                inputs=[
                    "kktix_pcc_item_embedding",
                    "kktix_smore_interaction_training_embedding",
                    "processed_kktix_interaction_graph",
                    "params:kktix_lightfm_configs",
                ],
                outputs="kktix_lightfm_pcc_smore_hpe_training_embedding",
                name="lightfm_pcc_smore_hpe",
            ),
        ]
    )
