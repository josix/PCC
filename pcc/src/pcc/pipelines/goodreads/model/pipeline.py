"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    remove_unseen_items_from_interaction_graph,
    build_semantic_content_graph,
    export_smore_format,
    smore_train,
    aggregate_item_emb,
    lightfm_pcc_smore,
)


def lightfm_pcc_smore_wrapper(func, smore_model_name: str = "mf"):
    def wrapper(*args, **kwargs):
        return func(
            *args,
            **kwargs,
            smore_model_name=smore_model_name,
        )

    return wrapper


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=remove_unseen_items_from_interaction_graph,
                inputs=[
                    "processed_goodreads_interaction_graph",
                    "experiment_unseen_goodreads_comics_graphic_books",
                ],
                outputs="goodreads_trainging_interaction_graph",
                name="remove_unseen_items_from_interaction_graph",
            ),
            node(
                func=build_semantic_content_graph,
                inputs=[
                    "goodreads_trainging_interaction_graph",
                    "processed_goodreads_content_graph",
                ],
                outputs="goodreads_training_semantic_content_graph",
                name="build_semantic_content_graph",
            ),
            node(
                func=export_smore_format,
                inputs=[
                    "goodreads_trainging_interaction_graph",
                    "processed_goodreads_content_graph",
                    "goodreads_training_semantic_content_graph",
                ],
                outputs=[
                    "goodreads_smore_interaction_training_graph",
                    "goodreads_smore_content_training_graph",
                    "goodreads_smore_semantic_content_training_graph",
                ],
                name="export_smore_format",
            ),
            node(
                func=smore_train,
                inputs=[
                    "goodreads_smore_interaction_training_graph",
                    "goodreads_smore_content_training_graph",
                    "goodreads_smore_semantic_content_training_graph",
                    "params:training_graph_configs",
                ],
                outputs=[
                    "goodreads_smore_interaction_training_embedding",
                    "goodreads_smore_content_training_embedding",
                    "goodreads_smore_semantic_content_training_embedding",
                ],
                name="smore_train",
            ),
            node(
                func=aggregate_item_emb,
                inputs=[
                    "goodreads_smore_content_training_embedding",
                    "goodreads_smore_semantic_content_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:aggregate_item_configs",
                ],
                outputs="goodreads_pcc_item_embedding",
                name="aggregate_item_emb",
            ),
            node(
                func=lightfm_pcc_smore_wrapper(
                    lightfm_pcc_smore, smore_model_name="mf"
                ),
                inputs=[
                    "goodreads_pcc_item_embedding",
                    "goodreads_smore_interaction_training_embedding",
                    "processed_goodreads_interaction_graph",
                    "params:lightfm_configs",
                ],
                outputs="goodreads_lightfm_pcc_smore_mf_training_embedding",
                name="lightfm_pcc_smore_mf",
            ),
        ]
    )
