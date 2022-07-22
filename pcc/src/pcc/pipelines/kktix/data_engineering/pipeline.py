"""
This is a boilerplate pipeline 'kktix'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    process_event_metadata,
    build_raw_interaction_graph,
    build_raw_content_graph,
)

from pcc.pipelines.goodreads.data_engineering.nodes import remove_sparse_nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_raw_interaction_graph,
                inputs=[
                    "kktix_interactions",
                    "processed_kktix_events",
                    "params:kktix_interaction_use_fields",
                ],
                outputs="kktix_raw_interaction_graph",
                name="build_kktix_raw_interaction_graph",
            ),
            node(
                func=process_event_metadata,
                inputs=[
                    "kktix_meta",
                    "params:kktix_content_use_fields",
                    "params:kktix_top_k_keywords",
                ],
                outputs="processed_kktix_events",
                name="process_kktix_raw_content_graph",
            ),
            node(
                func=build_raw_content_graph,
                inputs=[
                    "processed_kktix_events",
                ],
                outputs="kktix_raw_content_graph",
                name="build_kktix_raw_content_graph",
            ),
            node(
                func=remove_sparse_nodes,
                inputs=[
                    "kktix_raw_interaction_graph",
                    "kktix_raw_content_graph",
                    "params:kktix_sparse_degee_threshold",
                ],
                outputs=[
                    "processed_kktix_interaction_graph",
                    "processed_kktix_content_graph",
                ],
                name="remove_sparse_nodes",
            ),
        ]
    )
