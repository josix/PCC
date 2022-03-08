"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    remove_unseen_items_from_interaction_graph,
    build_semantic_content_graph,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=remove_unseen_items_from_interaction_graph,
                inputs=[
                    "processed_goodreads_interaction_graph",
                    "experiment_unseen_goodreads_comics_graphic_books",
                ],
                outputs="trainging_interaction_graph",
                name="remove_unseen_items_from_interaction_graph",
            ),
            node(
                func=build_semantic_content_graph,
                inputs=[
                    "trainging_interaction_graph",
                    "processed_goodreads_content_graph",
                ],
                outputs="training_semantic_content_graph",
                name="build_semantic_content_graph",
            ),
        ]
    )
