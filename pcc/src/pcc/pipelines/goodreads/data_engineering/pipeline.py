"""
This is a boilerplate pipeline 'goodreads'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    build_raw_interaction_graph,
    build_raw_content_graph,
    remove_sparse_nodes,
    process_book_metadata,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_raw_interaction_graph,
                inputs=[
                    "goodreads_comics_graphic_reviews",
                    "processed_goodreads_comics_graphic_books",
                    "params:interaction_use_fields",
                ],
                outputs="goodreads_comics_graphic_raw_interaction_graph",
                name="build_goodread_raw_interaction_graph",
            ),
            node(
                func=process_book_metadata,
                inputs=[
                    "goodreads_comics_graphic_books",
                    "params:content_use_fields",
                    "params:top_k_keywords",
                ],
                outputs="processed_goodreads_comics_graphic_books",
                name="process_goodread_raw_content_graph",
            ),
            node(
                func=build_raw_content_graph,
                inputs=[
                    "processed_goodreads_comics_graphic_books",
                ],
                outputs="goodreads_comics_graphic_raw_content_graph",
                name="build_goodread_raw_content_graph",
            ),
            node(
                func=remove_sparse_nodes,
                inputs=[
                    "goodreads_comics_graphic_raw_interaction_graph",
                    "goodreads_comics_graphic_raw_content_graph",
                    "params:sparse_degee_threshold",
                ],
                outputs=[
                    "processed_goodreads_interaction_graph",
                    "processed_goodreads_content_graph",
                ],
                name="remove_sparse_nodes",
            ),
        ]
    )
