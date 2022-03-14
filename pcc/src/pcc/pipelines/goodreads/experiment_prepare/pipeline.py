"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import select_unseen_items, generate_user_profile


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_unseen_items,
                inputs=[
                    "processed_goodreads_interaction_graph",
                    "params:unseen_items_ratio",
                ],
                outputs="experiment_unseen_goodreads_comics_graphic_books",
                name="select_unseen_items",
            ),
            node(
                func=generate_user_profile,
                inputs=[
                    "processed_goodreads_interaction_graph",
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "params:user_query_num",
                ],
                outputs="goodreads_experiment_user_profile",
                name="generate_user_profile",
            ),
        ],
    )
