"""
This is a boilerplate pipeline 'experiment_prepare'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import select_unseen_items
from pcc.pipelines.goodreads.experiment_prepare.nodes import generate_user_profile


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_unseen_items,
                inputs=[
                    "processed_kktix_interaction_graph",
                    "processed_kktix_events",
                    "params:split_date",
                ],
                outputs="experiment_unseen_kktix_events",
                name="select_unseen_items",
            ),
            node(
                func=generate_user_profile,
                inputs=[
                    "processed_kktix_interaction_graph",
                    "experiment_unseen_kktix_events",
                    "params:kktix_user_query_num",
                ],
                outputs="kktix_experiment_user_profile",
                name="generate_user_profile",
            ),
        ]
    )
