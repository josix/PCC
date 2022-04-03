"""
This is a boilerplate pipeline 'experiment'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
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


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=random_recommend,
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_exp_random_recommend_result",
                name="random_recommend",
            ),
            node(
                func=pcc_recommend,
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
                func=smore_rec_wrapper("line", smore_content_model_recommend),
                inputs=[
                    "experiment_unseen_goodreads_comics_graphic_books",
                    "goodreads_experiment_user_profile",
                    "goodreads_smore_content_training_embedding",
                    "processed_goodreads_content_graph",
                    "params:rec_num",
                    "params:exp_user_num",
                ],
                outputs="goodreads_smore_content_model_recommend_result",
                name="smore_content_model_recommend_line",
            ),
            node(
                func=tfidf_recommend,
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
