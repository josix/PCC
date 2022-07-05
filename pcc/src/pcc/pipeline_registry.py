"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from pcc.pipelines.goodreads.data_engineering import (
    pipeline as goodread_data_engineering_pipeline,
)
from pcc.pipelines.goodreads.experiment_prepare import (
    pipeline as goodread_experiment_prepare_pipeline,
)
from pcc.pipelines.goodreads.model import (
    pipeline as goodread_model_training_pipeline,
)
from pcc.pipelines.goodreads.experiment import (
    pipeline as goodread_model_experiment_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "__default__": goodread_data_engineering_pipeline.create_pipeline()
        + goodread_experiment_prepare_pipeline.create_pipeline()
        + goodread_model_training_pipeline.create_pipeline()
        + goodread_model_experiment_pipeline.i2i_rec_exp_pipeline(),
        "goodread_comics_graphic_DE": goodread_data_engineering_pipeline.create_pipeline(),
        "goodread_comics_graphic_experiment_prepare": goodread_experiment_prepare_pipeline.create_pipeline(),
        "goodread_comics_graphic_model_training": goodread_model_training_pipeline.create_pipeline(),
        "goodread_comics_graphic_experiment": goodread_model_experiment_pipeline.i2i_rec_exp_pipeline(),  # i2i for seen item rec exp
        "goodread_comics_graphic_ccs_exp": goodread_model_experiment_pipeline.ccs_exp_pipeline(),  # complete cold start i2i rec exp
    }
