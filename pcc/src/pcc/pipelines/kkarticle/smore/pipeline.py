from kedro.pipeline import Pipeline, node

from pcc.pipelines.kkarticle.smore.nodes import convert_graph_format, split_graph


def create_pipeline(model_name, **kwargs):
    common_prequisite = [
        node(
            func=split_graph,
            inputs=["kkarticle_processed_interaction_graph", "params:split_point",],
            outputs=["kkarticle_training_graph", "kkarticle_testing_graph"],
            name="split_graph",
        ),
        node(
            func=convert_graph_format,
            inputs="kkarticle_training_graph",
            outputs="kkarticle_training_graph_smore",
            name="convert_graph_format",
        ),
    ]
    model_to_pipeline = {"hpe": Pipeline(common_prequisite)}
    return model_to_pipeline[model_name]
