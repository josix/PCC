from kedro.pipeline import Pipeline, node

from pcc.pipelines.kkarticle.data_engineering.nodes import (
    build_graph,
    remove_sparse_nodes,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_graph,
                inputs="kkarticle_interactions",
                outputs="kkarticle_interaction_graph",
                name="build_graph",
            ),
            node(
                func=remove_sparse_nodes,
                inputs=["kkarticle_interaction_graph", "params:sparse_degee_threshold"],
                outputs="kkarticle_processed_interaction_graph",
                name="remove_sparse_nodes",
            ),
        ]
    )
