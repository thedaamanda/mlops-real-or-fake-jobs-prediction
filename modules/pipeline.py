"""
Author: Muhammad Theda Amanda
Usage: Create TFX Pipeline
"""
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline


def init_pipeline(
        pipeline_root: Text,
        pipeline_name: str,
        metadata_path: Text,
        components
) -> pipeline.Pipeline:
    """Initiate TFX pipeline

    Args:
        pipeline_root (Text): path to store pipeline artifacts
        pipeline_name (str): name of the pipeline
        metadata_path (Text): path to store metadata
        components (tuple): TFX pipeline components

    Returns:
        pipeline.Pipeline: TFX pipeline orchestration
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")

    beam_pipeline_args = [
        "--direct_running_mode=multi_processing",
        "----direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
        eam_pipeline_args=beam_pipeline_args,
    )
