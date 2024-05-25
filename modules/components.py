"""
Author: Muhammad Theda Amanda
Usage: Create TFX Pipeline Components
"""
import os
import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def init_components(args):
    """Initiate tfx pipeline components

    Args:
        args (dict): args that containts some dependencies

    Returns:
        tuple: TFX pipeline components
    """
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
        ])
    )

    # ExampleGen
    example_gen = CsvExampleGen(
        input_base=args['data_dir'],
        output_config=output)

    # StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    # ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # Transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(args['transform_module']),
    )

    # Tuner
    tuner = Tuner(
        module_file=os.path.abspath(args['tuner_module']),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=20),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=5),
    )

    # Trainer
    trainer = Trainer(
        module_file=os.path.abspath(args['trainer_module']),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']),
    )

    # Model Resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id("latest_blessed_model_resolver")

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='fraudulent')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001}
                        ),
                    ),
                ),
            ])
        ],
    )

    # Evaluator
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config
    )

    # Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=args['serving_directory']
            )
        ),
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
