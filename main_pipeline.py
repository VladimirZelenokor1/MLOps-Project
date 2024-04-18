import json
import logging
import sys
import argparse
from datetime import datetime

import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from src.features.build_transformer import (
    build_transformer,
    build_ctr_transformer,
    extract_target,
    process_count_features,
)

from src.models.model_fit_predict import  (
    train_model,
    predict_model,
    eval_model,
    serialize_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def train_pipeline(config_path: str):
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config_path
    )

    data: pd.DataFrame = read_data(training_pipeline_params.input_data_path)
    data["hour"] = data.hour.apply(lambda val: datetime.strptime(str(val), "%y%m%d%H"))

    logger.debug(f"Start training pipeline with params: {training_pipeline_params}")
    logger.debug(f"data: {data.shape} \n {data.info()} \n {data.nunique()}")

    transformer = build_transformer()
    processed_data = process_count_features(
        transformer, data, training_pipeline_params.feature_params
    )
    logger.debug(f"processed_data: {processed_data.shape} \n {processed_data.info()}"
                 f"\n {processed_data.nunique()} \n "
                 f"{processed_data[training_pipeline_params.feature_params.count_features]}"
                 )

    train_df, val_df = split_train_val_data(
        processed_data, training_pipeline_params.splitting_params
    )
    logger.debug(f"train_df.shape is {train_df.shape}")
    logger.debug(f"val_df.shape is {val_df.shape}")

    # Check distribution of target between train and test
    logger.info(f"train trg: \n {train_df['click'].value_counts() / train_df.shape[0]}")
    logger.info(f"test trg: \n {val_df['click'].value_counts() / val_df.shape[0]}")

    ctr_transformer = build_ctr_transformer(training_pipeline_params.feature_params)
    ctr_transformer.fit(train_df)
    logger.debug(f"mean_ctr: {ctr_transformer.mean_ctr}")

    # Prepare train features
    train_features = ctr_transformer.transform(train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.debug(
        f"train_features: {train_features.shape} \n {train_features.info()} \n {train_features.nunique()}"
    )

    # Prepare val features
    val_features = ctr_transformer.transform(val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    logger.debug(
        f"val_features: {val_features.shape} \n {val_features.info()} \n {val_features.nunique()}"
    )

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    predict_proba, preds = predict_model(model, val_features)
    metrics = eval_model(predict_proba, preds, val_target)
    logger.debug(f"preds/targets shapes: {(preds.shape, val_target.shape)}")

    # Dump metrics to json
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metric is {metrics}")

    # Serialize model
    serialize_model(model, training_pipeline_params.output_model_path)
    serialize_model(
        ctr_transformer, training_pipeline_params.output_ctr_transformer_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)
