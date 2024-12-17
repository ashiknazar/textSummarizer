from textSummarizer.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationPipeline
from textSummarizer.pipeline.sate_04_model_trainer_pipeline import ModelTrainerPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
STAGE_NAME= "Data Ingestion stage"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<<")
    data_ingestion= DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<< \n x=====x")
except Exception as e:
    logger.exception(e)
    raise e
    

STAGE_NAME= "Data Validation stage"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<<")
    data_validation= DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<< \n x=====x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME= "Data Transformation stage"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<<")
    data_transformation= DataTransformationPipeline()
    data_transformation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<< \n x=====x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME= "Model training stage"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<<")
    model_train= ModelTrainerPipeline()
    model_train.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<< \n x=====x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME= "Model Evaluation stage"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<<")
    model_evaluation= ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<< \n x=====x")
except Exception as e:
    logger.exception(e)
    raise e
