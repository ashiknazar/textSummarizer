from textSummarizer.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.logging import logger

STAGE_NAME= "Data Ingestion stage"
try:
    logger.info(f">>> stage {STAGE_NAME} started <<<<")
    data_ingestion= DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<< \n x=====x")
except Exception as e:
    logger.exception(e)
    raise e