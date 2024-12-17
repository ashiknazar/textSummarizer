from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.logging import logger
from textSummarizer.components.model_trainer import ModelTrainer


class ModelTrainerPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        modeltrainer = ModelTrainer(config=model_trainer_config)
        modeltrainer.train()


