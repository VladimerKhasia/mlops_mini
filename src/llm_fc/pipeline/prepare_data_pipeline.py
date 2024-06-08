from llm_fc import config_manager, logger 
from llm_fc.components import prepare_data

STAGE_NAME = "Data Preparation Stage"


class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = config_manager.ConfigurationManager()
        data_config = config.get_data_config()
        data_preparator = prepare_data.PrepareData(config=data_config)
        data_preparator.prepare_dataset()
        # data_preparator.get_train_test_splits()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Pipeline = DataPreparationPipeline()
        Pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e