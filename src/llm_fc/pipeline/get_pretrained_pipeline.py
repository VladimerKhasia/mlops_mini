from llm_fc import config_manager, logger 
from llm_fc.components import get_pretrained

STAGE_NAME = "Pretrained Model Stage"


class PretrainedModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = config_manager.ConfigurationManager()
        pretrained_config = config.get_pretrained_config()
        pretrained = get_pretrained.GetPretrained(config=pretrained_config)
        pretrained.get_pretrained()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Pipeline = PretrainedModelPipeline()
        Pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e