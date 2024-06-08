import os
import torch
from llm_fc import config_manager, logger 
from llm_fc.components import finetuning

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

STAGE_NAME = "Fine-tuning and Evaluation Stage"


class FinetuneModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = config_manager.ConfigurationManager()
        finetuning_config = config.get_finetuning_config()
        finetune = finetuning.Finetuning(config=finetuning_config)
        finetune.get_train_test_splits()
        finetune.train()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Pipeline = FinetuneModelPipeline()
        Pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e