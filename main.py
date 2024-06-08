## if you have not installed: pip install -e /path/to/locations/repo  or pip install -e .
## than you have to use: from src.llm_fc import logger: https://stackoverflow.com/questions/42609943/what-is-the-use-case-for-pip-install-e
## + "from llm_fc import logger" is possible because we have logging code in __init__.py

import wandb                    # https://wandb.ai/lavanyashukla/visualize-predictions/reports/Dashboard-Track-and-compare-experiments-visualize-results--VmlldzoyMTI4NjY
from settings import settings   # https://docs.wandb.ai/guides/integrations/huggingface
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from llm_fc import logger
from llm_fc.pipeline.prepare_data_pipeline import DataPreparationPipeline, STAGE_NAME as STAGE_NAME_1
from llm_fc.pipeline.get_pretrained_pipeline import PretrainedModelPipeline, STAGE_NAME as STAGE_NAME_2
from llm_fc.pipeline.finetuning_pipeline import FinetuneModelPipeline, STAGE_NAME as STAGE_NAME_3


#-------------------------- 1. Prepare Data 
try:
    logger.info(f">>>>>> stage {STAGE_NAME_1} started <<<<<<")
    Pipeline = DataPreparationPipeline()
    Pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME_1} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

#-------------------------- 2. Get Pretrained Model
try:
    logger.info(f">>>>>> stage {STAGE_NAME_2} started <<<<<<")
    Pipeline = PretrainedModelPipeline()
    Pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME_2} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

#-------------------------- 3. Fine-tune and Evaluate Pretrained Model
try:
    logger.info(f">>>>>> stage {STAGE_NAME_3} started <<<<<<")
    wandb.login(key=settings.WB_TOKEN)  #wandb.login(relogin=True)
    wandb.init(project="function calling finetuning & Evaluation") #, mode='offline' in case no server reporting wanted
    Pipeline = FinetuneModelPipeline()
    Pipeline.main()
    wandb.finish()
    logger.info(f">>>>>> stage {STAGE_NAME_3} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


