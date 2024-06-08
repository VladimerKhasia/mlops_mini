import os
from llm_fc import logger, schemas
from datasets import load_dataset, DatasetDict
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class PrepareData:
    def __init__(self, config: schemas.GetData):
        self.config = config

    def prepare_dataset(self, split="train"):
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading raw data from {self.config.data_id}")
            raw_dataset = load_dataset(self.config.data_id, split=split)
            logger.info(f"Raw data was successfully downloaded")
        except Exception as e:
            logger.error(f"Error downloading raw data: {e}")
            raise e
        #------------------------------ Comment out following three lines if you uncoment the rest of the code
        if not any(self.config.root_dir.iterdir()):
            raw_dataset.save_to_disk(str(self.config.root_dir))
            logger.info(f"Data was successfully processed and saved.")
        #------------------------------ Make sure to uncomment some code in prepare_data_pipeline.py as well
        
    #     def format_dataset(example):

    #         return example

    #     dataset = raw_dataset.map(format_dataset, batched=True)

    #     #TRAIN-TEST-SPLIT
    #     dataset = DatasetDict({
    #         # 'train': dataset.shuffle(seed=1024).select(range(98000)),         # for full data
    #         # 'test': dataset.shuffle(seed=1024).select(range(98000, 100187))
    #         'train': dataset.shuffle(seed=1024).select(range(1000)),            # for slice of data for quick demo
    #         'test': dataset.shuffle(seed=1024).select(range(2000, 2100))
    #     })
        
    #     dataset.save_to_disk(str(self.config.root_dir))
    #     logger.info(f"Data was successfully processed and saved.")
    #     return dataset
    
    # def get_train_test_splits(self):
    #     """Get batched train and test splits"""
    #     if not any(self.config.root_dir.iterdir()):
    #         dataset = self.prepare_dataset(split="train")
    #     else:
    #         dataset = load_dataset(str(self.config.root_dir))    
    #     train_dataset = dataset['train']
    #     test_dataset = dataset['test']

    #     return (train_dataset, test_dataset)




