import os
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    GenerationConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    Trainer,
    DataCollatorForLanguageModeling
)

from datasets import DatasetDict, load_from_disk #, load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
# import bitsandbytes as bnb
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from llm_fc.pipeline.get_pretrained_pipeline import PretrainedModelPipeline
# from llm_fc.pipeline.prepare_data_pipeline import DataPreparationPipeline
from llm_fc import schemas #, logger
from settings import settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Finetuning:

    def __init__(self, config: schemas.FinetuneSchema):
        self.config = config
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(self.config.pretrained_model_path) ##PretrainedModelPipeline().main() if you add return statement in main
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        # if self.config.gradient_checkpointing:
        #     self.model.config.use_cache = False
        self.lora_config = LoraConfig(
                            r=self.config.r ,
                            lora_alpha=self.config.lora_alpha ,
                            lora_dropout=self.config.lora_dropout ,
                            target_modules=self.config.target_modules ,
                            layers_to_transform=self.config.layers_to_transform ,     
                            bias="none",
                            task_type="CAUSAL_LM",
                        )
        self.training_arguments = TrainingArguments(
            per_device_train_batch_size = self.config.per_device_train_batch_size ,
            per_device_eval_batch_size = self.config.per_device_eval_batch_size ,
            # gradient_accumulation_steps = self.config.gradient_accumulation_steps ,
            # gradient_checkpointing=self.config.gradient_checkpointing ,
            optim = self.config.optim ,
            learning_rate = self.config.learning_rate ,
            #lr_scheduler_type=self.config.lr_scheduler_type ,
            bf16=self.config.bf16 ,
            max_grad_norm = self.config.max_grad_norm ,
            warmup_ratio = self.config.warmup_ratio ,
            group_by_length=self.config.group_by_length ,
            weight_decay=self.config.weight_decay ,
            num_train_epochs=self.config.num_train_epochs ,

            output_dir= self.config.output_dir ,
            logging_dir= self.config.logging_dir ,

            save_total_limit=self.config.save_total_limit , 
            save_strategy=self.config.save_strategy ,                      
            eval_strategy=self.config.eval_strategy ,
            load_best_model_at_end=self.config.load_best_model_at_end ,

            ## hf wandb integration: https://docs.wandb.ai/guides/integrations/huggingface
            logging_steps=self.config.logging_steps , # how often to log to W&B
            run_name=f"function_calling_{ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') }",
            report_to="wandb" if settings.WB_TOKEN else None, 
        )
 

    def get_train_test_splits(self):
        #raw_dataset = load_dataset(str(self.config.training_data_path))  ##DataPreparationPipeline().main() if you add return statement in main
        raw_dataset = load_from_disk(str(self.config.training_data_path))
        def format_dataset(example):

            return example

        dataset = raw_dataset.map(format_dataset, batched=True)

        #TRAIN-TEST-SPLIT
        total = round(len(dataset) * self.config.total_use)
        train_end = round(total * self.config.train_use)
        test_end = total
        dataset = DatasetDict({
            # 'train': dataset.shuffle(seed=1024).select(range(98000)),              # for full data
            # 'test': dataset.shuffle(seed=1024).select(range(98000, 100187))
            'train': dataset.shuffle(seed=1024).select(range(train_end)),            # for slice of data for quick demo
            'test': dataset.shuffle(seed=1024).select(range(train_end, test_end))
        })
     
        self.train_dataset = dataset['train']
        self.test_dataset = dataset['test']
        #return (self.train_dataset, self.train_dataset)


    def train(self):
        """Training"""
        self.model = get_peft_model(self.model, self.lora_config, adapter_name="function_calling")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_arguments,
            #callbacks=[callback],       #if in TrainingArguments you used: save_strategy="no",
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            peft_config=self.lora_config,
            dataset_text_field= "Text", #"text",
            max_seq_length=self.config.max_seq_length,
            tokenizer=self.tokenizer,
            packing=True,
            data_collator=self.data_collator,
        )
        # #----------------------------- UNCOMMENT FOR TRAINING ---------------------------------------
        self.trainer.train()   

        # Save the adapter weights
        adapter_path = self.config.output_dir / Path("function_calling_adapter")
        self.model.save_pretrained(adapter_path, "function_calling") 

        # Merge adaptor with the model and save entire fine-tuned model
        self.model = self.model.merge_and_unload()                   
        self.model.save_pretrained(self.config.finetuned_model_path, "function_calling")
        # #--------------------------------------------------------------------------------------------

        # # #------IF YOUR PC IS NOT CAPABLE OF TRAINING THE MODEL YOU CAN STILL CONTINUE LEARNING-------
        # # # You can use free google colab for training the model and put that model in directory: colab_trained_model/model".
        # # # You can put directory "colab_trained_model/" in .gitignore and in .dvcignore files
        # # # comment these two lines out in case of actual training.
        # model = AutoModelForCausalLM.from_pretrained("colab_trained_model/model")
        # model.save_pretrained("artifacts/finetuned/model")
        # # # -------------------------------------------------------------------------------------------






