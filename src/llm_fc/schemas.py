from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, confloat #, ConfigDict
#from pathlib import Path


#---------------------------------- here starts main configuration portion
class Root(BaseModel):
    root_dir: Path

class GetData(Root):
    # model_config = ConfigDict(strict=True) ## trict mode for all fields
    # root_dir: Path = Field(strict=True) # if you want to use strict mode on specific fields. 
    data_id: str 

class Pretrained(Root):
    id: str
    pretrained_model_path: Path
    pretrained_tokenizer_path: Path

class Finetune(Root):
    finetuned_model_path: Path

class ConfigSchema(BaseModel):
    artifacts_root: Path
    get_data: GetData
    pretrained: Pretrained
    finetune: Finetune

#---------------------------------- here starts parameter 
class ParamSchema(BaseModel):
    ##-------- TrainingArguments
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    # gradient_accumulation_steps:  int
    # gradient_checkpointing: bool = True

    optim: str
    learning_rate: float
    #lr_scheduler_type: str
    bf16: bool = True 
    max_grad_norm:  int
    warmup_ratio:  float
    group_by_length: bool = True
    weight_decay: float
    num_train_epochs: int
    output_dir: Path
    logging_dir: Path
    logging_steps: int
    save_total_limit: int  
    save_strategy: str                      
    eval_strategy: str
    load_best_model_at_end: bool = True
    ##-------- SFTTrainer
    dataset_text_field: str
    max_seq_length: int
    ## --------------for get_train_test_splits
    total_use: Optional[float] = confloat(gt=0, lt=1) #| None
    train_use: Optional[float] = confloat(gt=0, lt=1) #| None
    ## --------------for loraconfig
    r: int
    lora_alpha: int
    lora_dropout: float = confloat(gt=0, lt=1)
    target_modules: List[str]
    layers_to_transform: List[int]

#------------------------ Fine-tuning
class FinetuneSchema(ParamSchema):
    ##---------- from Finetune
    root_dir: Path
    finetuned_model_path: Path
    ##---------- for model and data
    pretrained_model_path: Path
    pretrained_tokenizer_path: Path
    training_data_path: Path
    
class ServiceSchema(ParamSchema):
    ## --------------for production service
    service_max_new_tokens: int
    service_temperature: float = Literal[0, 1]
    # service_top_p: float = Literal[0, 1]
    service_top_k: int
    service_repetition_penalty: float 
    service_do_sample: bool
    service_model_path: Path
    service_tokenizer_path: Path
    service_device: str 
