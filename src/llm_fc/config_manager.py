from pathlib import Path
from llm_fc import utils, schemas 


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = Path('artifact_config.yaml'),
        params_filepath = Path('params.yaml')
        ):

        self.config = schemas.ConfigSchema(**utils.read_yaml(config_filepath))
        self.params = schemas.ParamSchema(**utils.read_yaml(params_filepath))

        utils.create_directories([self.config.artifacts_root])


    
    def get_data_config(self) -> schemas.GetData:
        config = self.config.get_data

        utils.create_directories([config.root_dir])

        data_config = schemas.GetData(
            data_id=config.data_id,
            root_dir=config.root_dir
        )

        return data_config


    def get_pretrained_config(self) -> schemas.Pretrained:
        config = self.config.pretrained

        utils.create_directories([config.root_dir])

        model_config = schemas.Pretrained(
            id=config.id,
            root_dir=config.root_dir,
            pretrained_model_path=config.pretrained_model_path,
            pretrained_tokenizer_path=config.pretrained_tokenizer_path
        )

        return model_config 


    def get_finetuning_config(self) -> schemas.FinetuneSchema:
        config = self.config.finetune
        params = self.params
        model_config = self.get_pretrained_config()
        data_config = self.get_data_config() 
        
        utils.create_directories([config.root_dir])

        finetune_config = schemas.FinetuneSchema(
            ##---------- from Finetune
            root_dir=config.root_dir,
            finetuned_model_path=config.finetuned_model_path,

            ##---------- from FinetuneSchema specifics to inject model and data
            pretrained_model_path = model_config.pretrained_model_path,
            pretrained_tokenizer_path = model_config.pretrained_tokenizer_path,
            training_data_path = data_config.root_dir,

            ##---------- from what is shared between ParamSchema and FinetuneSchema 
            per_device_train_batch_size = params.per_device_train_batch_size ,
            per_device_eval_batch_size = params.per_device_eval_batch_size ,
            # gradient_accumulation_steps = params.gradient_accumulation_steps ,
            # gradient_checkpointing=params.gradient_checkpointing,
            optim = params.optim,
            learning_rate = params.learning_rate,
            #lr_scheduler_type=params.lr_scheduler_type,
            bf16=params.bf16 ,
            max_grad_norm = params.max_grad_norm ,
            warmup_ratio = params.warmup_ratio ,
            group_by_length= params.group_by_length ,
            weight_decay=params.weight_decay ,
            num_train_epochs=params.num_train_epochs ,

            output_dir= Path(config.root_dir) / Path(params.output_dir) ,
            logging_dir= Path(config.root_dir) / Path(params.logging_dir) ,
            
            logging_steps=params.logging_steps ,
            save_total_limit=params.save_total_limit ,
            save_strategy=params.save_strategy ,                     
            eval_strategy=params.eval_strategy ,
            load_best_model_at_end=params.load_best_model_at_end ,
            dataset_text_field= params.dataset_text_field ,
            max_seq_length= params.max_seq_length,
            ##------- for get_train_test_split
            total_use= params.total_use,
            train_use= params.train_use,
            ##------- for loraconfig
            r=params.r ,
            lora_alpha=params.lora_alpha ,
            lora_dropout=params.lora_dropout ,
            target_modules=params.target_modules ,
            layers_to_transform=params.layers_to_transform ,
            
        )

        return finetune_config 


