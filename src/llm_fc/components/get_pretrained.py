import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" #Just to avoid GPU access related unnecessary warning when using CPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_fc import logger, schemas
from settings import settings


class GetPretrained:
    def __init__(self, config: schemas.Pretrained):
        self.config = config
    
    def get_pretrained_model_tokenizer(self):
        # self.bnb_config = BitsAndBytesConfig(
        #                     load_in_4bit=True,
        #                     bnb_4bit_use_double_quant=True,
        #                     bnb_4bit_quant_type="nf4",
        #                     bnb_4bit_compute_dtype=torch.bfloat16
        #                     )
        self.model = AutoModelForCausalLM.from_pretrained(
                        self.config.id,
                        ##quantization_config=self.bnb_config,
                        ##attn_implementation="flash_attention_2",
                        #device_map="auto",           # Automatically distribute the model across available devices NO NEED ON CPU
                        torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model parameters - good for most devices including TPU
                        token=settings.HF_TOKEN
                    ) 
        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.config.id,
                            padding_side = 'right',
                            max_length=2099,  # Set the maximum length for tokenization
                            add_eos_token=True,
                            token=settings.HF_TOKEN
                        )
        self.model.save_pretrained(self.config.pretrained_model_path)
        self.tokenizer.save_pretrained(self.config.pretrained_tokenizer_path)

        #return (self.model, self.tokenizer)


    def get_pretrained(self):
        if not any(self.config.root_dir.iterdir()):
            self.get_pretrained_model_tokenizer()
        #     model, tokenizer = self.get_pretrained_model_tokenizer()
        # else:
        #     model = AutoModelForCausalLM.from_pretrained(self.config.pretrained_model_path)
        #     tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        # return (model, tokenizer)






