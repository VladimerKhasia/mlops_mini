import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from app.agents_tools import get_current_temperature, search_wikipedia
from app.agent_utils import *



model_path = service_params.service_model_path
tokenizer_path = service_params.service_tokenizer_path
device = service_params.service_device


llm = GemmaChatModel()
llm.model = AutoModelForCausalLM.from_pretrained(model_path,
                                            device_map="auto" if device!=torch.device('cpu') else None,
                                            #  torch_dtype=torch.bfloat16
                                            )
llm.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=True)
llm.stopping_criteria_list = get_stopping_criteria_list(llm.tokenizer)

tools = [get_current_temperature, search_wikipedia]  

chain = prompt | llm | ResponseParser | OpenAIFunctionsAgentOutputParser() 

conversational_agent = StatefulAgent(tools=tools, 
                                     chain=chain, 
                                     memory = Memory(), 
                                     max_symbols=20000)






