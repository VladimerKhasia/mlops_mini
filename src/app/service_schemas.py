from typing import Literal   
from pydantic import BaseModel
from pathlib import Path


class ChatMessage(BaseModel):
    user: str
    model: str
    tool: str
    tool_input: dict

class ServiceParams(BaseModel):
    service_max_new_tokens: int
    service_temperature: float = Literal[0, 1]
    # service_top_p: float = Literal[0, 1]
    service_top_k: int
    service_repetition_penalty: float 
    service_do_sample: bool
    service_model_path: Path
    service_tokenizer_path: Path
    service_device: str 
