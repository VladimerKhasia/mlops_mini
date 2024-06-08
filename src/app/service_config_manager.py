from pathlib import Path
from app import service_schemas
from llm_fc import utils, schemas


class ServiceConfigurationManager:
    def __init__(
        self,
        params_filepath = Path('params.yaml')
        ):
        self.params = schemas.ServiceSchema(**utils.read_yaml(params_filepath))

    
    def get_service_params(self) -> service_schemas.ServiceParams:
        params = self.params
        service_params = service_schemas.ServiceParams(
            service_max_new_tokens = params.service_max_new_tokens,
            service_temperature = params.service_temperature,
            # service_top_p = params.service_top_p,
            service_top_k = params.service_top_k,
            service_repetition_penalty = params.service_repetition_penalty,
            service_do_sample = params.service_do_sample,
            service_model_path = Path(params.service_model_path),
            service_tokenizer_path = Path(params.service_tokenizer_path),
            service_device = params.service_device,
        )

        return service_params