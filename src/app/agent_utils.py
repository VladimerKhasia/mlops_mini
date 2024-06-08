from transformers import (
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages.ai import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
#from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field
import re
import json
from typing import Any
from app.service_config_manager import ServiceConfigurationManager 


service_params = ServiceConfigurationManager().get_service_params()

class ListOfTokensStoppingCriteria(StoppingCriteria):
    """
    Class to define a stopping criterion based on a list of specific tokens.
    """
    def __init__(self, tokenizer, stop_tokens):
        self.tokenizer = tokenizer
        # Encode each stop token and store their IDs in a list
        self.stop_token_ids_list = [tokenizer.encode(stop_token, add_special_tokens=False) for stop_token in stop_tokens]

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last tokens generated match any of the stop token sequences
        for stop_token_ids in self.stop_token_ids_list:
            len_stop_tokens = len(stop_token_ids)
            if len(input_ids[0]) >= len_stop_tokens:
                if input_ids[0, -len_stop_tokens:].tolist() == stop_token_ids:
                    return True
        return False


get_stopping_criteria_list = lambda tokenizer: StoppingCriteriaList([
   ListOfTokensStoppingCriteria(tokenizer, ["<end_of_turn>"])])

class GemmaChatModel(BaseChatModel):
    """
    A custom chat model powered by Gemma from Hugging Face, designed to be informative, comprehensive, and engaging.
    See the custom model guide here: https://python.langchain.com/docs/modules/model_io/chat/custom_chat_model/
    """

    model_name: str = "gemma_chat_model"  # Replace with the actual Gemma model name
    task: str = "conversational"  # Task for the pipeline (conversational or summarization)
    #temperature = 0.0
    #n: int = 1500
    model : Any = None
    tokenizer : Any = None
    generation_config = GenerationConfig(
              max_new_tokens = service_params.service_max_new_tokens, 
              temperature=service_params.service_temperature,
              #top_p=service_params.service_top_p,
              top_k=service_params.service_top_k, 
              repetition_penalty=service_params.service_repetition_penalty, 
              do_sample=service_params.service_do_sample, 
              )
    stopping_criteria_list: Any = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Args:
            messages: The list of prompt messages.
            stop: Optional list of stop tokens.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments.

        Returns:
            A ChatResult object containing the generated response.
        """

        prompt = messages[-1].content #[: self.n]
        input_ids = self.tokenizer.encode(prompt,
                          return_tensors="pt",
                          add_special_tokens=False).to(self.model.device) ##self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(generation_config=self.generation_config,
                         input_ids=input_ids,
                         stopping_criteria=self.stopping_criteria_list,) #self.model.generate(**input_ids, max_new_tokens=self.n)  # , temperature=self.temperature
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)  #self.tokenizer.decode(outputs[0])
        #text = " ".join(text.split("\n"))

        start_index, end_index = text.find(""), text.rfind("")
        response = text[start_index+len(""):end_index].strip()

        message = AIMessage(content=response, additional_kwargs={}, response_metadata={"time_in_seconds": 3})
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of language model used: "gemma_chat_model".
        """
        return "gemma_chat_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of identifying parameters for LangChain callbacks.
        """
        return {"model_name": self.model_name, "task": self.task}


def ResponseParser(output):
  # Regular expression to find the last pair of <function_call> and </function_call>
  pattern = r'<function_call>(.*?)</function_call>'

  # Using re.findall to find all matches
  #string = output.return_values['output'] # if additionally using OpenAIFunctionsAgentOutputParser() #output.__dict__  dict_keys(['return_values', 'log', 'type'])
  string = output.content # if not using OpenAIFunctionsAgentOutputParser()
  matches = re.findall(pattern, string, re.DOTALL)

  # Extracting the last match
  if matches:
      last_match = matches[-1]
      #return json.loads(last_match.strip())
      #output.additional_kwargs = {'function_call' : json.loads(last_match.strip())}
      dictionary = json.loads(last_match.strip())
      dictionary["arguments"] = json.dumps(dictionary["arguments"]) # It is done for openAI specs
      output.additional_kwargs = {'function_call' : dictionary}
      #output.content = ''
      return output

class Memory(BaseModel):
  history: str = Field(default="")
  api_memory: List[dict] | list = Field(default=[])

class StatefulAgent:
  """
  """
  def __init__(self, tools, chain, memory, max_symbols):
    self.tools = tools
    # self.functions = [format_tool_to_openai_function(f) for f in self.tools]
    self.functions = [convert_to_openai_function(f) for f in self.tools]
    self.tool_map = {tool.name: tool for tool in self.tools}
    self.chain = chain
    self.memory = memory
    self.max_symbols = max_symbols

  def templater(self, input: str):
    input_template = f"""<bos><start_of_turn>system
    You are a helpful assistant with access to the following functions.
    Use them if required:
    <tool>
    {self.functions}
    </tool>

    To use these functions respond with:
    <function_call> {{"name": "function_name", "arguments": {{"arg_1": "value_1", "arg_2": "value_2", ...}}}} </function_call>

    Contains properties essential for the model to respond according to the tasks:
    <observation> {{"arg_1": "value_1", "arg_2": "value_2", "arg_3": "value_3", ...}} </observation>

    Edge cases you must handle:
    - If there are no functions that match the user request, you will respond politely that you cannot help.
    <end_of_turn>
    <start_of_turn>user
    {input}<end_of_turn>
    <start_of_turn>function_call
    """
    return input_template

  def splitter(self):
    latest_output = re.split(r'(<start_of_turn>user)', self.result.message_log[0].content)
    if len(latest_output) > 2:
        header = latest_output[0]
        output = ''.join(latest_output[1:])
    return (header, output)

  def run(self, user_input):

      input = self.memory.history + user_input
      self.result = self.chain.invoke({"input": self.templater(input)})
      model_output = self.tool_map[self.result.tool].run(self.result.tool_input)
      header, output = self.splitter()
      self.memory.history += output + " <start_of_turn>model\n  " + model_output + "<end_of_turn>\n"

      self.memory.api_memory.append({"user": user_input,
                                     "model": model_output,
                                     "tool": self.result.tool,
                                     "tool_input": self.result.tool_input
                                     })

      if len(self.memory.history) > self.max_symbols:
        #print("Maximal memory storage used")
        self.memory.history = ''
        del self.result
        return self.memory.api_memory

      return self.memory.api_memory
  
  def get_full_history_generator(self):
    """
    Yields elements of the chat history one by one (user message, model response).
    """
    for message in self.memory.api_memory:
      yield message


system_prompt = "You are a helpful assistant with access to the functions, which you use if required."
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
])

