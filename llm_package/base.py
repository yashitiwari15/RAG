# llm_package/base.py

from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self, model_name="gpt-4o-mini", max_tokens=1000, temperature=0.5):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def find_answer_gpt(self, query, context_metadata, previous_conversation):
        pass
