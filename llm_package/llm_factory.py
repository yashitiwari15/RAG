# llm_package/llm_factory.py

import os
from .azure_llm import AzureLLM
from .openai_llm import OpenAILLM
# from dotenv import load_dotenv

class LLMFactory:
    @staticmethod
    def get_llm_client():
        # load_dotenv(".env")
        client_type = os.getenv("CLIENT_TYPE")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        temperature = float(os.getenv("TEMPERATURE", 0.5))

        if client_type == "azure":
            return AzureLLM(
                api_key=os.getenv("AZURE_API_KEY"),
                endpoint=os.getenv("AZURE_ENDPOINT"),
                model_name=model_name,
                api_version=os.getenv("API_VERSION"),
                temperature=temperature
            )
        elif client_type == "openai":
            return OpenAILLM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=model_name,
                temperature=temperature
            )
        else:
            raise ValueError("Unsupported client type. Please set CLIENT_TYPE in the environment variables.")
