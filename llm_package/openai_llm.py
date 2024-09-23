# # llm_package/openai_llm.py

# from .base import LLM
# import json

# class OpenAILLM(LLM):
#     def __init__(self, api_key, model_name, max_tokens=1000, temperature=0.5): #model_name="gpt-4"
#         super().__init__(model_name=model_name, max_tokens = max_tokens, temperature=temperature)
#         from openai import OpenAI
#         self.client = OpenAI(api_key=api_key)

#     def find_answer_gpt(self, query, context_metadata, previous_conversation):
#         context_info = json.dumps(context_metadata, indent=2)
#         msgs = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", 
#                 "content": f"""Please answer the following question by returning only a JSON response. 
#                 Make sure that the response does not contain any additional markers such as '''json or '''.
#                 In Highlight array, return only those relevant sections of context that has 'summary_text' as null and are used to form answer.
#                 The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:
#         {{
#         "highlight": [
#             {{
#             "text": "[Return orignal_text shortened by truncating but retaining essence of the text.]",
#             "page_number": [Page number of the original_text],
#             "filename": "[PDF file name where this original_text]"
#             "source":[specify whether text is extracted from 'orignal_text' or 'summary_text' field of context_metadata]
#             }},
#             [Repeat for each highlight from different files and page numbers]
            
#         ],
#         "ans": "[Give a natural language according to the query from the context_metadata]"
#         }}

#         Conversation so far: {previous_conversation}

#         Answer the following question: {query}

#         Context metadata: {context_info}"""}
#             ]
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=msgs,
#             max_tokens=self.max_tokens,
#             temperature=self.temperature
#         )
#         return response.choices[0].message.content.strip()
from .base import LLM
import json
from openai import OpenAI

class OpenAILLM(LLM):
    def __init__(self, api_key, model_name, max_tokens=1000, temperature=0.5):
        super().__init__(model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        self.client = OpenAI(api_key=api_key)

    def find_answer_gpt(self, query, context_metadata, previous_conversation):
        context_info = json.dumps(context_metadata, indent=2)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", 
            "content": f"""Please answer the following question by returning only a JSON response. 
            Make sure that the response does not contain any additional markers such as '''json or '''.
            In Highlight array, return only those relevant sections of context that has 'summary_text' as null and are used to form answer.
            The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:
    {{
    "highlight": [
        {{
        "text": "[Return original_text shortened by truncating but retaining essence of the text.]",
        "page_number": [Page number of the original_text],
        "filename": "[PDF file name where this original_text]"
        "source":[specify whether text is extracted from 'original_text' or 'summary_text' field of context_metadata]
        }},
        [Repeat for each highlight from different files and page numbers]
        
    ],
    "ans": "[Give a natural language according to the query from the context_metadata]"
    }}

    Conversation so far: {previous_conversation}

    Answer the following question: {query}

    Context metadata: {context_info}"""}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        # Print and debug the response object structure
        print("Response from Azure OpenAI API:", response)

        # Check if response has 'choices' and it is a list
        if hasattr(response, 'choices') and isinstance(response.choices, list) and len(response.choices) > 0:
            try:
                # Correctly access the 'content' attribute of the message object
                answer_content = response.choices[0].message.content.strip()
                print("Extracted content from response:", answer_content)
                
                # Parse the content into a dictionary
                gpt_answer = json.loads(answer_content)
                print("Parsed gpt_answer:", gpt_answer)
                
                # Ensure gpt_answer is a dictionary before accessing its keys
                if isinstance(gpt_answer, dict):
                    # Safely extract "highlight" and "ans" fields
                    highlights = gpt_answer.get("highlight", [])
                    ans = gpt_answer.get("ans", None)
                    
                    # Print the types and content for further debugging
                    print("Type of 'highlights':", type(highlights))
                    print("Highlights:", highlights)
                    print("Type of 'ans':", type(ans))
                    print("Answer:", ans)
                    
                    # If 'ans' is not found, handle the error
                    if ans is None:
                        return {"error": "Answer field 'ans' not found in the GPT response."}
                    
                    # Return the valid highlights and the answer
                    return {
                        "highlight": highlights,
                        "ans": ans
                    }
                else:
                    print("gpt_answer is not a dictionary, it's:", type(gpt_answer))
                    return {"error": "Unexpected response format: gpt_answer is not a dictionary."}
            except Exception as e:
                print(f"Error extracting content: {str(e)}")
                return {"error": f"Error extracting content from response. Details: {str(e)}"}
        else:
            print("Unexpected response structure:", response)
            return {"error": "Unexpected response structure from API."}