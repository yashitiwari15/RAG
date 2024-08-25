import os
from dotenv import load_dotenv
from groq import Groq, GroqError

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise GroqError("GROQ_API_KEY environment variable is not set")

# Instantiation of Groq Client
client = Groq(api_key=api_key)

try:
    llm = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI Assistant. You explain every topic the user asks as if you are explaining it to a 5-year-old."
            },
            {
                "role": "user",
                "content": "What are Black Holes?",
            }
        ],
        model="llama-3.1-8b-instant",  # Replace with the correct model identifier for LLaMA 3
    )

    print(llm.choices[0].message.content)

except GroqError as e:
    print(f"An error occurred: {e}")
