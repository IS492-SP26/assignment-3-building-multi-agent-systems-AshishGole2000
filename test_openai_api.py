import os

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL"),
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)