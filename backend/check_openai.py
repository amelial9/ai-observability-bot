import os
from openai import OpenAI

client = OpenAI()

try:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print("OpenAI check passed!")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"OpenAI check failed: {e}")
