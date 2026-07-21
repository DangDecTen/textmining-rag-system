import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(
        self,
        model="llama-3.3-70b-versatile"
    ):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def build_prompt(self, query: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join(
            [
                f"[{doc['chunk_id']}]\n"
                f"{doc['text']}"
                for doc in contexts
            ]
        )

        return f"""
        You are a cybersecurity assistant specializing in MITRE ATT&CK. 
        Answer ONLY using the provided context.
        
        If the answer is not present in the context, say that you do not have enough information.
        
        Context:
        {context_text}
        
        Question:
        {query}
        
        Answer:"""

    def generate(self, query: str, contexts: list[dict]) -> str:
        prompt = self.build_prompt(query, contexts)

        response = (
        self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        ))
        return (response.choices[0].message.content)