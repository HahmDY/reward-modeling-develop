import os
from openai import OpenAI

class OpenAILLM:
    def __init__(self, 
                 model_name, 
                 base_url=None,
                 api_key=os.getenv("OPENAI_API_KEY")):
        self.model = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
            
    def chat_reset(self, system_prompt=None):
        if system_prompt is not None:
            self.system_prompt = system_prompt
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.system_prompt = None
            self.messages = []

    def generate(self, 
                 prompt,
                 temperature=0.0,
                 top_p=1.0,
                 max_tokens=1024):
        response = self.client.completions.create(model=self.model, 
                                                  prompt=prompt, 
                                                  max_tokens=max_tokens, 
                                                  temperature=temperature, 
                                                  top_p=top_p)
        return response.choices[0].text

    def chat_generate(self, 
                      messages,
                      temperature=0.0,
                      top_p=1.0,
                      max_tokens=1024):
        response = self.client.chat.completions.create(model=self.model, 
                                                    messages=messages, 
                                                    max_tokens=max_tokens, 
                                                    temperature=temperature, 
                                                    top_p=top_p)
        return response.choices[0].message.content
    
if __name__ == "__main__":
    llm = OpenAILLM(model_name="gpt-4.1-nano")
    print(llm.chat_generate(messages=[{"role": "system", "content": "Answer in Korean"},
                             {"role": "user", "content": "Hello, how are you?"}]))