from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os

with open('openaiapikey.txt', 'r') as infile:
    openai_api_key = infile.read()
    os.environ['OPENAI_API_KEY'] = openai_api_key

# Set the default LLMPredictor with the openai_api_key
from gpt_index.indices.service_context import ServiceContext

def answer_me(vectorIndex):
    max_input = 4096
    tokens = 256
    #chunk_size = 600
    max_chunk_overlap = 20

    videx = GPTSimpleVectorIndex.load_from_disk(vectorIndex)
    
    while True:
        prompt = input("Enter your question: ")
        response = videx.query(prompt,response_mode="compact")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    answer_me('vectorIndex.json')
