from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os

def create_vector_index(openai_api_key):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    # define langchain
    langchain = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)

    promptHelper = PromptHelper(langchain, max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    # load documents
    docs = SimpleDirectoryReader('knowledge').load_data()

    # define LLM
    #llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=tokens, openai_api_key=openai_api_key))
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens, openai_api_key=openai_api_key))

    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=promptHelper) 
    vectorIndex = GPTSimpleVectorIndex.from_documents(documents=docs,service_context=service_context)

    # create vector index
    #vectorIndex = GPTSimpleVectorIndex(documents=docs,llm_predictor=llmPredictor,prompt_helper=promptHelper)
    vectorIndex.save_to_disk('vectorIndex.json')
    return vectorIndex

with open('openaiapikey.txt', 'r') as infile:
    openai_api_key = infile.read()
    OpenAI.api_key = openai_api_key

    # load documents and create vector index
    vector_index = create_vector_index(openai_api_key)


