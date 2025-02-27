# experiments/explanatory_contextual_retrieval.py
# 
# a simple example with a trivial piece of text to showcase the late chunking method against 
# contextual retrieval method. contextual retrieval manually inserts context to each
# chunk, i.e. forces context to be around each chunk. so works as a good comparison
# to late chunking to see if the similarities are similar (which they appear to be)

from chunked_pooling.wrappers import load_model
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import numpy as np

import chunked_pooling
from chunked_pooling import chunked_pooling
from chunked_pooling.chunking import Chunker

from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer, pipeline

import requests
import os

def request_anthropic_api(prompt: str):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["content"][0]["text"]

def setup_local_llm(llm_name):
    
    model = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)

    def llm(prompt):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(inputs, max_new_tokens=512)
        text_output = tokenizer.batch_decode(outputs)[0]
        if "<|assistant|>" in text_output:
            text_output = text_output.split("<|assistant|>")[1].strip()
        return text_output
    
    return llm

def cosine_similarity(vector1, vector2):
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    return np.dot(vector1_norm, vector2_norm)

class LateChunkingEmbedder:

    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            chunking_strategy: str = "sentences",
            n_sentences: int = 1
        ):

        self.model = model
        self.tokenizer = tokenizer

        self.chunker = Chunker(chunking_strategy = chunking_strategy)
        self.n_sentences = n_sentences

    
    def run(self, document: str):
        annotations = [self.chunker.chunk(text=document, tokenizer=self.tokenizer, n_sentences=self.n_sentences)]
        model_inputs = self.tokenizer(
            document,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8192,
        )
        model_outputs = self.model(**model_inputs)
        self.output_embs = chunked_pooling(
            model_outputs, annotations, max_length=8192, 
        )[0]
        return self.output_embs

    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        query_embedding = self.model.encode(query)
        similarities = []
        for emb in self.output_embs:
            similarities.append(cosine_similarity(query_embedding, emb))
        
        return similarities


class ContextualRetrievalEmbedder():
    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            llm_name: str = "microsoft/Phi-3.5-mini-instruct",
            chunking_strategy: str = "fixed"
        ):

        self.llm = setup_local_llm(llm_name)
        # self.llm = request_anthropic_api

        self.prompt = """
        <document> 
        {{WHOLE_DOCUMENT}} 
        </document> 
        Here is the chunk we want to situate within the whole document 
        <chunk> 
        {{CHUNK_CONTENT}} 
        </chunk> 
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
        """.strip()

        self.model = model
        self.tokenizer = tokenizer

        self.chunker = Chunker(chunking_strategy = chunking_strategy)


    def _add_context(self, chunk: str, document: str):
        prompt = self.prompt.replace("{{WHOLE_DOCUMENT}}", document).replace("{{CHUNK_CONTENT}}", chunk)
        extra_context = self.llm(prompt)
        return extra_context + " " + chunk

    def _tokens_to_text(self, text: str, annotations: List[Tuple[int, int]]):
        tokens = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping
        chunks = []
        for start, end in annotations:
            chunk = text[token_offsets[start][0]:token_offsets[end-1][1]]
            chunks.append(chunk)
        return chunks
    
    def run(self, document: str):
        annotations = [self.chunker.chunk(text=document, tokenizer=self.tokenizer, n_sentences=1)]
        self.chunks = self._tokens_to_text(text=document, annotations=annotations[0])
        self.chunks = [self._add_context(chunk, document) for chunk in self.chunks]

        model_outputs = self.model.encode(self.chunks)
        self.output_embs = [model_outputs[i, :] for i in range(len(self.chunks))]
        return self.output_embs

    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        query_embedding = self.model.encode(query)
        similarities = []
        for emb in self.output_embs:
            similarities.append(cosine_similarity(query_embedding, emb))
        
        return similarities

        

if __name__ == "__main__":

    text = """
    The recent SEC filing provided insights into ACME Corp's performance for Q2 2023. 
    It highlighted a 3% revenue growth over the previous quarter. 
    The company, which had a revenue of $314 million in the prior quarter, showed steady progress. 
    They attributed this growth to strategic initiatives and operational efficiencies. 
    The report emphasized the company's resilience and ability to navigate market challenges, reflecting positively on their financial health and future prospects.
    """.strip().replace("\n", "")

    llm_model_name = "microsoft/Phi-3.5-mini-instruct"
    embedding_model_name = "jinaai/jina-embeddings-v2-small-en"

    embedding_model, has_instructions = load_model(embedding_model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)

    cr = ContextualRetrievalEmbedder(embedding_model, embedding_tokenizer, llm_model_name, chunking_strategy="sentences")
    cr.run(text);
    cr_cosine_similarities = cr.query("What is ACME Corp's revenue growth for Q2 2023?")

    lc = LateChunkingEmbedder(embedding_model, embedding_tokenizer)
    lc.run(text)
    lc_cosine_similarities = lc.query("What is ACME Corp's revenue growth for Q2 2023?")

    # import pandas as pd
    for i, (cr_similarity, lc_similarity) in enumerate(zip(cr_cosine_similarities, lc_cosine_similarities)):
        print(f"{text.split('.')[:-1][i].strip()}")
        print(f"Similarities: Contextual Retrieval: {cr_similarity:.4f} | Late Chunking: {lc_similarity:.4f}")
        print("")