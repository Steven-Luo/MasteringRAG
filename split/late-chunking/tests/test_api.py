import os
import numpy as np
from transformers import AutoModel, AutoTokenizer

from chunked_pooling import chunked_pooling
from chunked_pooling.wrappers import load_model
from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval

MODEL_NAME = 'jinaai/jina-embeddings-v3'

# Define Text and Chunk
CHUNKS = ["Organic skincare", "for sensitive skin", "with aloe vera and chamomile"]
FULL_TEXT = ' '.join(CHUNKS)


def load_api_results():
    import requests

    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.environ["JINA_API_TOKEN"]}',
    }
    data = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage",
        "dimensions": 1024,
        "late_chunking": True,
        "embedding_type": "float",
        "input": CHUNKS,
    }
    response = requests.post(url, headers=headers, json=data)
    data = response.json()
    return [np.array(x['embedding']) for x in data['data']]


def calculate_annotations(model, boundary_cues, model_has_instructions, tokenizer):
    if model_has_instructions:
        instr = model.get_instructions()[1]
        instr_tokens = tokenizer(instr, add_special_tokens=False)
        n_instruction_tokens = len(instr_tokens[0])
    else:
        n_instruction_tokens = 0
    chunk_annotations = [
        AbsTaskChunkedRetrieval._extend_special_tokens(
            annotations,
            n_instruction_tokens=n_instruction_tokens,
            include_prefix=True,
            include_sep=True,
        )
        for annotations in boundary_cues
    ]
    return chunk_annotations


def test_compare_v3_api_embeddings():
    # Load Model
    model, has_instr = load_model(MODEL_NAME, use_flash_attn=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Determine Boundary Cues
    tokenization = tokenizer(
        FULL_TEXT, return_offsets_mapping=True, add_special_tokens=False
    )
    boundary_cues = []
    chunk_i = 0
    last_cue = 0
    last_end = 0
    for i, (start, end) in enumerate(tokenization.offset_mapping):
        if end >= (last_end + len(CHUNKS[chunk_i])):
            boundary_cues.append((last_cue, i + 1))
            chunk_i += 1
            last_cue = i + 1
            last_end = end
    extended_boundary_cues = calculate_annotations(
        model, [boundary_cues], has_instr, tokenizer
    )

    # Append Instruction for Retrieval Task
    instr = model.get_instructions()[1]
    text_inputs = [instr + FULL_TEXT]
    model_inputs = tokenizer(
        text_inputs,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=8192,
    )
    model_outputs = model(**model_inputs)

    # Apply Late Chunking
    output_embs = chunked_pooling(
        model_outputs, extended_boundary_cues, max_length=8192
    )[0]
    api_embs = load_api_results()
    for local_emb, api_emb in zip(output_embs, api_embs):
        local_emb_norm = local_emb / np.linalg.norm(local_emb)
        api_emb_norm = api_emb / np.linalg.norm(api_emb)
        assert np.allclose(local_emb_norm, api_emb_norm, rtol=1e-02, atol=1e-02)
        assert 1.0 - np.dot(local_emb_norm, api_emb_norm) < 1e-3
