from transformers import AutoTokenizer

from run_chunked_eval import DEFAULT_CHUNK_SIZE, load_model

MODEL_NAME = 'jinaai/jina-embeddings-v3'


def test_instruction_handling(dummy_task_factory):
    model, has_instructions = load_model(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    task = dummy_task_factory(
        chunking_strategy='fixed',
        chunk_size=DEFAULT_CHUNK_SIZE,
        tokenizer=tokenizer,
        model_has_instructions=has_instructions,
    )
    n_instruction_tokens = len(
        tokenizer(model.get_instructions()[1], add_special_tokens=False)['input_ids']
    )
    annotations_one_token = task._calculate_annotations(model, ['A'])[0]
    assert len(annotations_one_token) == 1
    assert annotations_one_token[0] == (0, n_instruction_tokens + 3)
