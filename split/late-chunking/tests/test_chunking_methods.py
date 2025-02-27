import pytest
from transformers import AutoTokenizer

from chunked_pooling.chunking import CHUNKING_STRATEGIES, Chunker
from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval

EXAMPLE_TEXT_1 = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
PUNCTATIONS = ('.', '!', '?')


@pytest.mark.parametrize("n_sentences", [1, 2, 3, 4])
def test_chunk_by_sentences(n_sentences):
    strategy = 'sentences'
    model_name = 'jinaai/jina-embeddings-v2-small-en'
    chunker = Chunker(chunking_strategy=strategy)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    boundary_cues = chunker.chunk(
        text=EXAMPLE_TEXT_1,
        tokenizer=tokenizer,
        chunking_strategy=strategy,
        n_sentences=n_sentences,
    )
    extended_boundary_cues = AbsTaskChunkedRetrieval._extend_special_tokens(
        boundary_cues
    )
    model_inputs = tokenizer(
        EXAMPLE_TEXT_1,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=8192,
    )

    # check that the cues start with 0 and end with the last token
    assert extended_boundary_cues[0][0] == 0
    assert len(model_inputs.tokens()) == extended_boundary_cues[-1][1]

    # check that all chunks but the last one end with a punctuation
    assert all(
        model_inputs.tokens()[x:y][-1] in PUNCTATIONS
        for (x, y) in extended_boundary_cues[:-1]
    )

    # check that the last chunk ends with a "[SEP]" token
    last_cue = extended_boundary_cues[-1]
    assert model_inputs.tokens()[last_cue[0] : last_cue[1]][-1] == "[SEP]"

    # check that the boundary cues are continuous (no token is missing)
    assert all(
        [
            extended_boundary_cues[i][1] == extended_boundary_cues[i + 1][0]
            for i in range(len(extended_boundary_cues) - 1)
        ]
    )


@pytest.mark.parametrize(
    "boundary_cues", [[(0, 17), (17, 44), (44, 69)], [(0, 44), (44, 69)]]
)
def test_token_equivalence(boundary_cues):
    model_name = 'jinaai/jina-embeddings-v2-small-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokens = tokenizer.encode_plus(
        EXAMPLE_TEXT_1, add_special_tokens=False, return_offsets_mapping=True
    )
    for start_token_idx, end_token_idx in boundary_cues:
        decoded_text_chunk = tokenizer.decode(
            tokens.input_ids[start_token_idx:end_token_idx]
        )

        original_text_chunk = EXAMPLE_TEXT_1[
            tokens.offset_mapping[start_token_idx][0] : tokens.offset_mapping[
                end_token_idx - 1
            ][1]
        ]
        chunk_tokens_original = tokenizer.encode_plus(original_text_chunk)
        chunk_tokens_decoded = tokenizer.encode_plus(decoded_text_chunk)
        assert chunk_tokens_original == chunk_tokens_decoded


def test_chunker_initialization():
    for strategy in CHUNKING_STRATEGIES:
        chunker = Chunker(chunking_strategy=strategy)
        assert chunker.chunking_strategy == strategy


def test_invalid_chunking_strategy():
    with pytest.raises(ValueError):
        Chunker(chunking_strategy="invalid")


def test_chunk_by_tokens():
    chunker = Chunker(chunking_strategy="fixed")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    chunks = chunker.chunk(EXAMPLE_TEXT_1, tokenizer=tokenizer, chunk_size=10)
    assert len(chunks) > 1
    for start, end in chunks:
        assert end - start <= 10


@pytest.mark.parametrize(
    'model_name',
    ['jinaai/jina-embeddings-v2-small-en', 'sentence-transformers/all-MiniLM-L6-v2'],
)
def test_chunk_semantically(model_name):
    chunker = Chunker(chunking_strategy="semantic")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode_plus(
        EXAMPLE_TEXT_1, add_special_tokens=False, return_offsets_mapping=True
    )
    boundary_cues = chunker.chunk(
        EXAMPLE_TEXT_1,
        tokenizer=tokenizer,
        chunking_strategy='semantic',
        embedding_model_name=model_name,
    )

    # check if it returns boundary cues
    assert len(boundary_cues) > 0

    # test if bounaries are at the end of sentences
    for start_token_idx, end_token_idx in boundary_cues:
        assert (
            EXAMPLE_TEXT_1[tokens.offset_mapping[end_token_idx - 1][0]] in PUNCTATIONS
        )
        decoded_text_chunk = tokenizer.decode(
            tokens.input_ids[start_token_idx:end_token_idx]
        )

    # check that the boundary cues are continuous (no token is missing)
    assert all(
        [
            boundary_cues[i][1] == boundary_cues[i + 1][0]
            for i in range(len(boundary_cues) - 1)
        ]
    )


def test_empty_input():
    chunker = Chunker(chunking_strategy="fixed")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    chunks = chunker.chunk("", tokenizer=tokenizer, chunk_size=10)
    assert len(chunks) == 0


def test_input_shorter_than_chunk_size():
    short_text = "Short text."
    chunker = Chunker(chunking_strategy="fixed")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    chunks = chunker.chunk(short_text, tokenizer=tokenizer, chunk_size=20)
    assert len(chunks) == 1


@pytest.mark.parametrize("chunk_size", [10, 20, 50])
def test_various_chunk_sizes(chunk_size):
    chunker = Chunker(chunking_strategy="fixed")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    chunks = chunker.chunk(EXAMPLE_TEXT_1, tokenizer=tokenizer, chunk_size=chunk_size)
    assert len(chunks) > 0
    for start, end in chunks:
        assert end - start <= chunk_size


def test_chunk_method_with_different_strategies():
    chunker = Chunker(chunking_strategy="fixed")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fixed_chunks = chunker.chunk(EXAMPLE_TEXT_1, tokenizer=tokenizer, chunk_size=10)
    semantic_chunks = chunker.chunk(
        EXAMPLE_TEXT_1,
        tokenizer=tokenizer,
        chunking_strategy="semantic",
        embedding_model_name='jinaai/jina-embeddings-v2-small-en',
    )
    assert fixed_chunks != semantic_chunks


def test_chunk_by_sentences_different_n():
    chunker = Chunker(chunking_strategy="sentences")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    chunks_1 = chunker.chunk(EXAMPLE_TEXT_1, tokenizer=tokenizer, n_sentences=1)
    chunks_2 = chunker.chunk(EXAMPLE_TEXT_1, tokenizer=tokenizer, n_sentences=2)
    assert len(chunks_1) > len(chunks_2)
