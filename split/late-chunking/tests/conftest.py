import pytest
from mteb.abstasks.TaskMetadata import TaskMetadata

from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval


class DummyTask(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        dataset={
            'path': '~',
            'revision': '',
        },
        name='dummy',
        description='',
        type='Retrieval',
        category='s2p',
        reference=None,
        eval_splits=[],
        eval_langs=[],
        main_score='ndcg_at_10',
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data():
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture()
def dummy_task_factory():
    def _create_dummy_task(*args, **kwargs):
        return DummyTask(*args, **kwargs)

    return _create_dummy_task
