import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata

from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval


class SciFactChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name='SciFactChunked',
        dataset={
            'path': 'mteb/scifact',
            'revision': '0228b52cf27578f30900b9e5271d331663a030d7',
            'name': 'SciFact',
        },
        description=(
            'SciFact verifies scientific claims using evidence from the '
            'research literature containing scientific paper abstracts.'
        ),
        reference='https://github.com/allenai/scifact',
        type='Retrieval',
        category='s2p',
        eval_splits=['test'],
        eval_langs=['eng-Latn'],
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NarrativeQAChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name='NarrativeQAChunked',
        dataset={
            'path': 'narrativeqa',
            'revision': '2e643e7363944af1c33a652d1c87320d0871c4e4',
            'name': 'NarrativeQARetrieval',
        },
        reference='https://metatext.io/datasets/narrativeqa',
        description=(
            'NarrativeQA is a dataset for the task of question answering '
            'on long narratives. It consists of realistic QA instances '
            'collected from literature (fiction and non-fiction) '
            'and movie scripts. '
        ),
        type='Retrieval',
        category='s2p',
        eval_splits=['test'],
        eval_langs=['eng-Latn'],
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NFCorpusChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name="NFCorpusChunked",
        dataset={
            "path": "mteb/nfcorpus",
            "revision": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
            'name': 'NFCorpus',
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class QuoraChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name="QuoraChunked",
        dataset={
            "path": "mteb/quora",
            "revision": "e4e08e0b7dbe3c8700f0daef558ff32256715259",
            "name": "QuoraRetrieval",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            " question, find other (duplicate) questions."
        ),
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        type="Retrieval",
        category="s2s",
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FiQA2018Chunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name="FiQA2018Chunked",
        description="Financial Opinion Mining and Question Answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "mteb/fiqa",
            "revision": "27a168819829fe9bcd655c2df245fb19452e8e06",
            'name': 'FiQA2018',
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train", "dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TRECCOVIDChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name='TRECCOVIDChunked',
        description=(
            'TRECCOVID is an ad-hoc search challenge based on the '
            'COVID-19 dataset containing scientific articles '
            'related to the COVID-19 pandemic.'
        ),
        reference='https://ir.nist.gov/covidSubmit/index.html',
        dataset={
            'path': 'mteb/trec-covid',
            'revision': 'bb9466bac8153a0349341eb1b22e06409e78ef4e',
            'name': 'TRECCOVID',
        },
        type='Retrieval',
        category='s2p',
        eval_splits=['test'],
        eval_langs=['eng-Latn'],
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LEMBWikimQARetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBWikimQARetrieval.py
    """

    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBWikimQARetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "10039a580487dacecf79db69166e17ace3ede392",
            "name": "LEMBWikimQARetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("2wikimqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=None,
        socioeconomic_status=None,
        n_samples=None,
        avg_character_length=None,
        form=None,
        text_creation=None,
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{ho2020constructing,
                title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
                author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
                booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
                pages={6609--6625},
                year={2020}
            }
        """,
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: 500},
            "avg_character_length": {
                "test": {
                    "average_document_length": 37445.60333333333,
                    "average_query_length": 67.57,
                    "num_documents": 300,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = '2wikimqa'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True


class LEMBSummScreenFDRetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBWikimQARetrieval.py
    """

    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBSummScreenFDRetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "10039a580487dacecf79db69166e17ace3ede392",
            "name": "LEMBSummScreenFDRetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("summ_screen_fd subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=None,
        socioeconomic_status=None,
        n_samples=None,
        avg_character_length=None,
        form=None,
        text_creation=None,
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{ho2020constructing,
                title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
                author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
                booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
                pages={6609--6625},
                year={2020}
            }
        """,
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: 500},
            "avg_character_length": {
                "test": {
                    "average_document_length": 30854.327,
                    "average_query_length": 591.49,
                    "num_documents": 300,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = 'summ_screen_fd'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True


class LEMBQMSumRetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBWikimQARetrieval.py
    """

    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBQMSumRetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "10039a580487dacecf79db69166e17ace3ede392",
            "name": "LEMBQMSumRetrieval",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("qmsum subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=None,
        socioeconomic_status=None,
        n_samples=None,
        avg_character_length=None,
        form=None,
        text_creation=None,
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @inproceedings{ho2020constructing,
                title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
                author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
                booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
                pages={6609--6625},
                year={2020}
            }
        """,
        descriptive_stats={
            "n_samples": {_EVAL_SPLIT: 500},
            "avg_character_length": {
                "test": {
                    "average_document_length": 53335.817,
                    "average_query_length": 433.50,
                    "num_documents": 300,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset_dict = {**self.metadata.dataset}
        dataset_dict['name'] = 'qmsum'

        query_list = datasets.load_dataset(**dataset_dict)["queries"]
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**dataset_dict)["corpus"]
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**dataset_dict)["qrels"]
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True


class LEMBNeedleRetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBNeedleRetrieval.py
    """

    _EVAL_SPLIT = [
        "test_256",
        "test_512",
        "test_1024",
        "test_2048",
        "test_4096",
        "test_8192",
        "test_16384",
        "test_32768",
    ]

    metadata = TaskMetadata(
        name="LEMBNeedleRetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "needle",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("needle subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=_EVAL_SPLIT,
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_1",
        date=("2000-01-01", "2023-12-31"),
        domains=["Academic", "Blog", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @article{zhu2024longembed,
            title={LongEmbed: Extending Embedding Models for Long Context Retrieval},
            author={Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
            journal={arXiv preprint arXiv:2404.12096},
            year={2024}
            }
        """,
        descriptive_stats={
            "n_samples": {
                "test_256": 150,
                "test_512": 150,
                "test_1024": 150,
                "test_2048": 150,
                "test_4096": 150,
                "test_8192": 150,
                "test_16384": 150,
                "test_32768": 150,
            },
            "avg_character_length": {
                "test_256": {
                    "average_document_length": 1013.22,
                    "average_query_length": 60.48,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_512": {
                    "average_document_length": 2009.96,
                    "average_query_length": 57.3,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_1024": {
                    "average_document_length": 4069.9,
                    "average_query_length": 58.28,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_2048": {
                    "average_document_length": 8453.82,
                    "average_query_length": 59.92,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_4096": {
                    "average_document_length": 17395.8,
                    "average_query_length": 55.86,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_8192": {
                    "average_document_length": 35203.82,
                    "average_query_length": 59.6,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_16384": {
                    "average_document_length": 72054.8,
                    "average_query_length": 59.12,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_32768": {
                    "average_document_length": 141769.8,
                    "average_query_length": 58.34,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        for split in self._EVAL_SPLIT:
            context_length = int(split.split("_")[1])
            query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "queries"
            ]  # dict_keys(['qid', 'text'])
            query_list = query_list.filter(
                lambda x: x["context_length"] == context_length
            )
            queries = {row["qid"]: row["text"] for row in query_list}

            corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "corpus"
            ]  # dict_keys(['doc_id', 'text'])
            corpus_list = corpus_list.filter(
                lambda x: x["context_length"] == context_length
            )
            corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

            qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "qrels"
            ]  # dict_keys(['qid', 'doc_id'])
            qrels_list = qrels_list.filter(
                lambda x: x["context_length"] == context_length
            )
            qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

            self.corpus[split] = corpus
            self.queries[split] = queries
            self.relevant_docs[split] = qrels

        self.data_loaded = True


class LEMBPasskeyRetrievalChunked(AbsTaskChunkedRetrieval):
    """
    modified from https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/Retrieval/eng/LEMBPasskeyRetrieval.py
    """

    _EVAL_SPLIT = [
        "test_256",
        "test_512",
        "test_1024",
        "test_2048",
        "test_4096",
        "test_8192",
        "test_16384",
        "test_32768",
    ]

    metadata = TaskMetadata(
        name="LEMBPasskeyRetrievalChunked",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "passkey",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("passkey subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=_EVAL_SPLIT,
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_1",
        date=("2000-01-01", "2023-12-31"),
        domains=["Fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @article{zhu2024longembed,
            title={LongEmbed: Extending Embedding Models for Long Context Retrieval},
            author={Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
            journal={arXiv preprint arXiv:2404.12096},
            year={2024}
            }
        """,
        descriptive_stats={
            "n_samples": {
                "test_256": 150,
                "test_512": 150,
                "test_1024": 150,
                "test_2048": 150,
                "test_4096": 150,
                "test_8192": 150,
                "test_16384": 150,
                "test_32768": 150,
            },
            "avg_character_length": {
                "test_256": {
                    "average_document_length": 876.24,
                    "average_query_length": 38.1,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_512": {
                    "average_document_length": 1785.2,
                    "average_query_length": 37.76,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_1024": {
                    "average_document_length": 3607.18,
                    "average_query_length": 37.68,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_2048": {
                    "average_document_length": 7242.2,
                    "average_query_length": 37.8,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_4096": {
                    "average_document_length": 14518.16,
                    "average_query_length": 37.64,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_8192": {
                    "average_document_length": 29071.16,
                    "average_query_length": 37.54,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_16384": {
                    "average_document_length": 58175.16,
                    "average_query_length": 38.12,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
                "test_32768": {
                    "average_document_length": 116380.16,
                    "average_query_length": 37.74,
                    "num_documents": 100,
                    "num_queries": 50,
                    "average_relevant_docs_per_query": 1.0,
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        for split in self._EVAL_SPLIT:
            context_length = int(split.split("_")[1])
            query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "queries"
            ]  # dict_keys(['qid', 'text'])
            query_list = query_list.filter(
                lambda x: x["context_length"] == context_length
            )
            queries = {row["qid"]: row["text"] for row in query_list}

            corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "corpus"
            ]  # dict_keys(['doc_id', 'text'])
            corpus_list = corpus_list.filter(
                lambda x: x["context_length"] == context_length
            )
            corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

            qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
                "qrels"
            ]  # dict_keys(['qid', 'doc_id'])
            qrels_list = qrels_list.filter(
                lambda x: x["context_length"] == context_length
            )
            qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

            self.corpus[split] = corpus
            self.queries[split] = queries
            self.relevant_docs[split] = qrels

        self.data_loaded = True
