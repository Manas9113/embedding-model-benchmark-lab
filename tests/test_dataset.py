"""
Unit tests for DatasetLoader and IRDataset using synthetic fixtures.
Ensures parser handles JSONL, TSV, headers, and edge cases correctly.
"""

import json
from pathlib import Path
import pytest
from src.data.dataset_loader import DatasetLoader, IRDataset


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Creates a temporary miniature BEIR-structured dataset for testing."""
    dataset_dir = tmp_path / "toy_dataset"
    dataset_dir.mkdir()
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir()

    # 1. Mock corpus.jsonl
    corpus_records = [
        {"_id": "doc1", "title": "Doc One", "text": "Medical research on diet."},
        {"_id": "doc2", "title": "Doc Two", "text": "Cardiovascular health and exercise."},
        {"_id": "doc3", "title": "Doc Three", "text": "Unrelated topic on astronomy."},
    ]
    with open(dataset_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in corpus_records:
            f.write(json.dumps(r) + "\n")

    # 2. Mock queries.jsonl
    query_records = [
        {"_id": "q1", "text": "healthy diet guidelines"},
        {"_id": "q2", "text": "how to improve heart health"},
    ]
    with open(dataset_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for r in query_records:
            f.write(json.dumps(r) + "\n")

    # 3. Mock qrels/test.tsv
    qrels_content = (
        "query-id\tcorpus-id\tscore\n"
        "q1\tdoc1\t2\n"
        "q2\tdoc2\t1\n"
    )
    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f:
        f.write(qrels_content)

    return dataset_dir


def test_loader_parses_corpus_correctly(temp_dataset_dir: Path):
    loader = DatasetLoader(data_dir=str(temp_dataset_dir))
    corpus = loader.load_corpus()

    assert len(corpus) == 3
    assert "doc1" in corpus
    assert corpus["doc1"]["title"] == "Doc One"
    assert "diet" in corpus["doc1"]["text"]


def test_loader_parses_queries_correctly(temp_dataset_dir: Path):
    loader = DatasetLoader(data_dir=str(temp_dataset_dir))
    queries = loader.load_queries()

    assert len(queries) == 2
    assert queries["q1"] == "healthy diet guidelines"
    assert queries["q2"] == "how to improve heart health"


def test_loader_parses_qrels_correctly(temp_dataset_dir: Path):
    loader = DatasetLoader(data_dir=str(temp_dataset_dir))
    qrels = loader.load_qrels(split="test")

    assert len(qrels) == 2
    assert qrels["q1"]["doc1"] == 2
    assert qrels["q2"]["doc2"] == 1


def test_validation_reports_accurate_stats(temp_dataset_dir: Path):
    loader = DatasetLoader(data_dir=str(temp_dataset_dir))
    dataset = loader.load_dataset(split="test", auto_download=False)

    stats = loader.validate(dataset)
    assert stats["num_corpus_documents"] == 3
    assert stats["num_test_queries"] == 2
    assert stats["total_relevance_judgments"] == 2
    assert stats["missing_document_references"] == 0
    assert stats["score_distribution"][2] == 1
    assert stats["score_distribution"][1] == 1
