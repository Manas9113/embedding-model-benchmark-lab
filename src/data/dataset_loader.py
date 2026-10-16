"""
Dataset loader and validator for the BEIR NFCorpus information retrieval dataset.
Handles automated download, extraction, parsing, and integrity checks.
"""

import csv
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("dataset_loader")


@dataclass
class IRDataset:
    """Represents an Information Retrieval dataset adhering to BEIR standards."""

    name: str
    corpus: Dict[str, Dict[str, str]] = field(default_factory=dict)  # doc_id -> {"title": ..., "text": ...}
    queries: Dict[str, str] = field(default_factory=dict)  # query_id -> query_text
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)  # query_id -> {doc_id -> relevance_score}

    @property
    def num_documents(self) -> int:
        return len(self.corpus)

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_qrels(self) -> int:
        return sum(len(docs) for docs in self.qrels.values())


class DatasetLoader:
    """Manages downloading, caching, and parsing BEIR benchmark datasets."""

    def __init__(self, data_dir: str = "data/nfcorpus", download_url: str | None = None):
        self.data_dir = Path(data_dir)
        self.download_url = (
            download_url
            or "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
        )

    def download_and_extract(self, force_download: bool = False) -> Path:
        """Downloads the zipped dataset if not present, and extracts it to data_dir."""
        corpus_file = self.data_dir / "corpus.jsonl"
        queries_file = self.data_dir / "queries.jsonl"
        qrels_file = self.data_dir / "qrels" / "test.tsv"

        if not force_download and corpus_file.exists() and queries_file.exists() and qrels_file.exists():
            logger.info("Dataset already exists at %s. Skipping download.", self.data_dir)
            return self.data_dir

        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir.parent / "nfcorpus.zip"

        logger.info("Downloading dataset from %s ...", self.download_url)
        try:
            import certifi
            import ssl
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            import ssl
            ssl_ctx = ssl._create_unverified_context()

        try:
            with urllib.request.urlopen(self.download_url, context=ssl_ctx) as response:
                with open(str(zip_path), "wb") as out_file:
                    while chunk := response.read(65536):
                        out_file.write(chunk)
        except Exception as e:
            logger.warning("Download with certifi context failed: %s. Retrying with fallback...", e)
            import ssl
            fallback_ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(self.download_url, context=fallback_ctx) as response:
                with open(str(zip_path), "wb") as out_file:
                    while chunk := response.read(65536):
                        out_file.write(chunk)

        logger.info("Downloaded %s (%.2f MB)", zip_path.name, zip_path.stat().st_size / (1024 * 1024))

        logger.info("Extracting archive to %s ...", self.data_dir.parent)
        with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
            zip_ref.extractall(str(self.data_dir.parent))

        if zip_path.exists():
            zip_path.unlink()  # Clean up zip file to save disk space

        logger.info("Dataset successfully extracted to %s", self.data_dir)
        return self.data_dir

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        """Loads corpus.jsonl into a dictionary mapping doc_id -> {title, text}."""
        corpus_path = self.data_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found at {corpus_path}")

        corpus: Dict[str, Dict[str, str]] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc_id = str(doc["_id"])
                    corpus[doc_id] = {
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Skipping malformed corpus entry at line %d: %s", line_num, e)

        logger.info("Loaded %d documents from corpus.jsonl", len(corpus))
        return corpus

    def load_queries(self) -> Dict[str, str]:
        """Loads queries.jsonl into a dictionary mapping query_id -> query_text."""
        queries_path = self.data_dir / "queries.jsonl"
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found at {queries_path}")

        queries: Dict[str, str] = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    q = json.loads(line)
                    query_id = str(q["_id"])
                    queries[query_id] = q.get("text", "")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Skipping malformed query entry at line %d: %s", line_num, e)

        logger.info("Loaded %d queries from queries.jsonl", len(queries))
        return queries

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, int]]:
        """Loads qrels/{split}.tsv into query_id -> {doc_id -> relevance_score}."""
        qrels_path = self.data_dir / "qrels" / f"{split}.tsv"
        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found at {qrels_path}")

        qrels: Dict[str, Dict[str, int]] = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            # Check for header
            header = next(reader, None)
            start_row = 1
            if header and header[0].lower() in ["query-id", "query_id"]:
                pass  # header was skipped
            else:
                # First row was actual data
                if header and len(header) >= 3:
                    q_id, d_id, score = header[0], header[1], int(float(header[2]))
                    qrels.setdefault(q_id, {})[d_id] = score

            for row in reader:
                start_row += 1
                if not row or len(row) < 3:
                    continue
                q_id, d_id = str(row[0]), str(row[1])
                score = int(float(row[2]))
                qrels.setdefault(q_id, {})[d_id] = score

        logger.info("Loaded qrels for %d queries from %s", len(qrels), qrels_path)
        return qrels

    def load_dataset(self, split: str = "test", auto_download: bool = True) -> IRDataset:
        """Complete workflow to download, parse, and assemble the IRDataset."""
        if auto_download:
            self.download_and_extract()

        corpus = self.load_corpus()
        all_queries = self.load_queries()
        qrels = self.load_qrels(split=split)

        # Filter queries to only those present in the test qrels
        active_queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}

        dataset = IRDataset(
            name="nfcorpus",
            corpus=corpus,
            queries=active_queries,
            qrels=qrels,
        )
        self.validate(dataset)
        return dataset

    def validate(self, dataset: IRDataset) -> Dict[str, Any]:
        """Validates dataset integrity and logs distribution statistics."""
        missing_docs = 0
        total_qrels = 0
        score_distribution: Dict[int, int] = {}

        for qid, judgments in dataset.qrels.items():
            for doc_id, score in judgments.items():
                total_qrels += 1
                score_distribution[score] = score_distribution.get(score, 0) + 1
                if doc_id not in dataset.corpus:
                    missing_docs += 1

        avg_doc_len = sum(len(d["text"]) for d in dataset.corpus.values()) / max(len(dataset.corpus), 1)
        avg_query_len = sum(len(q) for q in dataset.queries.values()) / max(len(dataset.queries), 1)

        stats = {
            "num_corpus_documents": dataset.num_documents,
            "num_test_queries": dataset.num_queries,
            "total_relevance_judgments": total_qrels,
            "missing_document_references": missing_docs,
            "score_distribution": score_distribution,
            "avg_document_char_length": round(avg_doc_len, 1),
            "avg_query_char_length": round(avg_query_len, 1),
        }

        logger.info("=== Dataset Integrity & Validation Report ===")
        logger.info("Documents in corpus: %d", stats["num_corpus_documents"])
        logger.info("Evaluated test queries: %d", stats["num_test_queries"])
        logger.info("Total gold relevance pairs: %d", stats["total_relevance_judgments"])
        logger.info("Relevance score breakdown: %s", stats["score_distribution"])
        logger.info("Avg document length: %.1f chars | Avg query length: %.1f chars", avg_doc_len, avg_query_len)

        if missing_docs > 0:
            logger.warning("Found %d qrel references to documents not in the corpus!", missing_docs)
        else:
            logger.info("Validation passed: All gold qrel references exist in the corpus.")

        return stats
