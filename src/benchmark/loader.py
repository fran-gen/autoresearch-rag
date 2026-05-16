from __future__ import annotations

import json
import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx


logger = logging.getLogger(__name__)


DEFAULT_RELEASE_BASE = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/latest/download"
)
QUESTIONS_RAW_URL = (
    "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main/questions.jsonl"
)
EXTRA_QUESTIONS_RAW_URL = (
    "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main/extra_questions.jsonl"
)

DSID_PREFIX_RE = re.compile(r"^(dsid_[a-f0-9]+)")


@dataclass(slots=True)
class BenchmarkDocument:
    document_id: str
    source_type: str
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    question_type: str
    source_types: list[str]
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _document_from_txt(path: Path, source_type: str = "unknown") -> BenchmarkDocument:
    """Parse a benchmark .txt document file.

    File names are prefixed with the dataset UUID (dsid_...) followed by a
    semantic name.  The first line of the file is the title; the rest is body.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem
    match = DSID_PREFIX_RE.match(stem)
    document_id = match.group(1) if match else stem

    lines = text.split("\n", 1)
    title = lines[0].strip() if lines else ""
    body = lines[1].strip() if len(lines) > 1 else ""

    return BenchmarkDocument(
        document_id=document_id,
        source_type=source_type,
        title=title,
        body=body,
    )


def _document_from_json_record(record: dict[str, Any]) -> BenchmarkDocument:
    """Fallback parser for JSON-format document records."""
    return BenchmarkDocument(
        document_id=record.get("document_id", record.get("id", "")),
        source_type=record.get("source_type", "unknown"),
        title=record.get("title", ""),
        body=record.get("body", record.get("content", "")),
        metadata=record.get("metadata", {}),
    )


def _question_from_record(record: dict[str, Any]) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=record.get("question_id", ""),
        question=record.get("question", ""),
        question_type=record.get("question_type", "unknown"),
        source_types=record.get("source_types", []),
        expected_doc_ids=record.get("expected_doc_ids", []),
        gold_answer=record.get("gold_answer", ""),
        answer_facts=record.get("answer_facts", []),
    )


class EnterpriseRagBenchLoader:
    """Load EnterpriseRAG-Bench data and provide MVP subsets.

    Supports two document layouts:
      1. .txt files with dsid_ prefixed names (official export format)
      2. .jsonl files with JSON records (alternative/generated format)
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.documents_dir = self.root_dir / "docs"
        self.bench_dir = self.root_dir / "bench"
        self.questions_subset_path = self.bench_dir / "questions_subset.jsonl"
        self.extra_questions_path = self.bench_dir / "extra_questions.jsonl"

    def resolved_questions_path(self) -> Path | None:
        """Prefer filtered subset, then full benchmark questions under bench/, then legacy root."""
        for candidate in (
            self.questions_subset_path,
            self.bench_dir / "questions.jsonl",
            self.root_dir / "questions.jsonl",
        ):
            if candidate.exists():
                return candidate
        return None

    @property
    def questions_path(self) -> Path:
        """Backward-compatible path used for downloads and logging."""
        resolved = self.resolved_questions_path()
        return resolved if resolved is not None else self.questions_subset_path

    def ensure_data_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.bench_dir.mkdir(parents=True, exist_ok=True)

    def benchmark_exists(self) -> bool:
        has_questions = self.resolved_questions_path() is not None
        if not has_questions or not self.documents_dir.exists():
            return False
        has_txt = any(self.documents_dir.rglob("*.txt"))
        has_jsonl = any(self.documents_dir.glob("*.jsonl"))
        return has_txt or has_jsonl

    def document_count(self) -> int:
        if not self.documents_dir.exists():
            return 0
        txt_count = sum(1 for _ in self.documents_dir.rglob("*.txt"))
        if txt_count:
            return txt_count
        count = 0
        for jsonl_file in sorted(self.documents_dir.glob("*.jsonl")):
            with jsonl_file.open("r", encoding="utf-8") as file:
                count += sum(1 for line in file if line.strip())
        return count

    def download_questions(self) -> None:
        """Download questions.jsonl and extra_questions.jsonl from GitHub."""
        self.ensure_data_dirs()
        logger.info("Benchmark root: %s", self.root_dir.resolve())
        logger.info("Saving benchmark questions under: %s", self.bench_dir.resolve())
        urls = [
            (QUESTIONS_RAW_URL, self.bench_dir / "questions.jsonl"),
            (EXTRA_QUESTIONS_RAW_URL, self.extra_questions_path),
        ]
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            for url, target in urls:
                if target.exists():
                    logger.info("Questions file already exists: %s", target.resolve())
                    continue
                logger.info("Downloading %s to %s", url, target.resolve())
                response = client.get(url)
                response.raise_for_status()
                target.write_bytes(response.content)
                logger.info("Saved %s bytes to %s", len(response.content), target.resolve())

    def download_release_files(
        self,
        include_all_documents: bool = False,
        release_base_url: str = DEFAULT_RELEASE_BASE,
        document_fraction: float = 1.0,
        download_chunk_size: int = 1024 * 1024,
        extraction_log_every: int = 5000,
        extraction_sleep_seconds: float = 0.0,
        extraction_yield_every: int = 500,
    ) -> None:
        """Download benchmark assets from a GitHub release.

        For MVP this is opt-in because document archives are large (~GB).
        Questions are always downloaded from the repo directly.
        """
        self.download_questions()

        if not include_all_documents:
            logger.info("Skipping document archive download; include_all_documents is false.")
            return

        if not 0 < document_fraction <= 1:
            raise ValueError("document_fraction must be greater than 0 and less than or equal to 1.")

        archive_path = self.root_dir / "all_documents.zip"
        if not archive_path.exists():
            archive_tmp_path = archive_path.with_suffix(".zip.tmp")
            logger.info(
                "Downloading document archive from %s to %s",
                f"{release_base_url}/all_documents.zip",
                archive_path.resolve(),
            )
            bytes_written = 0
            with httpx.Client(timeout=None, follow_redirects=True) as client:
                with client.stream("GET", f"{release_base_url}/all_documents.zip") as response:
                    response.raise_for_status()
                    with archive_tmp_path.open("wb") as file:
                        for chunk in response.iter_bytes(chunk_size=download_chunk_size):
                            if not chunk:
                                continue
                            file.write(chunk)
                            bytes_written += len(chunk)
                            if bytes_written % (100 * 1024 * 1024) < download_chunk_size:
                                logger.info("Downloaded %.1f MB to %s", bytes_written / 1024 / 1024, archive_tmp_path.resolve())
            archive_tmp_path.replace(archive_path)
            logger.info("Saved document archive (%s bytes) to %s", bytes_written, archive_path.resolve())
        else:
            logger.info("Document archive already exists: %s", archive_path.resolve())

        if archive_path.exists():
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "r") as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                members.sort(key=lambda member: member.filename)
                if document_fraction < 1:
                    member_count = max(1, int(len(members) * document_fraction))
                    logger.info(
                        "Extracting %s of %s document files (fraction %.2f) from %s to %s",
                        member_count,
                        len(members),
                        document_fraction,
                        archive_path.resolve(),
                        self.documents_dir.resolve(),
                    )
                    members = members[:member_count]
                else:
                    logger.info(
                        "Extracting all %s document files from %s to %s",
                        len(members),
                        archive_path.resolve(),
                        self.documents_dir.resolve(),
                    )
                extracted = 0
                skipped = 0
                root = self.documents_dir.resolve()
                for member in members:
                    target = (self.documents_dir / member.filename).resolve()
                    try:
                        target.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(f"Archive member would extract outside documents dir: {member.filename}")

                    if target.exists() and target.stat().st_size == member.file_size:
                        skipped += 1
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp_target = target.with_name(f"{target.name}.tmp")
                        with archive.open(member) as source, tmp_target.open("wb") as destination:
                            shutil.copyfileobj(source, destination, length=download_chunk_size)
                        tmp_target.replace(target)
                        extracted += 1

                    processed = extracted + skipped
                    if extraction_log_every > 0 and processed % extraction_log_every == 0:
                        logger.info(
                            "Processed %s/%s archive files into %s (%s extracted, %s skipped)",
                            processed,
                            len(members),
                            self.documents_dir.resolve(),
                            extracted,
                            skipped,
                        )
                    if extraction_sleep_seconds > 0 and extraction_yield_every > 0 and processed % extraction_yield_every == 0:
                        time.sleep(extraction_sleep_seconds)

                logger.info(
                    "Document extraction complete under %s (%s extracted, %s skipped)",
                    self.documents_dir.resolve(),
                    extracted,
                    skipped,
                )

    def _load_txt_documents(self, max_documents: int | None = None) -> list[BenchmarkDocument]:
        """Load .txt documents from the benchmark export (official format)."""
        documents: list[BenchmarkDocument] = []
        for source_dir in sorted(self.documents_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            source_type = source_dir.name
            for txt_file in sorted(source_dir.rglob("*.txt")):
                documents.append(_document_from_txt(txt_file, source_type=source_type))
                if max_documents and len(documents) >= max_documents:
                    return documents
        # Also handle flat layout (no source-type subdirectories)
        for txt_file in sorted(self.documents_dir.glob("*.txt")):
            documents.append(_document_from_txt(txt_file))
            if max_documents and len(documents) >= max_documents:
                return documents
        return documents

    def _load_jsonl_documents(self, max_documents: int | None = None) -> list[BenchmarkDocument]:
        """Fallback: load documents from .jsonl files."""
        documents: list[BenchmarkDocument] = []
        for jsonl_file in sorted(self.documents_dir.glob("*.jsonl")):
            for record in _read_jsonl(jsonl_file):
                documents.append(_document_from_json_record(record))
                if max_documents and len(documents) >= max_documents:
                    return documents
        return documents

    def iter_documents(self) -> Iterator[BenchmarkDocument]:
        if not self.documents_dir.exists():
            return

        has_txt = any(self.documents_dir.rglob("*.txt"))
        if has_txt:
            for source_dir in sorted(self.documents_dir.iterdir()):
                if not source_dir.is_dir():
                    continue
                source_type = source_dir.name
                for txt_file in sorted(source_dir.rglob("*.txt")):
                    yield _document_from_txt(txt_file, source_type=source_type)
            for txt_file in sorted(self.documents_dir.glob("*.txt")):
                yield _document_from_txt(txt_file)
            return

        for jsonl_file in sorted(self.documents_dir.glob("*.jsonl")):
            for record in _read_jsonl(jsonl_file):
                yield _document_from_json_record(record)

    def load_documents(self, max_documents: int | None = None) -> list[BenchmarkDocument]:
        if not self.documents_dir.exists():
            return []
        try:
            next(self.documents_dir.rglob("*.txt"))
            return self._load_txt_documents(max_documents)
        except StopIteration:
            return self._load_jsonl_documents(max_documents)

    def load_questions(
        self,
        max_questions: int | None = None,
        include_extra_questions: bool = False,
    ) -> list[BenchmarkQuestion]:
        path = self.resolved_questions_path()
        if path is None:
            return []

        records = _read_jsonl(path)
        if include_extra_questions and self.extra_questions_path.exists():
            records.extend(_read_jsonl(self.extra_questions_path))

        questions = [_question_from_record(record) for record in records]
        if max_questions:
            return questions[:max_questions]
        return questions

    def load_mvp_subset(
        self,
        max_docs: int = 20000,
        max_questions: int = 208,
    ) -> tuple[list[BenchmarkDocument], list[BenchmarkQuestion]]:
        documents = self.load_documents(max_documents=max_docs)
        questions = self.load_questions(max_questions=max_questions)
        return documents, questions
