from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from src.models import ExperimentResult, ExperimentSpec, Hypothesis


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class ExperimentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()

    async def upsert_hypothesis(self, hypothesis: Hypothesis) -> None:
        payload = hypothesis.model_dump_json()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO hypotheses(id, payload, created_at)
                VALUES (?, ?, ?)
                """,
                (hypothesis.id, payload, hypothesis.created_at.isoformat()),
            )
            await conn.commit()

    async def upsert_experiment(self, experiment: ExperimentSpec) -> None:
        payload = experiment.model_dump_json()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO experiments(id, hypothesis_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.hypothesis_id,
                    payload,
                    experiment.created_at.isoformat(),
                ),
            )
            await conn.commit()

    async def upsert_result(self, result: ExperimentResult) -> None:
        payload = result.model_dump_json()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO results(experiment_id, payload, created_at)
                VALUES (?, ?, ?)
                """,
                (result.experiment_id, payload, result.created_at.isoformat()),
            )
            await conn.commit()

    async def list_hypotheses(self) -> list[Hypothesis]:
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT payload FROM hypotheses")
            rows = await cursor.fetchall()
        return [Hypothesis.model_validate(json.loads(row[0])) for row in rows]

    async def list_experiments(self) -> list[ExperimentSpec]:
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT payload FROM experiments")
            rows = await cursor.fetchall()
        return [ExperimentSpec.model_validate(json.loads(row[0])) for row in rows]

    async def list_results(self) -> list[ExperimentResult]:
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("SELECT payload FROM results")
            rows = await cursor.fetchall()
        return [ExperimentResult.model_validate(json.loads(row[0])) for row in rows]

    async def get_result(self, experiment_id: str) -> ExperimentResult | None:
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "SELECT payload FROM results WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return ExperimentResult.model_validate(json.loads(row[0]))
